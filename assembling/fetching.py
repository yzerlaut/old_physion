import numpy as np
import os, sys, pathlib
from PIL import Image
from scipy.optimize import minimize

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import day_folder, create_day_folder, generate_filename_path, load_dict

def get_list_of_datafiles(data_folder):

    list_of_folders = [os.path.join(day_folder(data_folder), d)\
                       for d in os.listdir(day_folder(data_folder)) if os.path.isdir(os.path.join(day_folder(data_folder), d))]
    
    return [os.path.join(d,'visual-stim.npz')\
              for d in list_of_folders if os.path.isfile(os.path.join(d,'visual-stim.npz'))]

def last_datafile(data_folder):
    return get_list_of_datafiles(data_folder)[-1]

def get_multimodal_dataset(filename, image_sampling_number=10):

    # -- VISUAL STIMULATION -- #
    # presentation metadata
    data = load_dict(filename)
    # presented images (snapshot of last frame)
    data['images'] = []
    i=1
    while os.path.isfile(filename.replace('visual-stim.npz', 'frame%i.tiff' %i)) or\
          os.path.isfile(filename.replace('visual-stim.npz', 'frame0%i.tiff' %i)) or\
          os.path.isfile(filename.replace('visual-stim.npz', 'frame00%i.tiff' %i)):
        if os.path.isfile(filename.replace('visual-stim.npz', 'frame%i.tiff' %i)):
            im = Image.open(filename.replace('visual-stim.npz', 'frame%i.tiff' %i))
        elif os.path.isfile(filename.replace('visual-stim.npz', 'frame0%i.tiff' %i)):
            im = Image.open(filename.replace('visual-stim.npz', 'frame0%i.tiff' %i))
        elif os.path.isfile(filename.replace('visual-stim.npz', 'frame00%i.tiff' %i)):
            im = Image.open(filename.replace('visual-stim.npz', 'frame00%i.tiff' %i))
        data['images'].append(np.rot90(np.array(im).mean(axis=2), k=3))
        i+=1

    # -- NI DAQ ACQUISTION -- #
    if os.path.isfile(filename.replace('visual-stim.npz', 'NIdaq.npy')):
        data['NIdaq'] = np.load(filename.replace('visual-stim.npz', 'NIdaq.npy'))
        data['t'] = np.arange(data['NIdaq'].shape[1])/data['NIdaq-acquisition-frequency']
        
    # -- PTGREY CAMERA -- #
    if (image_sampling_number is not None) and os.path.isfile(filename.replace('visual-stim.npz', 'camera-times.npy')):
        data['camera-times'] = np.load(filename.replace('visual-stim.npz', 'camera-times.npy'))
        data['camera-imgs'] = []
        images_ID = np.linspace(1, len(data['camera-times']), image_sampling_number, dtype=int)
        print(len(images_ID))    
        for i in images_ID:
            data['camera-imgs'].append(np.rot90(np.load(filename.replace('visual-stim.npz', 'camera-imgs'+os.path.sep+'%i.npy' % i)),k=3))

    # --  CALCIUM IMAGING -- #
    
    return data

# import 

##################################
## functions to fit for the realignement
def heaviside(x):
    return (np.sign(x+1e-12)+1.)/2. # 1e-12 to avoid the 1/2 value
def step(x, x1, x2):
    return heaviside(x-x1)*heaviside(x2-x)
def resp_function(t, t1, t2, T):
    return heaviside(t-t1)*(1-np.exp(-(t-t1)/T))+heaviside(t-t2)*(np.exp(-(t-t2)/T)-1)
def waveform(t, T, n=4):
    output, k = 0*t, 0
    for i in np.arange(2):
        if k<=n:
            output += resp_function(t, 0.5*i, 0.5*i+0.2, T)
            k+=1
    for i in np.arange(1, int(t.max())):
        if k<=n:
            output += resp_function(t, i, i+0.2, T)
            k+=1
    return output


def find_onset_time(t, photodiode_signal, npulses):
    def to_minimize(x):
        return np.abs(photodiode_signal-x[2]-x[3]*waveform(t-x[0], x[1], n=npulses)).sum()
    try:
        res = minimize(to_minimize,
                       [0.001, 0.02, 0.1, 0.3],
                       method = 'SLSQP', # 'L-BFGS-B',# TNC, SLSQP, Powel
                       bounds=[(-.1,0.5), (0.002, 0.1), (0.001, 0.2), (0.1, 0.4)])
        return res.x
    except ValueError:
        return None


def transform_into_realigned_episodes(data, debug=False):

    print('... Realigning data [...] ')

    if debug:
        from datavyz import ges as ge
        
    # extract parameters
    dt = 1./data['NIdaq-acquisition-frequency']
    tlim = [0, data['time_stop'][-1]]
    
    if 'camera-times' in data:
        tlim = [0, data['camera-times'][-1]]
    if 'NIdaq' in data:
        tlim = [0, data['NIdaq'].shape[1]*dt] # overrides the obove
    data['tlim'] = tlim
    
    t0 = data['time_start'][0]
    length = data['presentation-duration']+data['presentation-interstim-period']
    npulses = int(data['presentation-duration'])
    data['time_start_realigned'] = []
    Nepisodes = np.sum(data['time_start']<tlim[1])
    for i in range(Nepisodes):
        cond = (data['t']>=t0-.3) & (data['t']<=t0+length)
        x = find_onset_time(data['t'][cond]-t0, data['NIdaq'][0,cond], npulses)
        if debug and (i>5) and (i<15):
            ge.plot(data['t'][cond], Y=[data['NIdaq'][0,cond], x[2]+x[3]*waveform(data['t'][cond]-t0-x[0], x[1], npulses)])
            ge.show()
        if x is not None:
            t0+=x[0]
            data['time_start_realigned'].append(t0)
            t0+=length
    data['time_start_realigned'] = np.array(data['time_start_realigned'])
    
    print('[ok] Data realigned ')
    
    l = data['presentation-interstim-period']+data['presentation-duration']
    data['t_realigned'] = -data['presentation-interstim-period']/2+np.arange(int(l/dt))*dt
    data['NIdaq_realigned'] = []
    for i, t0 in enumerate(data['time_start_realigned']):
        cond = (data['t']>(t0+data['t_realigned'][0]))
        data['NIdaq_realigned'].append(data['NIdaq'][:,cond][:,:len(data['t_realigned'])])
    print('[ok] Data transformed to episodes ')
        
if __name__=='__main__':

    import tempfile
    data = get_multimodal_dataset(last_datafile(tempfile.gettempdir()))
    # transform_into_realigned_episodes(data, debug=True)
    # transform_into_realigned_episodes(data)
    # print(len(data['time_start_realigned']), len(data['NIdaq_realigned']))

    
    print('max blank time of FaceCamera: %.0f ms' % (1e3*np.max(np.diff(data['camera-times']))))
