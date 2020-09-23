import numpy as np
import os, sys, pathlib
from PIL import Image

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import day_folder, create_day_folder, generate_filename_path, load_dict

def get_list_of_datafiles(data_folder):

    list_of_folders = [os.path.join(day_folder(data_folder), d)\
                       for d in os.listdir(day_folder(data_folder)) if os.path.isdir(os.path.join(day_folder(data_folder), d))]
    
    return [os.path.join(d,'visual-stim.npz')\
              for d in list_of_folders if os.path.isfile(os.path.join(d,'visual-stim.npz'))]

def last_datafile(data_folder):
    return get_list_of_datafiles(data_folder)[-1]


class Dataset:
    
    def __init__(self, datafolder):
        
        self.datafolder = datafolder

        self.Screen, self.Locomotion, self.Electrophy,\
            self.Pupil, self.Calcium = None, None, None, None, None

        # metadata should always be there
        self.metadata = np.load(os.path.join(self.datafolder,'metadata.npy'),
                                allow_pickle=True).item()
        print(self.metadata)
        # also the NIdaq data should always be there !
        data = np.load(os.path.join(self.datafolder,'NIdaq.npy'))
        self.NIdaq_Tstart = np.load(os.path.join(self.datafolder, 'NIdaq.start.npy'))[0]

        self.init_screen_data(data)
        # if ('with-VisualStim' in self.metadata) and self.metadata['with-VisualStim']:
        #     self.init_screen_data()
        # if self.metadata['with-Locomotion']:
        #     self.init_locomotion_data(data)
        # if self.metadata['with-Pupil']:
        #     self.init_pupil_data()
        # if self.metadata['with-Electrophy']:
        #     self.init_electrophy_data(data)

    def init_locomotion_data(self, data):
        self.Locomotion = {'times':np.arange(data.shape[1])/self.metadata['NIdaq-acquisition-frequency'],
                           'trace':data[1,:]}
        
    def init_pupil_data(self, data):
        pass
    
    def init_electrophy_data(self, data):
        self.Eletrophy = {'times':np.arange(data.shape[1])/self.metadata['NIdaq-acquisition-frequency'],
                          'trace':data[0,:]}

        
    def init_screen_data(self, data):
        
        self.Screen = {'times':np.arange(data.shape[1])/self.metadata['NIdaq-acquisition-frequency'],
                       'photodiode':data[0,:]}
        # Screen images

        # (sample) images displayed on the screen
        self.Screen['imgs'] = []
        i=1
        while os.path.isfile(os.path.join(self.datafolder,'screen-frames', 'frame%i.tiff' %i)) or\
              os.path.isfile(os.path.join(self.datafolder,'screen-frames', 'frame0%i.tiff' %i)) or\
              os.path.isfile(os.path.join(self.datafolder,'screen-frames', 'frame00%i.tiff' %i)):
            if os.path.isfile(os.path.join(self.datafolder,'screen-frames', 'frame%i.tiff' %i)):
                fn = os.path.join(self.datafolder,'screen-frames', 'frame%i.tiff' %i)
            elif os.path.isfile(os.path.join(self.datafolder,'screen-frames', 'frame0%i.tiff' %i)):
                fn = os.path.join(self.datafolder,'screen-frames', 'frame0%i.tiff' %i)
            elif os.path.isfile(os.path.join(self.datafolder,'screen-frames', 'frame00%i.tiff' %i)):
                fn = os.path.join(self.datafolder,'screen-frames', 'frame00%i.tiff' %i)
            self.Screen['imgs'].append(fn)
            i+=1

        self.realigned_from_photodiode(self, debug=False, verbose=False)

        
    def realigned_from_photodiode(self, debug=False, verbose=True):

        if verbose:
            print('... Realigning data [...] ')

        if debug:
            from datavyz import ges as ge

        # extract parameters
        dt = 1./self.metadata['NIdaq-acquisition-frequency']
        tlim = [0, self.Screen['times'][-1]]

        t0 = self.metadata['time_start'][0]
        length = self.metadata['presentation-duration']+self.metadata['presentation-interstim-period']
        npulses = int(self.metadata['presentation-duration'])
        self.metadata['time_start_realigned'] = []
        Nepisodes = np.sum(self.metadata['time_start']<tlim[1])
        for i in range(Nepisodes):
            cond = (self.Screen['times']>=t0-.3) & (self.Screen['times']<=t0+length)
            tnew, integral, threshold = find_onset_time(self.Screen['times'][cond]-t0, self.Screen['photodiode'][cond], npulses)
            if debug and ((i<3) or (i>Nepisodes-3)):
                ge.plot(self.Screen['times'][cond], self.Screen['photodiode'][cond])
                ge.plot(self.Screen['times'][cond], Y=[integral, integral*0+threshold])
                ge.show()
            t0+=tnew
            self.metadata['time_start_realigned'].append(t0)
            t0+=length
        self.metadata['time_start_realigned'] = np.array(self.metadata['time_start_realigned'])
        self.metadata['time_stop_realigned'] = self.metadata['time_start_realigned']+self.metadata['presentation-duration']

        if verbose:
            print('[ok] Data realigned')

        # l = self.metadata['presentation-interstim-period']+self.metadata['presentation-duration']
        # self.metadata['t_realigned'] = -self.metadata['presentation-interstim-period']/2+np.arange(int(l/dt))*dt
        # self.metadata['NIdaq_realigned'] = []
        # for i, t0 in enumerate(self.metadata['time_start_realigned']):
        #     cond = (self.Screen['times']>(t0+self.metadata['t_realigned'][0]))
        #     self.metadata['NIdaq_realigned'].append(self.metadata['NIdaq'][:,cond][:,:len(self.metadata['t_realigned'])])
        # if verbose:
        #     print('[ok] Data transformed to episodes')
    
    # # -- NI DAQ ACQUISTION -- #
    # if os.path.isfile(filename.replace('visual-stim.npz', 'NIdaq.npy')):
    #     data['NIdaq'] = np.load(filename.replace('visual-stim.npz', 'NIdaq.npy'))
    #     data['NIdaq-Tstart'] = np.load(filename.replace('visual-stim.npz', 'NIdaq.start.npy'))[0]
    #     data['t'] = np.arange(data['NIdaq'].shape[1])/data['NIdaq-acquisition-frequency']
        
        
    
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
        data['NIdaq-Tstart'] = np.load(filename.replace('visual-stim.npz', 'NIdaq.start.npy'))[0]
        data['t'] = np.arange(data['NIdaq'].shape[1])/data['NIdaq-acquisition-frequency']
        
    # -- PTGREY CAMERA -- #
    # if (image_sampling_number is not None) and os.path.isfile(filename.replace('visual-stim.npz', 'FaceCamera-times.npy')):
    #     data['FaceCamera-times'] = np.load(filename.replace('visual-stim.npz', 'FaceCamera-times.npy'))
    #     if 'NIdaq-Tstart' in data:
    #         data['FaceCamera-times'] -= data['NIdaq-Tstart']
    #     data['FaceCamera-imgs'] = []
    #     images_ID = np.linspace(0, len(data['FaceCamera-times'])-1, image_sampling_number, dtype=int)
    #     for i in images_ID:
    #         data['FaceCamera-imgs'].append(np.rot90(np.load(filename.replace('visual-stim.npz', 'FaceCamera-imgs'+os.path.sep+'%i.npy' % i)),k=3))

    # --  CALCIUM IMAGING -- #
    
    return data

def find_onset_time(t, photodiode_signal, npulses,
                    time_for_threshold=5e-3):

    H, bins = np.histogram(photodiode_signal, bins=100)
    baseline = bins[np.argmax(H)]

    integral = np.cumsum(photodiode_signal-baseline)*(t[1]-t[0])

    threshold = time_for_threshold*np.max(photodiode_signal)
    t0 = t[np.argwhere(integral>threshold)[0][0]]
    return t0, integral, threshold

def transform_into_realigned_episodes(dataset, debug=False, verbose=True):

    if verbose:
        print('... Realigning data [...] ')

    if debug:
        from datavyz import ges as ge
        
    # extract parameters
    dt = 1./dataset.data['NIdaq-acquisition-frequency']
    
    tlim = [0, dataset.Screen['times'][-1]]
        
    t0 = dataset.metadata['time_start'][0]
    length = data['presentation-duration']+data['presentation-interstim-period']
    npulses = int(data['presentation-duration'])
    data['time_start_realigned'] = []
    Nepisodes = np.sum(data['time_start']<tlim[1])
    for i in range(Nepisodes):
        cond = (data['t']>=t0-.3) & (data['t']<=t0+length)
        tnew, integral, threshold = find_onset_time(data['t'][cond]-t0, data['NIdaq'][0,cond], npulses)
        if debug and ((i<3) or (i>Nepisodes-3)):
            ge.plot(data['t'][cond], data['NIdaq'][0,cond])
            ge.plot(data['t'][cond], Y=[integral, integral*0+threshold])
            ge.show()
        t0+=tnew
        data['time_start_realigned'].append(t0)
        t0+=length
    data['time_start_realigned'] = np.array(data['time_start_realigned'])
    
    if verbose:
        print('[ok] Data realigned')
    
    l = data['presentation-interstim-period']+data['presentation-duration']
    data['t_realigned'] = -data['presentation-interstim-period']/2+np.arange(int(l/dt))*dt
    data['NIdaq_realigned'] = []
    for i, t0 in enumerate(data['time_start_realigned']):
        cond = (data['t']>(t0+data['t_realigned'][0]))
        data['NIdaq_realigned'].append(data['NIdaq'][:,cond][:,:len(data['t_realigned'])])
    if verbose:
        print('[ok] Data transformed to episodes')

        
if __name__=='__main__':

    if sys.argv[-1]=='photodiode':
        fn = '/home/yann/DATA/2020_09_23/16-40-54/'
        dataset = Dataset(fn)
        dataset.realigned_from_photodiode()
        print(dataset.metadata['time_start_realigned'], dataset.metadata['time_start'])
        # data = get_multimodal_dataset(fn)
        # transform_into_realigned_episodes(data, debug=True)

        # import matplotlib.pylab as plt
        # H, bins = np.histogram(data['NIdaq'][0], bins=50)
        # baseline = bins[np.argmax(H)]
        # # plt.hist(data['NIdaq'][0], bins=50)
        # plt.plot(np.cumsum(data['NIdaq'][0][:10000]-baseline))
        # # plt.plot(np.cumsum(data['NIdaq'][0][:10000]-data['NIdaq'][0][:1000].mean()))
        # # plt.plot(data['NIdaq'][0][:10000])
        # # plt.plot(data['NIdaq'][0][:10000]*0+baseline)
        # plt.show()
        
    else:
        import json
        DFFN = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'master', 'data-folder.json') # DATA-FOLDER-FILENAME
        with open(DFFN, 'r') as fp:
            df = json.load(fp)['folder']
        data = get_multimodal_dataset(last_datafile(df))
        transform_into_realigned_episodes(data, debug=True)
        # transform_into_realigned_episodes(data)
        # print(len(data['time_start_realigned']), len(data['NIdaq_realigned']))

        # print('max blank time of FaceCamera: %.0f ms' % (1e3*np.max(np.diff(data['FaceCamera-times']))))
        # import matplotlib.pylab as plt
        # plt.hist(1e3*np.diff(data['FaceCamera-times']))
        # plt.show()
