import numpy as np
import os, sys, pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import day_folder, create_day_folder, generate_filename_path, check_datafolder

def get_list_of_datafiles(data_folder):

    list_of_folders = [os.path.join(day_folder(data_folder), d)\
                       for d in os.listdir(day_folder(data_folder)) if os.path.isdir(os.path.join(day_folder(data_folder), d))]
    
    return [os.path.join(d,'visual-stim.npz')\
              for d in list_of_folders if os.path.isfile(os.path.join(d,'visual-stim.npz'))]

def last_datafile(data_folder):
    return get_list_of_datafiles(data_folder)[-1]


class Dataset:
    
    def __init__(self, datafolder,
                 modalities=['Screen', 'Locomotion', 'Electrophy', 'Pupil','Calcium']):
        
        for key in modalities:
            setattr(self, key, None) # all modalities to None by default
            
        self.datafolder = datafolder
        self.metadata = check_datafolder(self.datafolder, modalities)

        if self.metadata['NIdaq']: # loading the NIdaq data only once
            data = np.load(os.path.join(self.datafolder, 'NIdaq.npy'))
            self.NIdaq_Tstart = np.load(os.path.join(self.datafolder, 'NIdaq.start.npy'))[0]
            
        if self.metadata['VisualStim'] and ('Screen' in modalities):
            self.init_screen_data(data)
        if self.metadata['NIdaq'] and ('Locomotion' in modalities):
            self.init_locomotion_data(data)
        if self.metadata['NIdaq'] and ('Electrophy' in modalities):
            self.init_electrophy_data(data)

        if self.metadata['FaceCamera'] and ('Pupil' in modalities):
            self.init_pupil_data()
            
            
    ##############################################
    ###           Locomotion data              ###
    ##############################################
    def init_locomotion_data(self, data):
        self.Locomotion = {'times':np.arange(data.shape[1])/self.metadata['NIdaq-acquisition-frequency'],
                           'trace':data[1,:]}
        
    ##############################################
    ###           Pupil data                   ###
    ##############################################
    def init_pupil_data(self):
        self.Pupil = {}
        if os.path.isfile(os.path.join(self.datafolder,'pupil-data.npy')):
            data = np.load(os.path.join(self.datafolder,'pupil-data.npy'),
                           allow_pickle=True).item()
            self.Pupil = {'times':data['times'],
                          'imgs':data['filenames'],
                          'diameter':np.sqrt(data['sx-corrected']*data['sy-corrected'])}
        else:
            self.Pupil['times'] = np.load(os.path.join(self.datafolder, 'FaceCamera-times.npy'))
            self.Pupil['imgs'] = np.array(sorted(os.listdir(os.path.join(self.datafolder,
                                                                         'FaceCamera-imgs'))))
            self.Pupil['diameter'] = 0.*self.Pupil['times']

    
    ##############################################
    ###           Electrophy data              ###
    ##############################################
    def init_electrophy_data(self, data):
        self.Electrophy={'times':np.arange(data.shape[1])/self.metadata['NIdaq-acquisition-frequency'],
                         'trace':data[0,:]}


    ##############################################
    ###               Screen data              ###
    ##############################################
    def init_screen_data(self, data):
        
        self.Screen = {'times':np.arange(data.shape[1])/self.metadata['NIdaq-acquisition-frequency'],
                       'photodiode':data[0,:]}
        # Screen (sample) images displayed on the screen
        self.Screen['imgs'] = os.listdir(os.path.join(self.datafolder,'screen-frames'))
        # realigning according to photodiode signal
        self.realign_from_photodiode()

        # adding stimulus metadata
        data = np.load(os.path.join(self.datafolder,'visual-stim.npy'), allow_pickle=True).item()
        print(data)
        self.metadata = {**self.metadata ,**data}
        
        
    def realign_from_photodiode(self, debug=False, verbose=True):

        if verbose:
            print('---> Realigning data with respect to photodiode signal [...] ')

        if debug:
            from datavyz import ges as ge

        success = True
        
        # extract parameters
        dt = 1./self.metadata['NIdaq-acquisition-frequency']
        tlim, tnew = [0, self.Screen['times'][-1]], 0

        t0 = self.metadata['time_start'][0]
        length = self.metadata['presentation-duration']+self.metadata['presentation-interstim-period']
        npulses = int(self.metadata['presentation-duration'])
        self.metadata['time_start_realigned'] = []
        Nepisodes = np.sum(self.metadata['time_start']<tlim[1])
        for i in range(Nepisodes):
            cond = (self.Screen['times']>=t0-.3) & (self.Screen['times']<=t0+length)
            try:
                tnew, integral, threshold = find_onset_time(self.Screen['times'][cond]-t0, self.Screen['photodiode'][cond], npulses)
                if debug and ((i<3) or (i>Nepisodes-3)):
                    ge.plot(self.Screen['times'][cond], self.Screen['photodiode'][cond])
                    ge.plot(self.Screen['times'][cond], Y=[integral, integral*0+threshold])
                    ge.show()
            except Exception:
                success = False # one exception is enough to make it fail
            t0+=tnew
            self.metadata['time_start_realigned'].append(t0)
            t0+=length

        if verbose:
            if success:
                print('[ok]          --> succesfully realigned')
            else:
                print('[X]          --> realignement failed')
        if success:
            self.metadata['time_start_realigned'] = np.array(self.metadata['time_start_realigned'])
            self.metadata['time_stop_realigned'] = self.metadata['time_start_realigned']+\
                self.metadata['presentation-duration']
        else:
            self.metadata['time_start_realigned'] = np.array([])
            self.metadata['time_stop_realigned'] = np.array([])

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
        
        
    
# def get_multimodal_dataset(filename, image_sampling_number=10):

#     # -- VISUAL STIMULATION -- #
#     # presentation metadata
#     data = load_dict(filename)
#     # presented images (snapshot of last frame)
#     data['images'] = []
#     i=1
#     while os.path.isfile(filename.replace('visual-stim.npz', 'frame%i.tiff' %i)) or\
#           os.path.isfile(filename.replace('visual-stim.npz', 'frame0%i.tiff' %i)) or\
#           os.path.isfile(filename.replace('visual-stim.npz', 'frame00%i.tiff' %i)):
#         if os.path.isfile(filename.replace('visual-stim.npz', 'frame%i.tiff' %i)):
#             im = Image.open(filename.replace('visual-stim.npz', 'frame%i.tiff' %i))
#         elif os.path.isfile(filename.replace('visual-stim.npz', 'frame0%i.tiff' %i)):
#             im = Image.open(filename.replace('visual-stim.npz', 'frame0%i.tiff' %i))
#         elif os.path.isfile(filename.replace('visual-stim.npz', 'frame00%i.tiff' %i)):
#             im = Image.open(filename.replace('visual-stim.npz', 'frame00%i.tiff' %i))
#         data['images'].append(np.rot90(np.array(im).mean(axis=2), k=3))
#         i+=1

#     # -- NI DAQ ACQUISTION -- #
#     if os.path.isfile(filename.replace('visual-stim.npz', 'NIdaq.npy')):
#         data['NIdaq'] = np.load(filename.replace('visual-stim.npz', 'NIdaq.npy'))
#         data['NIdaq-Tstart'] = np.load(filename.replace('visual-stim.npz', 'NIdaq.start.npy'))[0]
#         data['t'] = np.arange(data['NIdaq'].shape[1])/data['NIdaq-acquisition-frequency']
        
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
    
    # return data

def find_onset_time(t, photodiode_signal, npulses,
                    time_for_threshold=5e-3):
    """
    the threshold of integral increase corresponds to spending X-ms at half the maximum
    """
    H, bins = np.histogram(photodiode_signal, bins=100)
    baseline = bins[np.argmax(H)]

    integral = np.cumsum(photodiode_signal-baseline)*(t[1]-t[0])

    threshold = time_for_threshold*np.max(photodiode_signal)
    t0 = t[np.argwhere(integral>threshold)[0][0]]
    return t0-time_for_threshold, integral, threshold


        
if __name__=='__main__':

    if sys.argv[-1]=='photodiode':
        fn = '/home/yann/DATA/2020_09_23/16-40-54/'
        # dataset = Dataset(fn)

        data = np.load(os.path.join(fn, 'NIdaq.npy'))
        import matplotlib.pylab as plt
        H, bins = np.histogram(data[0,:10000], bins=50)
        baseline = bins[np.argmax(H)]
        plt.figure()
        plt.hist(data[0,:10000], bins=50)
        plt.figure()
        plt.plot(np.cumsum(data[0,:][:10000]-baseline))
        plt.figure()
        plt.plot(data[0,:][:10000])
        plt.plot(data[0,:][:10000]*0+baseline)
        # plt.plot(data['NIdaq'][0][:10000])
        plt.show()
        
    else:
        fn = '/home/yann/DATA/2020_09_23/16-40-54/'
        dataset = Dataset(fn)
        
        # import json
        # DFFN = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'master', 'data-folder.json') # DATA-FOLDER-FILENAME
        # with open(DFFN, 'r') as fp:
        #     df = json.load(fp)['folder']
        # data = get_multimodal_dataset(last_datafile(df))
        # transform_into_realigned_episodes(data, debug=True)
        
        # transform_into_realigned_episodes(data)
        # print(len(data['time_start_realigned']), len(data['NIdaq_realigned']))

        # print('max blank time of FaceCamera: %.0f ms' % (1e3*np.max(np.diff(data['FaceCamera-times']))))
        # import matplotlib.pylab as plt
        # plt.hist(1e3*np.diff(data['FaceCamera-times']))
        # plt.show()
