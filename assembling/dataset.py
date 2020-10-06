import numpy as np
import os, sys, pathlib

import skvideo.io

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import day_folder, create_day_folder, generate_filename_path,\
    check_datafolder, get_files_with_extension


##############################################
###      Some general signal types         ###
##############################################

class SingleValueTimeSerie:
    
    def __init__(self, signal, dt,
                 t0=0):
        self.t = np.arange(len(signal))*dt+t0
        self.val = signal # value


class ImageTimeSeries:
    
    def __init__(self, folder,
                 dt=None, times=None,
                 extension='.npy',
                 lazy_loading=True,
                 t0=0):
        """
        IMAGES can be initialized
        """
            
        self.BINARY_IMAGES, self.VIDS, self.PICS = None, None, None
        FILES = get_files_with_extension(folder, extension=extension)
        if extension=='.npy':
            self.BINARY_IMAGES = FILES
        elif extension=='.avi':
            self.VIDS= FILES
        elif extension=='.mp4':
            self.VIDS= FILES
        elif extension=='.jpeg':
            self.PICS = FILES
        else:
            print('Extension', extension, ' not recognized !')

        if times is not None:
            self.t = times
        else:
            if dt is None:
                dt = 1
            self.t = np.arange(len(FILES))*dt+t0

        if (self.VIDS is not None) and not lazy_loading:
            # we pre-load the video !
            self.IMAGES = np.empty(0)
            for i, fn in enumerate(self.VIDS[:1]):
                x = skvideo.io.vread(fn)
                if i==0:
                    self.IMAGES = x[:,:,:,0]
                else:
                    self.IMAGES = np.vstack([self.IMAGES, x[:,:,:,0]])
            

    def grab_frame(self, t,
                   force_previous_time=False,
                   verbose=False, with_time=False):

        # finding the image index at that time
        if force_previous_time:
            i0 = np.argmin((t-self.t)**2) # ADAPT HERE
        else:
            i0 = np.argmin((t-self.t)**2)
        if verbose:
            print('found t=', self.t[i0], 'for t=', t)
        # return the image
        if self.BINARY_IMAGES is not None:
            if with_time:
                return self.t[i0], np.load(self.BINARY_IMAGES[i0])
            else:
                return np.load(self.BINARY_IMAGES[i0])
        elif self.IMAGES is not None:
            if with_time:
                return self.t[i0], self.IMAGES[i0]
            else:
                return self.IMAGES[i0]
        # else:
        #     return self.t[i0], self.im[i0]
        
        
##############################################
###               Screen data              ###
##############################################
        
class ScreenData(ImageTimeSeries):

    def __init__(self, datafolder, metadata,
                 NIdaq_trace=None,
                 lazy_loading=True):

        folder = os.path.join(datafolder, 'screen-frames')

        # COMPUTE TIMES FRO METADATA
        
        super().__init__(folder, 
                         extension='.tiff',
                         lazy_loading=lazy_loading)
        
        if NIdaq_trace is not None:
            self.photodiode = SingleValueTimeSerie(NIdaq_trace,
                                                   dt = 1./metadata['NIdaq-acquisition-frequency'])
        else:
            self.photodiode = None #

        # videodata = skvideo.io.vread("video_file_name")  
        # print(videodata.shape)
            
    def grab_frame(self, t):
        pass
    
##############################################
###           Locomotion data              ###
##############################################

class LocomotionData:
    def __init__(self):
        pass

        
def init_locomotion_data(self, data):
    self.Locomotion = {'times':np.arange(data.shape[1])/self.metadata['NIdaq-acquisition-frequency'],
                       'trace':data[Locomotion_NIdaqChannel[0],:]+\
                       data[Locomotion_NIdaqChannel[1],:]}

##############################################
###           Pupil data                   ###
##############################################

class PupilData(ImageTimeSeries):

    def __init__(self, datafolder, metadata,
                 dt=None, times=None,
                 t0=0,
                 compressed_version=True):

        times = np.load(os.path.join(datafolder, 'FaceCamera-times.npy'))

        if compressed_version:
            folder = os.path.join(datafolder, 'FaceCamera-compressed')
            extension, lazy_loading ='.mp4', False
        else:
            folder = os.path.join(datafolder, 'FaceCamera-imgs')
            extension='.npy',
            
        super().__init__(folder, times=times,
                         extension=extension,
                         lazy_loading=lazy_loading)

        if os.path.isfile(os.path.join(datafolder,'pupil-data.npy')):
            data = np.load(os.path.join(datafolder,'pupil-data.npy'),
                           allow_pickle=True).item()
            # we add the diameter
            setattr(self, 't', data['times'])
            setattr(self, 'diameter', np.sqrt(data['sx-corrected']*data['sy-corrected']))
        else:
            # we fill with zeros
            setattr(self, 'diameter', 0*times)
            
    
##############################################
###           Electrophy data              ###
##############################################
# def init_electrophy_data(self, data):
#     self.Electrophy={'times':np.arange(data.shape[1])/self.metadata['NIdaq-acquisition-frequency'],
#                      'trace':data[Electrophy_NIdaqChannel,:]}


        
##############################################
###         Multimodal dataset             ###
##############################################

MODALITIES = ['Screen', 'Locomotion', 'Electrophy', 'Pupil','Calcium']

class Dataset:
    
    def __init__(self, datafolder,
                 Electrophy_NIdaqChannel=0, # switch to 
                 Locomotion_NIdaqChannels=[1,2],
                 compressed_version=True,
                 modalities=MODALITIES):
        """

        by default we take all modalities, you can restrict them using "modalities"
        """
        for key in modalities:
            setattr(self, key, None) # all modalities to None by default
            
        self.datafolder = datafolder
        self.metadata = check_datafolder(self.datafolder, modalities)

        if self.metadata['NIdaq']: # loading the NIdaq data only once
            data = np.load(os.path.join(self.datafolder, 'NIdaq.npy'))
            self.NIdaq_Tstart = np.load(os.path.join(self.datafolder, 'NIdaq.start.npy'))[0]
        if self.metadata['VisualStim'] and ('Screen' in modalities):
            self.Screen = ScreenData(self.datafolder, self.metadata,
                                     NIdaq_trace=data[0,:])
        # if self.metadata['NIdaq'] and ('Locomotion' in modalities):
        #     self.Locomotion = LocomotionData(self.datafolder, metadata)
        # if self.metadata['NIdaq'] and ('Electrophy' in modalities):
        #     self.Electrophyinit_electrophy_data(data)
        if self.metadata['FaceCamera'] and ('Pupil' in modalities):
            self.Pupil = PupilData(self.datafolder, self.metadata,
                                   compressed_version=compressed_version)

        if self.Screen.photodiode is not None:
            self.realign_from_photodiode()

            
    def realign_from_photodiode(self, debug=False, verbose=True):

        if verbose:
            print('---> Realigning data with respect to photodiode signal [...] ')

        if debug:
            from datavyz import ges as ge

        success = True
        
        # extract parameters
        dt = 1./self.metadata['NIdaq-acquisition-frequency']
        tlim, tnew = [0, self.Screen.photodiode.t[-1]], 0

        t0 = self.metadata['time_start'][0]
        length = self.metadata['presentation-duration']+self.metadata['presentation-interstim-period']
        npulses = int(self.metadata['presentation-duration'])
        self.metadata['time_start_realigned'] = []
        Nepisodes = np.sum(self.metadata['time_start']<tlim[1])
        for i in range(Nepisodes):
            cond = (self.Screen.photodiode.t>=t0-.3) & (self.Screen.photodiode.t<=t0+length)
            try:
                tnew, integral, threshold = find_onset_time(self.Screen.photodiode.t[cond]-t0,
                                                            self.Screen.photodiode.val[cond], npulses)
                if debug and ((i<3) or (i>Nepisodes-3)):
                    ge.plot(self.Screen.photodiode.t[cond], self.Screen.photodiode.val[cond])
                    ge.plot(self.Screen.photodiode.t[cond], Y=[integral, integral*0+threshold])
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
        fn = '/home/yann/DATA/2020_09_11/13-40-10/'

        dataset = Dataset(fn, compressed_version=True)

        frame = dataset.Pupil.grab_frame(0, verbose=True)
        from datavyz import ges
        ges.image(frame)
        ges.show()
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
