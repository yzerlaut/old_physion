import numpy as np
import os, sys, pathlib

import skvideo.io
from PIL import Image

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import day_folder, create_day_folder, generate_filename_path,\
    check_datafolder, get_files_with_extension


##############################################
###      Some general signal types         ###
##############################################

class SingleValueTimeSerie:
    
    def __init__(self, signal, dt=1.,
                 times = None,
                 t0=0):
        if times is not None:
            self.t = times
        else:
            self.t = np.arange(len(signal))*dt+t0
        self.val = signal # value


class ImageTimeSeries:
    
    def __init__(self, folder,
                 dt=None,
                 times=None, frame_sampling=None,
                 extension='.npy',
                 lazy_loading=True,
                 t0=0):
        """
        IMAGES can be initialized
        """
            
        self.BINARY_IMAGES, self.VIDS, self.PICS = None, None, None
        self.IMAGES = None
        
        FILES = sorted(get_files_with_extension(folder,
                                                extension=extension))

        ### ASSOCIATING A TEMPORAL SAMPLING
        # ---------------------------------------------
        if times is not None:
            self.t = times
        else:
            if dt is None:
                dt = 1
            self.t = np.arange(len(FILES))*dt+t0

        ### DEALING WITH FRAME SAMPLING
        # ---------------------------------------------
        # you can specify a specific subsampling here !
        if frame_sampling is not None and times is None:
            print('/!\ Need to pass the specific times if you subsample the frames /!\ ')
        elif frame_sampling is None:
            frame_sampling = np.arange(len(self.t))

        ### LOADING FRAMES
        # ---------------------------------------------
        if extension=='.npy':
            self.BINARY_IMAGES = [FILES[f] for f in frame_sampling]
        elif extension=='.avi':
            self.VIDS= FILES
        elif extension=='.mp4':
            self.VIDS= FILES
        elif extension=='.jpeg':
            self.PICS = FILES
        elif extension=='.tiff':
            self.IMAGES = []
            for fn in FILES:
                self.IMAGES.append(np.array(Image.open(fn)))
        else:
            print('Extension', extension, ' not recognized !')

        
        if (self.VIDS is not None):
            if lazy_loading:
                # we just make a map between indices and videos/frames
                self.index_frame_map = []
                for i, fn in enumerate(self.VIDS):
                    s = fn.split('imgs-')[1].replace(extension, '').split('-')
                    i0, i1 = int(s[0]), int(s[1])
                    for i, iframe in enumerate(np.arange(i0, i1+1)):
                        if iframe in frame_sampling:
                            self.index_frame_map.append([fn, i])
            else:
                print('Loading the full-set of videos [...]')
                # we pre-load the video !
                self.IMAGES = []
                for i, fn in enumerate(self.VIDS):
                    s = fn.split('imgs-')[1].replace(extension, '').split('-')
                    i0, i1 = int(s[0]), int(s[1])
                    x = skvideo.io.vread(fn)
                    for i, iframe in enumerate(np.arange(i0, i1+1)):
                        if iframe in frame_sampling:
                            self.IMAGES.append(x[i,:,:,0])

    def grab_frame(self, t,
                   force_previous_time=False,
                   verbose=False, with_time=False):

        # finding the image index at that time
        if force_previous_time:
            i0 = np.arange(len(self.t))[self.t<=t][-1]
        else:
            i0 = np.argmin((t-self.t)**2)
        if verbose:
            print('found t=', self.t[i0], 'for t=', t)
            
        # return the image
        if self.IMAGES is not None:
            im = self.IMAGES[i0]
        elif self.BINARY_IMAGES is not None:
            im = np.load(self.BINARY_IMAGES[i0])
        else:
            # we have loaded it using the "lazy_loading" option
            fn, index = self.index_frame_map[i0]
            x = skvideo.io.vread(fn)
            im = x[index,:,:,0]
            
        if with_time:
            return self.t[i0], im
        else:
            return im
        
##############################################
###               Screen data              ###
##############################################
        
class ScreenData(ImageTimeSeries):

    def __init__(self, datafolder, metadata,
                 NIdaq_trace=None):

        folder = os.path.join(datafolder, 'screen-frames')

        self.set_times_from_metadata(metadata)
              
        super().__init__(folder, 
                         extension='.tiff',
                         times = self.t)
        
        if NIdaq_trace is not None:
            self.photodiode = SingleValueTimeSerie(NIdaq_trace,
                                                   dt = 1./metadata['NIdaq-acquisition-frequency'])
        else:
            self.photodiode = None #

    def set_times_from_metadata(self, metadata,
                                realigned=False):
        """
        Adding a custom procedure to set the times of the frames
        (so that you can recall it after re-alignement)
        """
        if realigned:
            time_start = metadata['time_start_realigned']
        else:
            time_start = metadata['time_start']
            
        times = [0] # starting with the pre-frame
        for ts in time_start:
            times = times + [ts, ts+metadata['presentation-duration']]
        times.append(1e10) # adding a last point very far in the future
        self.t = np.array(times)
            
##############################################
###           Locomotion data              ###
##############################################

class LocomotionData:
    """ NOT USED YET "SingleValueTimeSerie" is enough ! """
    def __init__(self, *args):
        super().__init__(*args)
    
        
def init_locomotion_data(self, data):
    self.Locomotion = {'times':np.arange(data.shape[1])/self.metadata['NIdaq-acquisition-frequency'],
                       'trace':data[Locomotion_NIdaqChannel[0],:]+\
                       data[Locomotion_NIdaqChannel[1],:]}

##############################################
###           Face data                   ###
##############################################

class FaceData(ImageTimeSeries):

    def __init__(self, datafolder, metadata,
                 dt=None, times=None,
                 t0=0, sampling_rate=None,
                 compressed_version=True):

        times = np.load(os.path.join(datafolder,
                                     'FaceCamera-times.npy'))

        self.build_temporal_sampling(times,
                                     sampling_rate=sampling_rate)

        if compressed_version or (not os.path.isdir(os.path.join(datafolder, 'FaceCamera-imgs'))):
            # means we have a compressed version
            super().__init__(os.path.join(datafolder, 'FaceCamera-compressed'),
                             times=self.t,
                             frame_sampling=self.iframes,
                             extension='.avi')
        else:
            super().__init__(os.path.join(datafolder, 'FaceCamera-imgs'),
                             times=self.t,
                             frame_sampling=self.iframes,
                             extension='.npy')


    def build_temporal_sampling(self, times,
                                sampling_rate=None):

        if sampling_rate is None:
            self.sampling_rate = 1./np.diff(times).mean()
            self.t = times
            self.iframes = np.arange(len(times))
        else:
            self.sampling_rate = sampling_rate
            t0, t, self.iframes, self.t = times[0], times[0], [], []
            while t<=times[-1]:
                it = np.argmin((times-t)**2)
                self.iframes.append(it)
                self.t.append(t-t0)
                t+=1./self.sampling_rate
            self.t = np.array(self.t)
            

##############################################
###           Pupil data                   ###
##############################################

class PupilData(FaceData):
    """
    Same than FaceData but 

    -> need to introduce a zoom and a saturation
    -> need to override the grab_frame method
    """
    
    def __init__(self, datafolder, metadata,
                 dt=None, times=None,
                 t0=0, sampling_rate=None,
                 compressed_version=True):

        super().__init__(datafolder, metadata,
                         dt=dt, times=times,
                         t0=t0, sampling_rate=sampling_rate,
                         compressed_version=compressed_version)

        # Adding ROI data to the object
        folder = os.path.join(datafolder, 'pupil-ROIs.npy')
        if os.path.isdir(folder):
            rois = np.load(folder,allow_pickle=True).item()
            setattr(self, 'saturation', rois['ROIsaturation'])
            setattr(self, 'reflectors', rois['reflectors'])
            setattr(self, 'ellipse', rois['ROIellipse'])
        else:
            for key in ['saturation', 'reflectors', 'ellipse']:
                setattr(self, key, None)
        
        # Adding processed pupil data to the object
        folder = os.path.join(datafolder,'pupil-data.npy')
        if os.path.isfile(folder):
            data = np.load(folder, allow_pickle=True).item()
            setattr(self, 'processed', data)
            # adding the diameter
            self.processed['diameter'] = np.sqrt(data['sx-corrected']*data['sy-corrected'])
        else:
            setattr(self, 'processed', None)
            

            
    
##############################################
###           Electrophy data              ###
##############################################

class ElectrophyData(SingleValueTimeSerie):
    """ NOT USED YET "SingleValueTimeSerie" is enough ! """
    def __init__(self, *args):
        super().__init__(*args)
        
        
##############################################
###         Multimodal dataset             ###
##############################################

MODALITIES = ['Screen', 'Locomotion', 'Electrophy', 'Face', 'Pupil','Calcium']

class Dataset:
    
    def __init__(self, datafolder,
                 Photodiode_NIdaqChannel=0, # switch to 
                 Electrophy_NIdaqChannel=0, # switch to 
                 Locomotion_NIdaqChannels=[1,2],
                 compressed_version=True,
                 FaceCamera_frame_rate=None,
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

        # Screen and visual stim
        if self.metadata['VisualStim'] and ('Screen' in modalities):
            self.Screen = ScreenData(self.datafolder, self.metadata,
                                     NIdaq_trace=data[Photodiode_NIdaqChannel,:])
        elif 'Screen' in modalities:
            print('[X] Screen data not found !')

        # Locomotion
        if self.metadata['NIdaq'] and (data.shape[0]>max(Locomotion_NIdaqChannels)) and ('Locomotion' in modalities):
            self.Locomotion = SingleValueTimeSerie(data[Locomotion_NIdaqChannels[0],:]+data[Locomotion_NIdaqChannels[1],:],
                                                   dt = 1./self.metadata['NIdaq-acquisition-frequency'])
        elif 'Locomotion' in modalities:
            print('[X] Locomotion data not found !')

        # Electrophy
        if self.metadata['NIdaq'] and (data.shape[0]>Electrophy_NIdaqChannel) and ('Electrophy' in modalities):
            self.Electrophy = SingleValueTimeSerie(data[Electrophy_NIdaqChannel,:],
                                                   dt = 1./self.metadata['NIdaq-acquisition-frequency'])
        elif 'Electrophy' in modalities:
            print('[X] Electrophy data not found !')

        # Face images
        if self.metadata['FaceCamera'] and ('Face' in modalities):
            self.Face = FaceData(self.datafolder, self.metadata,
                                 sampling_rate=FaceCamera_frame_rate,
                                 compressed_version=compressed_version)
        elif 'Face' in modalities:
            print('[X] Face data not found !')

        # Pupil
        if self.metadata['FaceCamera'] and ('Pupil' in modalities):
            self.Pupil = PupilData(self.datafolder, self.metadata,
                                   sampling_rate=FaceCamera_frame_rate,
                                   compressed_version=compressed_version)
        elif 'Pupil' in modalities:
            print('[X] Pupil data not found !')

        # Realignement if possible
        if ('Screen' in modalities) and self.metadata['NIdaq'] and (self.Screen is not None) and (self.Screen.photodiode is not None):
            self.realign_from_photodiode()
            self.Screen.set_times_from_metadata(self.metadata,
                                                realigned=True)
            
            
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

    fn = '/home/yann/DATA/2020_09_11/13-40-10/'
    fn = '/home/yann/DATA/2020_09_23/16-40-54/'
    fn = '/home/yann/DATA/2020_10_07/16-02-19/'
    if sys.argv[-1]=='photodiode':

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

        dataset = Dataset(fn,
                          # FaceCamera_frame_rate=1.,
                          modalities=['Face', 'Pupil'])

        # print(dataset.Pupil.t)
        print(len(dataset.Pupil.t), len(dataset.Pupil.iframes), len(dataset.Pupil.index_frame_map))
        # frame = dataset.Pupil.grab_frame(30, verbose=True)
        
        # from datavyz import ges
        # ges.image(frame)
        # ges.show()
        
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
