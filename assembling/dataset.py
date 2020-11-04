import numpy as np
import os, sys, pathlib

import skvideo.io
from PIL import Image

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import day_folder, create_day_folder, generate_filename_path,\
    check_datafolder, get_files_with_extension

from behavioral_monitoring.locomotion import compute_position_from_binary_signals

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
                 times=None,
                 frame_sampling=None,
                 extension='.npy',
                 lazy_loading=True,
                 compression_metadata=None,
                 t0=0):
        """
        IMAGES can be initialized
        """

        self.extension=extension
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
        elif extension=='.npz':
            self.VIDS = FILES
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

        if lazy_loading and self.VIDS is not None:
            # we just make a map between indices and videos/frames
            self.index_frame_map = []
            for i, fn in enumerate(self.VIDS):
                s = fn.split('imgs-')[1].replace(extension, '').split('-')
                i0, i1 = int(s[0]), int(s[1])
                for i, iframe in enumerate(np.arange(i0, i1+1)):
                    if iframe in frame_sampling:
                        self.index_frame_map.append([fn, i])
        elif (self.VIDS is not None):
            print('Pre-loading the full-set of videos [...]')
            self.IMAGES = []
            if extension=='.npz':
                # we pre-load the video !
                for i, fn in enumerate(self.VIDS):
                    s = fn.split('imgs-')[1].replace(extension, '').split('-')
                    i0, i1 = int(s[0]), int(s[1])
                    x = np.load(fn)
                    for i, iframe in enumerate(np.arange(i0, i1+1)):
                        if iframe in frame_sampling:
                            self.IMAGES.append(x[i,:,:,0])
            else:
                for i, fn in enumerate(self.VIDS):
                    s = fn.split('imgs-')[1].replace(extension, '').split('-')
                    i0, i1 = int(s[0]), int(s[1])
                    x = skvideo.io.vread(fn)
                    for i, iframe in enumerate(np.arange(i0, i1+1)):
                        if iframe in frame_sampling:
                            self.IMAGES.append(x[i,:,:,0])

    def grab_frame(self, t,
                   force_previous_time=False,
                   verbose=False,
                   with_time=False,
                   with_index=False):

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
        elif self.extension=='.npz':
            fn, index = self.index_frame_map[i0]
            x = np.load(fn)['arr_0']
            im = x[index,:,:]
        else:
            # we have loaded it using the "lazy_loading" option
            fn, index = self.index_frame_map[i0]
            x = skvideo.io.vread(fn)
            im = x[index,:,:,0]
            
        if with_time and with_index:
            return i0, self.t[i0], im
        elif with_time:
            return self.t[i0], im
        elif with_index:
            return i0, im
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
            setattr(self, 'time_start', np.array(metadata['time_start_realigned']))
            setattr(self, 'time_stop', np.array(metadata['time_stop_realigned']))
        else:
            setattr(self, 'time_start', np.array(metadata['time_start']))
            setattr(self, 'time_stop', np.array(metadata['time_stop']))

        times = [0] # starting with the pre-frame
        for ts in self.time_start:
            times = times + [ts, ts+metadata['presentation-duration']]
        times.append(1e10) # adding a last point very far in the future
        self.t = np.array(times)
        
            
##############################################
###           Locomotion data              ###
##############################################

class LocomotionData:
    """
    We split the binary signal here, then
    see ../behavioral_monitoring/locomotion.py for the algorithm to decode the rotary-encoder signal

    should follow the same attributes than the  "SingleValueTimeSerie"
    """
    
    def __init__(self, binary_signal, dt=1.,
                 times = None,
                 t0=0):

        A = binary_signal[0]%2
        B = np.round(binary_signal[0]/2, 0)

        if times is not None:
            self.t = times
        else:
            self.t = np.arange(binary_signal.shape[1])*dt+t0
            
        self.val = compute_position_from_binary_signals(A, B)
        

##############################################
###           Face data                   ###
##############################################

class FaceData(ImageTimeSeries):

    def __init__(self, datafolder, metadata,
                 dt=None, times=None,
                 t0=None, sampling_rate=None,
                 lazy_loading=True,
                 compressed_version=False):

        times = np.load(os.path.join(datafolder,
                                     'FaceCamera-times.npy'))
        if t0 is None:
            t0 = times[0] # just in case, but should be relative to NIdaq.start
            
        times = times-t0 
        
        self.build_temporal_sampling(times,
                                     sampling_rate=sampling_rate)

        if compressed_version and (not os.path.isdir(os.path.join(datafolder, 'FaceCamera-imgs'))):
            
            # means we have a compressed version
            compression_metadata = np.load(os.path.join(datafolder, 'FaceCamera-compressed', 'metadata.npy'),
                                           allow_pickle=True).item()
            
            super().__init__(os.path.join(datafolder, 'FaceCamera-compressed'),
                             times=self.t,
                             frame_sampling=self.iframes,
                             lazy_loading=lazy_loading,
                             extension=compression_metadata['extension'])
        else:
            print('trying to load')
            super().__init__(os.path.join(datafolder, 'FaceCamera-imgs'),
                             times=self.t,
                             frame_sampling=self.iframes,
                             extension='.npy')


    def build_temporal_sampling(self, times,
                                sampling_rate=None):

        if sampling_rate is None:
            self.sampling_rate = 1./np.diff(times).mean()
            print(self.sampling_rate)
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
                 lazy_loading=True,
                 t0=None, sampling_rate=None,
                 compressed_version=True):

        super().__init__(datafolder, metadata,
                         dt=dt, times=times,
                         lazy_loading=lazy_loading,
                         t0=t0, sampling_rate=sampling_rate,
                         compressed_version=compressed_version)

        # Adding ROI data to the object
        fn = os.path.join(datafolder, 'pupil-ROIs.npy')
        if os.path.isfile(fn):
            rois = np.load(fn,allow_pickle=True).item()
            for key in rois:
                setattr(self, key, rois[key])
        else:
            for key in ['saturation', 'reflectors', 'ellipse']:
                setattr(self, key, None)
        # to be filled once the images are loaded
        for key in ['xmin', 'xmax', 'ymin', 'ymax']:
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
                 Photodiode_NIdaqChannel=0,
                 Electrophy_NIdaqChannel=1,
                 # Locomotion_NIdaqDigitalChannels=[0,1], # not NEEDED, we just read the digital channels
                 compressed_version=False,
                 lazy_loading=True,
                 FaceCamera_frame_rate=None,
                 modalities=MODALITIES):
        """

        by default we take all modalities, you can restrict them using "modalities"
        """
        for key in MODALITIES:
            setattr(self, key, None) # all modalities to None by default
            
        self.datafolder = datafolder
        self.metadata = check_datafolder(self.datafolder, modalities)
        
        try:
            data = np.load(os.path.join(self.datafolder, 'NIdaq.npy'), allow_pickle=True).item()
            self.NIdaq_Tstart = np.load(os.path.join(self.datafolder, 'NIdaq.start.npy'))[0]
        except FileNotFoundError:
            print('No NIdaq file available')
            data=None
            self.NIdaq_Tstart = None

        # Screen and visual stim
        if self.metadata['VisualStim'] and ('Screen' in modalities):
            self.Screen = ScreenData(self.datafolder, self.metadata,
                                     NIdaq_trace=data['analog'][Photodiode_NIdaqChannel,:])
        elif 'Screen' in modalities:
            print('[X] Screen data not found !')

        # Locomotion
        if self.metadata['Locomotion'] and ('Locomotion' in modalities):
            self.Locomotion = LocomotionData(data['digital'],
                                             dt = 1./self.metadata['NIdaq-acquisition-frequency'])
        elif 'Locomotion' in modalities:
            print('[X] Locomotion data not found !')

        # Electrophy
        if self.metadata['Electrophy'] and ('Electrophy' in modalities):
            self.Electrophy = SingleValueTimeSerie(data['analog'][Electrophy_NIdaqChannel,:],
                                                   dt = 1./self.metadata['NIdaq-acquisition-frequency'])
        elif 'Electrophy' in modalities:
            print('[X] Electrophy data not found !')

        # Face images
        if self.metadata['FaceCamera'] and ('Face' in modalities):
            self.Face = FaceData(self.datafolder, self.metadata,
                                 sampling_rate=FaceCamera_frame_rate,
                                 lazy_loading=lazy_loading,
                                 t0 = self.NIdaq_Tstart,
                                 compressed_version=compressed_version)
        elif 'Face' in modalities:
            print('[X] Face data not found !')

        # Pupil
        if self.metadata['FaceCamera'] and ('Pupil' in modalities):
            self.Pupil = PupilData(self.datafolder, self.metadata,
                                   lazy_loading=lazy_loading,
                                   sampling_rate=FaceCamera_frame_rate,
                                   t0 = self.NIdaq_Tstart,
                                   compressed_version=compressed_version)
        elif 'Pupil' in modalities:
            print('[X] Pupil data not found !')

        if (self.Screen is not None) and ('Screen' in modalities):
            # Realignement if possible
            success = False
            if (self.Screen.photodiode is not None):
                success = self.realign_from_photodiode()
                self.Screen.set_times_from_metadata(self.metadata,
                                                    realigned=True)
            if not success:
                self.Screen.set_times_from_metadata(self.metadata,
                                                    realigned=False)
            
            
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
        return success


def find_onset_time(t, photodiode_signal, npulses,
                    time_for_threshold=10e-3):
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
                          compressed_version=False,
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
