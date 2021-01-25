from psychopy import visual, core, event, clock, monitors # some libraries from PsychoPy
import numpy as np
import itertools, os, sys, pathlib, subprocess, time, datetime, json
import pynwb
from hdmf.data_utils import DataChunkIterator
from hdmf.backends.hdf5.h5_utils import H5DataIO
from dateutil.tz import tzlocal
import pytz

sys.path.append(str(pathlib.Path(__file__).resolve().parents[0]))
import screens, noise, natural_images

def stop_signal(parent):
    if (len(event.getKeys())>0) or (parent.stop_flag):
        parent.stop_flag = True
        parent.statusBar.showMessage('stimulation stopped !')
        return True
    else:
        return False


class visual_stim:

    def __init__(self,
                 protocol=None,
                 protocol_file='',
                 screen = 'Lilliput',
                 stimuli_folder=os.path.join(os.path.expanduser('~'), 'DATA', 'STIMULI'),
                 task='visualize',
                 demo=False):
        """
        """
        if protocol is None:
            with open(protocol_file, 'r') as fp:
                self.protocol = json.load(fp)
        else:
            self.protocol = protocol
        self.task = task
        if 'filename' not in self.protocol:
            protocol['filename'] = protocol['Stimulus']

        self.load_screen(screen)
        
        self.stimuli_folder = stimuli_folder
        self.win, self.io, self.nwbfile = None, None, None
        self.demo = demo


    def load_screen(self, screen):
        self.screen = screens.SCREENS[screen]
        self.screen['shape'] = (self.screen['resolution'][1], self.screen['resolution'][0])
        
        
    def init_presentation(self):
        if self.demo or (self.protocol['Setup']=='demo-mode'):
            self.monitor = monitors.Monitor('testMonitor')
            self.win = visual.Window((int(self.screen['resolution'][0]/2),
                                      int(self.screen['resolution'][1]/2)),
                                     monitor=self.monitor, screen=0,
                                     units='pix', color=-1) #create a window
        else:
            self.monitor = monitors.Monitor(self.screen['name'])
            self.win = visual.Window((self.screen['resolution'][0],
                                      self.screen['resolution'][1]),
                                     monitor=self.monitor,
                                     screen=self.screen['screen_id'],
                                     fullscr=True, units='deg', color=-1)
            
    # Gamma correction 
    def gamma_corrected_lum(self, level):
        """ input between -1 and 1 """
        return 2*np.power(((level+1.)/2./self.screen['gamma_correction']['k']), 1./self.screen['gamma_correction']['gamma'])-1.
    
    def gamma_corrected_contrast(self, contrast):
        """ input between 0 and 1 """
        return np.power(contrast/self.screen['gamma_correction']['k'], 1./self.screen['gamma_correction']['gamma'])
    
    # initialize all quantities
    def init_experiment(self, protocol, keys):

        self.experiment, self.PATTERNS = {}, []

        if protocol['Presentation']=='Single-Stimulus':
            for key in protocol:
                if key.split(' (')[0] in keys:
                    self.experiment[key.split(' (')[0]] = [protocol[key]]
                    self.experiment['index'] = [0]
                    self.experiment['repeat'] = [0]
                    self.experiment['time_start'] = [protocol['presentation-prestim-period']]
                    self.experiment['time_stop'] = [protocol['presentation-duration']+protocol['presentation-prestim-period']]
        else: # MULTIPLE STIMS
            VECS, FULL_VECS = [], {}
            for key in keys:
                FULL_VECS[key], self.experiment[key] = [], []
                if protocol['N-'+key]>1:
                    VECS.append(np.linspace(protocol[key+'-1'], protocol[key+'-2'],protocol['N-'+key]))
                else:
                    VECS.append(np.array([protocol[key+'-2']]))
            for vec in itertools.product(*VECS):
                for i, key in enumerate(keys):
                    FULL_VECS[key].append(vec[i])
                    
            self.experiment['index'], self.experiment['repeat'] = [], []
            self.experiment['time_start'], self.experiment['time_stop'] = [], []

            index_no_repeat = np.arange(len(FULL_VECS[key]))

            Nrepeats = max([1,protocol['N-repeat']])
            index, repeat = [], []
            for r in range(Nrepeats):
                # SHUFFLING IF NECESSARY
                if (protocol['Presentation']=='Randomized-Sequence'):
                    np.random.seed(protocol['shuffling-seed'])
                    np.random.shuffle(index_no_repeat)
                index += list(index_no_repeat)
                repeat += list(r*np.ones(len(index_no_repeat)))
            index, repeat = np.array(index), np.array(repeat)

            for n, i in enumerate(index[protocol['starting-index']:]):
                for key in keys:
                    self.experiment[key].append(FULL_VECS[key][i])
                self.experiment['index'].append(i)
                self.experiment['repeat'].append(repeat[n+protocol['starting-index']])
                self.experiment['time_start'].append(protocol['presentation-prestim-period']+\
                                                     n*protocol['presentation-duration']+n*protocol['presentation-interstim-period'])
                self.experiment['time_stop'].append(protocol['presentation-prestim-period']+\
                                                     (n+1)*protocol['presentation-duration']+n*protocol['presentation-interstim-period'])

    def write_protocol(self, filename):
        # to overwrite the protocol with the filename info
        with open(filename, 'w') as f:
            json.dump(self.protocol, f, indent=4)
        
    # the close function
    def close(self):
        if self.io is not None:
            self.io.close()
        if self.win is not None:
            self.win.close()

    def quit(self):
        if self.io is not None:
            self.io.close()
        core.quit()

    def check_movie(self):
        if ('movie_filename' in self.protocol) and\
           os.path.isfile(os.path.join(self.stimuli_folder, self.protocol['movie_filename'])):
            return True
        else:
            return False

    def preload_movie(self):

        if self.check_movie():
            self.io = pynwb.NWBHDF5IO(os.path.join(self.stimuli_folder, self.protocol['movie_filename']), 'r')
            self.nwbfile = self.io.read()
        else:
            print(self.protocol['movie_filename'], 'not found in ', self.stimuli_folder)

            
    def run(self, parent):
        if self.nwbfile is not None:
            start = clock.getTime()
            index, imax = 0, self.nwbfile.stimulus['visual-stimuli'].data.shape[0]
            while index<imax and not parent.stop_flag:
                frame = visual.ImageStim(self.win,
                                         image=self.nwbfile.stimulus['visual-stimuli'].data[index,:,:],
                                         units='pix', size=self.win.size)
                frame.draw()
                self.win.flip()
                index = int((clock.getTime()-start)*self.nwbfile.stimulus['visual-stimuli'].rate)
        else:
            print(' Need to generate and preload the nwbfile movie')
            print(' /!\ running not possible ! /!\ ')
                
            
    def add_monitoring_signal(self, x, y, img, tnow, episode_start):
        cond = (x<self.screen['monitoring_square']['size']) & (y<self.screen['monitoring_square']['size'])
        if len(np.argwhere(((tnow-episode_start)>=self.screen['monitoring_square']['on_times']) &\
                           ((tnow-episode_start)<=self.screen['monitoring_square']['on_times']+\
                            self.screen['monitoring_square']['on_duration'])))>0:
            img[cond] = 1
        else:
            img[cond] = -1

    def generate_movie(self):

        filename = self.protocol['filename'].split(os.path.sep)[-1].replace('.json',
                                    '_'+datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')+'.nwb')

        self.protocol['movie_filename'] = filename
        self.nwbfile = pynwb.NWBFile(identifier=filename,
                                     session_description='Movie file for stimulus presentation',
                                     session_start_time=datetime.datetime.now(pytz.utc),
                                     experiment_description=str(self.protocol),
                                     experimenter='Yann Zerlaut',
                                     lab='Bacci and Rebola labs',
                                     institution='Paris Brain Institute',
                                     source_script=str(pathlib.Path(__file__).resolve()),
                                     source_script_file_name=str(pathlib.Path(__file__).resolve()),
                                     file_create_date=datetime.datetime.now(pytz.utc))

        data = DataChunkIterator(data=self.frame_generator())
        # WITH COMPRESSION
        # dataC = H5DataIO(data=data,
        #                  compression='gzip',
        #                  compression_opts=4)
        frame_stimuli = pynwb.image.ImageSeries(name='visual-stimuli',
                                                # data=dataC, # putting compressed data
                                                data=data,
                                                unit='NA',
                                                starting_time=0.,
                                                rate=self.protocol['movie_refresh_freq'])
        self.nwbfile.add_stimulus(frame_stimuli)

        io = pynwb.NWBHDF5IO(os.path.join(self.stimuli_folder, self.protocol['movie_filename']), 'w')

        io.write(self.nwbfile)
        io.close()

    def cm_to_angle(self, value):
        return 180./np.pi*np.arctan(value/self.screen['distance_from_eye'])
    
    def pix_to_angle(self, value):
        return self.cm_to_angle(value/self.screen['resolution'][0]*\
                                self.screen['width'])

    
    def angle_meshgrid(self):
        x = np.linspace(self.cm_to_angle(-self.screen['width']/2.),
                        self.cm_to_angle(self.screen['width']/2.),
                        self.screen['resolution'][0])
        z = np.linspace(self.cm_to_angle(-self.screen['height']/2.),
                        self.cm_to_angle(self.screen['height']/2.),
                        self.screen['resolution'][1])
        return np.meshgrid(x, z)

    def pixel_meshgrid(self):
        return np.meshgrid(np.arange(self.screen['resolution'][0]),
                           np.arange(self.screen['resolution'][1]))

    
    # def angle_to_cm(self, value):
    #     return self.screen['distance_from_eye']*np.tan(np.pi/180.*value)
    
    # def angle_to_pix(self, value):
    #     return self.screen['resolution'][0]/self.screen['width']*\
    #         self.angle_to_cm(value)

                
    # def horizontal_angle_to_pixel(self, value):
    #     """ 0-angle is the center of the screen"""
    #     x0 = int(self.screen['resolution'][0]/2.)
    #     return self.angle_to_pix(value)+x0
    
    # def vertical_angle_to_pixel(self, value):
    #     """ 0-angle is the center of the screen"""
    #     z0 = int(self.screen['resolution'][1]/2.)
    #     return self.angle_to_pix(value)+z0
    
    def frame_generator(self, nmax=100):
        print('  /!\ Should be implemented in the child class !! /!\ ')
        
        
#####################################################
##  ----   PRESENTING VARIOUS LIGHT LEVELS  --- #####           
#####################################################

class light_level_single_stim(visual_stim):

    def __init__(self, **args):
        
        super().__init__(**args)
        if self.task=='generate':
            super().init_experiment(self.protocol, ['light-level'])
        
    def frame_generator(self):
        """
        Generator creating a random number of chunks (but at most max_chunks) of length chunk_length containing
        random samples of sin([0, 2pi]).
        """
        x, y = self.pixel_meshgrid()
        # prestim
        for i in range(int(self.protocol['presentation-prestim-period']*self.protocol['movie_refresh_freq'])):
            yield np.ones(self.screen['shape'])*(2*self.protocol['presentation-prestim-screen']-1)
        for episode, start in enumerate(self.experiment['time_start']):
            img = self.experiment['light-level'][episode]+0.*x
            img = self.gamma_corrected_lum(img)
            for i in range(int(self.protocol['presentation-duration']*self.protocol['movie_refresh_freq'])):
                self.add_monitoring_signal(x, y, img, i/self.protocol['movie_refresh_freq'], 0)
                yield img
            if episode<len(self.experiment['time_start'])-1:
                # adding interstim
                for i in range(int(self.protocol['presentation-interstim-period']*self.protocol['movie_refresh_freq'])):
                    yield np.ones(self.screen['shape'])*(2*self.protocol['presentation-interstim-screen']-1)
        # poststim
        for i in range(int(self.protocol['presentation-poststim-period']*self.protocol['movie_refresh_freq'])):
            yield np.ones(self.screen['shape'])*(2*self.protocol['presentation-poststim-screen']-1)
        return        

            
#####################################################
##  ----   PRESENTING FULL FIELD GRATINGS   --- #####           
#####################################################

class full_field_grating_stim(visual_stim):

    def __init__(self, **args):
        
        super().__init__(**args)
        if self.task=='generate':
            super().init_experiment(self.protocol, ['spatial-freq', 'angle', 'contrast'])
        
    def frame_generator(self):
        """
        Generator creating a random number of chunks (but at most max_chunks) of length chunk_length containing
        random samples of sin([0, 2pi]).
        """
        xp, zp = self.pixel_meshgrid()
        x, z = self.angle_meshgrid()
        # prestim
        for i in range(int(self.protocol['presentation-prestim-period']*self.protocol['movie_refresh_freq'])):
            yield np.ones(self.screen['shape'])*(2*self.protocol['presentation-prestim-screen']-1)
        for episode, start in enumerate(self.experiment['time_start']):
            angle = self.experiment['angle'][episode]
            spatial_freq = self.experiment['spatial-freq'][episode]
            contrast = self.experiment['contrast'][episode]
            x_rot = x*np.cos(angle/180.*np.pi)+z*np.sin(angle/180.*np.pi)
            img = np.cos(2*np.pi*spatial_freq*x_rot)
            img = self.gamma_corrected_lum(img)
            for i in range(int(self.protocol['presentation-duration']*self.protocol['movie_refresh_freq'])):
                self.add_monitoring_signal(xp, zp, img, i/self.protocol['movie_refresh_freq'], 0)
                yield img
            if episode<len(self.experiment['time_start'])-1:
                # adding interstim
                for i in range(int(self.protocol['presentation-interstim-period']*self.protocol['movie_refresh_freq'])):
                    yield np.ones(self.screen['shape'])*(2*self.protocol['presentation-interstim-screen']-1)
        # poststim
        for i in range(int(self.protocol['presentation-poststim-period']*self.protocol['movie_refresh_freq'])):
            yield np.ones(self.screen['shape'])*(2*self.protocol['presentation-poststim-screen']-1)
        return        

            
class drifting_full_field_grating_stim(visual_stim):

    def __init__(self, **args):
        
        super().__init__(**args)
        if self.task=='generate':
            super().init_experiment(self.protocol, ['spatial-freq', 'angle', 'contrast', 'speed'])
        
    def frame_generator(self):
        """
        Generator creating a random number of chunks (but at most max_chunks) of length chunk_length containing
        random samples of sin([0, 2pi]).
        """
        xp, zp = self.pixel_meshgrid()
        x, z = self.angle_meshgrid()
        # prestim
        for i in range(int(self.protocol['presentation-prestim-period']*self.protocol['movie_refresh_freq'])):
            yield np.ones(self.screen['shape'])*(2*self.protocol['presentation-prestim-screen']-1)
        for episode, start in enumerate(self.experiment['time_start']):
            angle = self.experiment['angle'][episode]
            spatial_freq = self.experiment['spatial-freq'][episode]
            contrast = self.experiment['contrast'][episode]
            speed = self.experiment['speed'][episode]
            x_rot = x*np.cos(angle/180.*np.pi)+z*np.sin(angle/180.*np.pi)
            for i in range(int(self.protocol['presentation-duration']*self.protocol['movie_refresh_freq'])):
                img = np.cos(2*np.pi*spatial_freq*x_rot+2*np.pi*speed*i/self.protocol['movie_refresh_freq'])
                img = self.gamma_corrected_lum(img)
                self.add_monitoring_signal(xp, zp, img, i/self.protocol['movie_refresh_freq'], 0)
                yield img
            if episode<len(self.experiment['time_start'])-1:
                # adding interstim
                for i in range(int(self.protocol['presentation-interstim-period']*self.protocol['movie_refresh_freq'])):
                    yield np.ones(self.screen['shape'])*(2*self.protocol['presentation-interstim-screen']-1)
        # poststim
        for i in range(int(self.protocol['presentation-poststim-period']*self.protocol['movie_refresh_freq'])):
            yield np.ones(self.screen['shape'])*(2*self.protocol['presentation-poststim-screen']-1)
        return        
            
#####################################################
##  ----    PRESENTING CENTERED GRATINGS    --- #####           
#####################################################

class center_grating_stim(visual_stim):
    
    def __init__(self, **args):
        
        super().__init__(**args)
        if self.task=='generate':
            super().init_experiment(self.protocol, ['x-center', 'y-center', 'radius','spatial-freq', 'angle', 'contrast', 'bg-color'])
        
    def frame_generator(self):
        """
        Generator creating a random number of chunks (but at most max_chunks) of length chunk_length containing
        random samples of sin([0, 2pi]).
        """
        xp, zp = self.pixel_meshgrid()
        x, z = self.angle_meshgrid()
        # prestim
        for i in range(int(self.protocol['presentation-prestim-period']*self.protocol['movie_refresh_freq'])):
            yield np.ones(self.screen['shape'])*(2*self.protocol['presentation-prestim-screen']-1)
        for episode, start in enumerate(self.experiment['time_start']):
            angle = self.experiment['angle'][episode]
            spatial_freq = self.experiment['spatial-freq'][episode]
            contrast = self.experiment['contrast'][episode]
            xcenter, zcenter = self.experiment['x-center'][episode], self.experiment['y-center'][episode]
            radius = self.experiment['radius'][episode]
            bg_color = self.experiment['bg-color'][episode]
            x_rot = x*np.cos(angle/180.*np.pi)+z*np.sin(angle/180.*np.pi)
            circle_cond = (((x.flatten()-xcenter)**2+(z.flatten()-zcenter)**2)<=radius**2)
            img0 = (2*bg_color-1)*np.ones(self.screen['shape']).flatten()
            img0[circle_cond] = np.cos(2*np.pi*spatial_freq*x_rot.flatten()[circle_cond])
            img = self.gamma_corrected_lum(img0).reshape(self.screen['shape'])
            for i in range(int(self.protocol['presentation-duration']*self.protocol['movie_refresh_freq'])):
                self.add_monitoring_signal(xp, zp, img, i/self.protocol['movie_refresh_freq'], 0)
                yield img
            if episode<len(self.experiment['time_start'])-1:
                # adding interstim
                for i in range(int(self.protocol['presentation-interstim-period']*self.protocol['movie_refresh_freq'])):
                    yield np.ones(self.screen['shape'])*(2*self.protocol['presentation-interstim-screen']-1)
        # poststim
        for i in range(int(self.protocol['presentation-poststim-period']*self.protocol['movie_refresh_freq'])):
            yield np.ones(self.screen['shape'])*(2*self.protocol['presentation-poststim-screen']-1)
        return        

            
class drifting_center_grating_stim(visual_stim):
    
    def __init__(self, protocol):
        super().__init__(protocol)
        super().init_experiment(protocol, ['x-center', 'y-center', 'radius','spatial-freq', 'angle', 'contrast', 'speed', 'bg-color'])

        print(self.experiment['bg-color'])
        # then manually building patterns
        for i in range(len(self.experiment['index'])):
            self.PATTERNS.append([\
                                  visual.GratingStim(win=self.win,
                                                     pos=[self.experiment['x-center'][i], self.experiment['y-center'][i]],
                                                     size=self.experiment['radius'][i], mask='circle',
                                                     sf=self.experiment['spatial-freq'][i],
                                                     ori=self.experiment['angle'][i],
                                                     contrast=self.gamma_corrected_contrast(self.experiment['contrast'][i]))])


#####################################################
##  ----    PRESENTING OFF-CENTERED GRATINGS    --- #####           
#####################################################

class off_center_grating_stim(visual_stim):
    
    def __init__(self, protocol):
        super().__init__(protocol)
        super().init_experiment(protocol, ['x-center', 'y-center', 'radius','spatial-freq', 'angle', 'contrast', 'bg-color'])

        # then manually building patterns
        for i in range(len(self.experiment['index'])):
            self.PATTERNS.append([\
                                  visual.GratingStim(win=self.win,
                                                     size=1000, pos=[0,0],
                                                     sf=self.experiment['spatial-freq'][i],
                                                     ori=self.experiment['angle'][i],
                                                     contrast=self.gamma_corrected_contrast(self.experiment['contrast'][i])),
                                  visual.GratingStim(win=self.win,
                                                     pos=[self.experiment['x-center'][i], self.experiment['y-center'][i]],
                                                     size=self.experiment['radius'][i],
                                                     mask='circle', sf=0,
                                                     color=self.gamma_corrected_lum(self.experiment['bg-color'][i]))])

            
class drifting_off_center_grating_stim(visual_stim):
    
    def __init__(self, protocol):
        super().__init__(protocol)
        super().init_experiment(protocol, ['x-center', 'y-center', 'radius','spatial-freq', 'angle', 'contrast', 'bg-color', 'speed'])

        # then manually building patterns
        for i in range(len(self.experiment['index'])):
            self.PATTERNS.append([\
                                  # Surround grating
                                  visual.GratingStim(win=self.win,
                                                     size=1000, pos=[0,0],
                                                     sf=self.experiment['spatial-freq'][i],
                                                     ori=self.experiment['angle'][i],
                                                     contrast=self.gamma_corrected_contrast(self.experiment['contrast'][i])),
                                  # + center Mask
                                  visual.GratingStim(win=self.win,
                                                     pos=[self.experiment['x-center'][i], self.experiment['y-center'][i]],
                                                     size=self.experiment['radius'][i],
                                                     mask='circle', sf=0, contrast=0,
                                                     color=self.gamma_corrected_lum(self.experiment['bg-color'][i]))])


#####################################################
##  ----    PRESENTING SURROUND GRATINGS    --- #####           
#####################################################
        
class surround_grating_stim(visual_stim):

    def __init__(self, protocol):
        super().__init__(protocol)
        super().init_experiment(protocol, ['x-center', 'y-center', 'radius-start', 'radius-end','spatial-freq', 'angle', 'contrast', 'bg-color'])

        # then manually building patterns
        for i in range(len(self.experiment['index'])):
            self.PATTERNS.append([\
                                  visual.GratingStim(win=self.win,
                                                     size=1000, pos=[0,0], sf=0,
                                                     color=self.gamma_corrected_lum(self.experiment['bg-color'][i])),
                                  visual.GratingStim(win=self.win,
                                                     pos=[self.experiment['x-center'][i], self.experiment['y-center'][i]],
                                                     size=self.experiment['radius-end'][i],
                                                     mask='circle', 
                                                     sf=self.experiment['spatial-freq'][i],
                                                     ori=self.experiment['angle'][i],
                                                     contrast=self.gamma_corrected_contrast(self.experiment['contrast'][i])),
                                  visual.GratingStim(win=self.win,
                                                     pos=[self.experiment['x-center'][i], self.experiment['y-center'][i]],
                                                     size=self.experiment['radius-start'][i],
                                                     mask='circle', sf=0,
                                                     color=self.gamma_corrected_lum(self.experiment['bg-color'][i]))])

class drifting_surround_grating_stim(visual_stim):

    def __init__(self, protocol):
        super().__init__(protocol)
        super().init_experiment(protocol, ['x-center', 'y-center', 'radius-start', 'radius-end','spatial-freq', 'angle', 'contrast', 'bg-color', 'speed'])

        # then manually building patterns
        for i in range(len(self.experiment['index'])):
            self.PATTERNS.append([\
                                  visual.GratingStim(win=self.win,
                                                     size=1000, pos=[0,0], sf=0,contrast=0,
                                                     color=self.gamma_corrected_lum(self.experiment['bg-color'][i])),
                                  visual.GratingStim(win=self.win,
                                                     pos=[self.experiment['x-center'][i], self.experiment['y-center'][i]],
                                                     size=self.experiment['radius-end'][i],
                                                     mask='circle', 
                                                     sf=self.experiment['spatial-freq'][i],
                                                     ori=self.experiment['angle'][i],
                                                     contrast=self.gamma_corrected_contrast(self.experiment['contrast'][i])),
                                  visual.GratingStim(win=self.win,
                                                     pos=[self.experiment['x-center'][i], self.experiment['y-center'][i]],
                                                     size=self.experiment['radius-start'][i],
                                                     mask='circle', sf=0,contrast=0,
                                                     color=self.gamma_corrected_lum(self.experiment['bg-color'][i]))])
        

#####################################################
##  ----    PRESENTING NATURAL IMAGES       --- #####
#####################################################

NI_directory = os.path.join(str(pathlib.Path(__file__).resolve().parents[1]), 'NI_bank')
        
class natural_image(visual_stim):

    def __init__(self, protocol):

        # from visual_stim.psychopy_code.preprocess_NI import load, img_after_hist_normalization
        # from .preprocess_NI import load, img_after_hist_normalization
        
        super().__init__(protocol)
        super().init_experiment(protocol, ['Image-ID'])

        for i in range(len(self.experiment['index'])):
            filename = os.listdir(NI_directory)[int(self.experiment['Image-ID'][i])]
            img = load(os.path.join(NI_directory, filename))
            img = 2*self.gamma_corrected_contrast(img_after_hist_normalization(img))-1 # normalization + gamma_correction
            # rescaled_img = adapt_to_screen_resolution(img, (SCREEN[0], SCREEN[1]))

            self.PATTERNS.append([visual.ImageStim(self.win, image=img.T,
                                                   units='pix', size=self.win.size)])


#####################################################
##  --    WITH VIRTUAL SCENE EXPLORATION    --- #####
#####################################################


def generate_VSE(duration=5,
                 mean_saccade_duration=2.,# in s
                 std_saccade_duration=1.,# in s
                 # saccade_amplitude=50.,
                 saccade_amplitude=100, # in pixels, TO BE PUT IN DEGREES
                 seed=0):
    """
    to do: clean up the VSE generator
    """
    print('generating Virtual-Scene-Exploration [...]')
    
    np.random.seed(seed)
    
    tsaccades = np.cumsum(np.clip(mean_saccade_duration+np.abs(np.random.randn(int(1.5*duration/mean_saccade_duration))*std_saccade_duration),
                                  mean_saccade_duration/2., 1.5*mean_saccade_duration))

    x = np.array(np.clip(np.random.randn(len(tsaccades))*saccade_amplitude, 0, saccade_amplitude), dtype=int)
    y = np.array(np.clip(np.random.randn(len(tsaccades))*saccade_amplitude, 0, saccade_amplitude), dtype=int)
    
    return {'t':np.array([0]+list(tsaccades)),
            'x':np.array([0]+list(x)),
            'y':np.array([0]+list(y)),
            'max_amplitude':saccade_amplitude}

            

class natural_image_vse(visual_stim):

    def __init__(self, protocol):

        super().__init__(protocol)
        super().init_experiment(protocol, ['Image-ID', 'VSE-seed', 'mean-saccade-duration', 'std-saccade-duration'])

        print(self.experiment)
        self.VSEs = [] # array of Virtual-Scene-Exploration
        for i in range(len(self.experiment['index'])):

            vse = generate_VSE(duration=protocol['presentation-duration'],
                               mean_saccade_duration=self.experiment['mean-saccade-duration'][i],
                               std_saccade_duration=self.experiment['std-saccade-duration'][i],
                               saccade_amplitude=100, # in pixels, TO BE PUT IN DEGREES
                               seed=int(self.experiment['VSE-seed'][i]+self.experiment['Image-ID'][i]))

            self.VSEs.append(vse)
            
            filename = os.listdir(NI_directory)[int(self.experiment['Image-ID'][i])]
            img = load(os.path.join(NI_directory, filename))
            img = 2*self.gamma_corrected_contrast(img_after_hist_normalization(img))-1 # normalization + gamma_correction
            # rescaled_img = adapt_to_screen_resolution(img, (SCREEN[0], SCREEN[1]))
            sx, sy = img.T.shape

            self.PATTERNS.append([])
            
            IMAGES = []
            for i in range(len(vse['t'])):
                ix, iy = vse['x'][i], vse['y'][i]
                new_im = img.T[ix:sx-vse['max_amplitude']+ix,\
                               iy:sy-vse['max_amplitude']+iy]
                self.PATTERNS[-1].append(visual.ImageStim(self.win,
                                                          image=new_im,
                                                          units='pix', size=self.win.size))
            
    

#####################################################
##  ----    PRESENTING BINARY NOISE         --- #####
#####################################################

class sparse_noise(visual_stim):
    
    def __init__(self, protocol):

        super().__init__(protocol)
        super().init_experiment(protocol,
            ['square-size', 'sparseness', 'mean-refresh-time', 'jitter-refresh-time'])
        
        self.STIM = build_sparse_noise(protocol['presentation-duration'],
                                       self.monitor,
                                       square_size=protocol['square-size (deg)'],
                                       noise_mean_refresh_time=protocol['mean-refresh-time (s)'],
                                       noise_rdm_jitter_refresh_time=protocol['jitter-refresh-time (s)'],
                                       seed=protocol['noise-seed (#)'])
        
        self.experiment = {'refresh-times':self.STIM['t']}
            

class dense_noise(visual_stim):

    def __init__(self, protocol):

        super().__init__(protocol)
        super().init_experiment(protocol,
                ['square-size', 'sparseness', 'mean-refresh-time', 'jitter-refresh-time'])

        self.STIM = build_dense_noise(protocol['presentation-duration'],
                                      self.monitor,
                                      square_size=protocol['square-size (deg)'],
                                      noise_mean_refresh_time=protocol['mean-refresh-time (s)'],
                                      noise_rdm_jitter_refresh_time=protocol['jitter-refresh-time (s)'],
                                      seed=protocol['noise-seed (#)'])

        self.experiment = {'refresh-times':self.STIM['t']}
            
    
class dummy_parent:
    def __init__(self):
        self.stop_flag= False

if __name__=='__main__':

    import argparse
    parser=argparse.ArgumentParser(description="Generate visual stimuli",
                       formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("protocol_file",
     default=os.path.join(str(pathlib.Path(__file__).resolve().parents[1]),\
                          'exp', 'protocols', 'light-levels.json'))
    parser.add_argument('-t', "--task",\
        help="""
        Task to be performed, either:
        - generate
        - visualize/see/demo
        - play
        - both
        """, default='see')
    parser.add_argument('-sf', "--stimuli_folder",
                        default=os.path.join(os.path.expanduser('~'),
                                             'DATA', 'STIMULI'))
    parser.add_argument('-sc', "--screen", default='Dell-P2018H')
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    task = args.task # need to keep track of original value
    
    if os.path.isfile(args.protocol_file):
        with open(args.protocol_file, 'r') as fp:
            args.protocol = json.load(fp)

        if task in ['generate', 'both']:
            args.task = 'generate'
            if args.protocol['Stimulus']=='light-level':
                stim = light_level_single_stim(**vars(args))
            elif (args.protocol['Stimulus']=='full-field-grating'):
                stim = full_field_grating_stim(**vars(args))
            elif (args.protocol['Stimulus']=='center-grating'):
                stim = center_grating_stim(**vars(args))
            elif (args.protocol['Stimulus']=='off-center-grating'):
                stim = off_center_grating_stim(**vars(args))
            elif (args.protocol['Stimulus']=='surround-grating'):
                stim = surround_grating_stim(**vars(args))
            elif (args.protocol['Stimulus']=='drifting-full-field-grating'):
                stim = drifting_full_field_grating_stim(**vars(args))
            elif (args.protocol['Stimulus']=='drifting-center-grating'):
                stim = drifting_center_grating_stim(**vars(args))
            elif (args.protocol['Stimulus']=='drifting-off-center-grating'):
                stim = drifting_off_center_grating_stim(**vars(args))
            elif (args.protocol['Stimulus']=='drifting-surround-grating'):
                stim = drifting_surround_grating_stim(**vars(args))
            elif (args.protocol['Stimulus']=='Natural-Image'):
                stim = natural_image(**vars(args))
            elif (args.protocol['Stimulus']=='Natural-Image+VSE'):
                stim = natural_image_vse(**vars(args))
            elif (args.protocol['Stimulus']=='sparse-noise'):
                if args.protocol['Presentation']=='Single-Stimulus':
                    stim = sparse_noise(**vars(args))
                else:
                    print('Noise stim have to be done as "Single-Stimulus" !')
            elif (args.protocol['Stimulus']=='dense-noise'):
                if args.protocol['Presentation']=='Single-Stimulus':
                    stim = dense_noise(**vars(args))
                else:
                    print('Noise stim have to be done as "Single-Stimulus" !')
            else:
                print('Protocol not recognized !')
                stim = None
            
            # common to all protocols
            if stim is not None:
                stim.generate_movie()
                stim.write_protocol(args.protocol_file)

        if task in ['visualize', 'see', 'both']:
            args.task = 'see'
            stim = visual_stim(**vars(args))
            stim.preload_movie()
            parent = dummy_parent()
            stim.init_presentation()
            stim.run(parent=parent)
            
    else:
        print('Need to pass a valid ')


