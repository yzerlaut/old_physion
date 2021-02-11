from psychopy import visual, core, event, clock, monitors, tools # some libraries from PsychoPy
import numpy as np
import itertools, os, sys, pathlib, subprocess, time
 
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from screens import SCREENS
from psychopy_code.noise import sparse_noise_generator, dense_noise_generator
from psychopy_code.preprocess_NI import load, img_after_hist_normalization

def build_stim(protocol):
    """
    """
    if (protocol['Presentation']=='multiprotocol'):
        return multiprotocol(protocol)
    elif (protocol['Stimulus']=='light-level'):
        return light_level_single_stim(protocol)
    elif (protocol['Stimulus']=='full-field-grating'):
        return full_field_grating_stim(protocol)
    elif (protocol['Stimulus']=='center-grating'):
        return center_grating_stim(protocol)
    elif (protocol['Stimulus']=='off-center-grating'):
        return off_center_grating_stim(protocol)
    elif (protocol['Stimulus']=='surround-grating'):
        return surround_grating_stim(protocol)
    elif (protocol['Stimulus']=='drifting-full-field-grating'):
        return drifting_full_field_grating_stim(protocol)
    elif (protocol['Stimulus']=='drifting-center-grating'):
        return drifting_center_grating_stim(protocol)
    elif (protocol['Stimulus']=='drifting-off-center-grating'):
        return drifting_off_center_grating_stim(protocol)
    elif (protocol['Stimulus']=='drifting-surround-grating'):
        return drifting_surround_grating_stim(protocol)
    elif (protocol['Stimulus']=='Natural-Image'):
        return natural_image(protocol)
    elif (protocol['Stimulus']=='Natural-Image+VSE'):
        return natural_image_vse(protocol)
    elif (protocol['Stimulus']=='sparse-noise'):
        if protocol['Presentation']=='Single-Stimulus':
            return sparse_noise(protocol)
        else:
            print('Noise stim have to be done as "Single-Stimulus" !')
    elif (protocol['Stimulus']=='dense-noise'):
        if protocol['Presentation']=='Single-Stimulus':
            return dense_noise(protocol)
        else:
            print('Noise stim have to be done as "Single-Stimulus" !')
    elif (protocol['Stimulus']=='gaussian-blobs'):
        return gaussian_blobs(protocol)
    else:
        print('Protocol not recognized !')
        return None

    
def stop_signal(parent):
    if (len(event.getKeys())>0) or (parent.stop_flag):
        parent.stop_flag = True
        if hasattr(parent, 'statusBar'):
            parent.statusBar.showMessage('stimulation stopped !')
        return True
    else:
        return False


class visual_stim:

    def __init__(self, protocol, demo=False, store_frame=False):
        """
        """
        self.protocol = protocol
        self.store_frame = store_frame
        if ('store_frame' in protocol):
            self.store_frame = bool(protocol['store_frame'])
        
        self.screen = SCREENS[self.protocol['screen']]
        
        if not ('no-window' in self.protocol):
            self.monitor = monitors.Monitor(self.screen['name'])
            self.monitor.setDistance(self.screen['distance_from_eye'])
            self.win = visual.Window(self.screen['resolution'], monitor=self.monitor,
                                     screen=self.screen['screen_id'], fullscr=self.screen['fullscreen'],
                                     units='pix', color=-1)

            self.k, self.gamma = self.screen['gamma_correction']['k'], self.screen['gamma_correction']['gamma']

            # blank screens
            self.blank_start = visual.GratingStim(win=self.win, size=10000, pos=[0,0], sf=0,
                                                  color=self.protocol['presentation-prestim-screen'], units='pix')
            if 'presentation-interstim-screen' in self.protocol:
                self.blank_inter = visual.GratingStim(win=self.win, size=10000, pos=[0,0], sf=0,
                                                  color=self.protocol['presentation-interstim-screen'], units='pix')
            self.blank_end = visual.GratingStim(win=self.win, size=10000, pos=[0,0], sf=0,
                                    color=self.protocol['presentation-poststim-screen'], units='pix')

            if self.screen['monitoring_square']['location']=='top-right':
                pos = [int(x/2.-self.screen['monitoring_square']['size']/2.) for x in self.screen['resolution']]
            elif self.screen['monitoring_square']['location']=='bottom-left':
                pos = [int(-x/2.+self.screen['monitoring_square']['size']/2.) for x in self.screen['resolution']]
            elif self.screen['monitoring_square']['location']=='top-left':
                pos = [int(-self.screen['resolution'][0]/2.+self.screen['monitoring_square']['size']/2.),
                       int(self.screen['resolution'][1]/2.-self.screen['monitoring_square']['size']/2.)]
            elif self.screen['monitoring_square']['location']=='bottom-right':
                pos = [int(self.screen['resolution'][0]/2.-self.screen['monitoring_square']['size']/2.),
                       int(-self.screen['resolution'][1]/2.+self.screen['monitoring_square']['size']/2.)]
            else:
                print(30*'-'+'\n /!\ monitoring square location not recognized !!')

            self.on = visual.GratingStim(win=self.win, size=self.screen['monitoring_square']['size'], pos=pos, sf=0,
                                         color=self.screen['monitoring_square']['color-on'], units='pix')
            self.off = visual.GratingStim(win=self.win, size=self.screen['monitoring_square']['size'],  pos=pos, sf=0,
                                          color=self.screen['monitoring_square']['color-off'], units='pix')

            # initialize the times for the monitoring signals
            self.Ton = int(1e3*self.screen['monitoring_square']['time-on'])
            self.Toff = int(1e3*self.screen['monitoring_square']['time-off'])
            self.Tfull, self.Tfull_first = int(self.Ton+self.Toff), int((self.Ton+self.Toff)/2.)

        
    ################################
    #  ---   Gamma correction  --- #
    ################################
    def gamma_corrected_lum(self, level):
        return 2*np.power(((level+1.)/2./self.k), 1./self.gamma)-1.
    def gamma_corrected_contrast(self, contrast):
        return np.power(contrast/self.k, 1./self.gamma)

    
    ################################
    #  ---       Geometry      --- #
    ################################
    def pixel_meshgrid(self):
        return np.meshgrid(np.arange(self.screen['resolution'][0]),
                           np.arange(self.screen['resolution'][1]))
    
    def cm_to_angle(self, value):
        return 180./np.pi*np.arctan(value/self.screen['distance_from_eye'])
    
    def pix_to_angle(self, value):
        return self.cm_to_angle(value/self.screen['resolution'][0]*self.screen['width'])
    
    def angle_meshgrid(self):
        x = np.linspace(self.cm_to_angle(-self.screen['width']/2.),
                        self.cm_to_angle(self.screen['width']/2.),
                        self.screen['resolution'][0])
        z = np.linspace(self.cm_to_angle(-self.screen['width']/2.)*self.screen['resolution'][1]/self.screen['resolution'][0],
                        self.cm_to_angle(self.screen['width']/2.)*self.screen['resolution'][1]/self.screen['resolution'][0],
                        self.screen['resolution'][1])
        return np.meshgrid(x, z)

    def angle_to_cm(self, value):
        return self.screen['distance_from_eye']*np.tan(np.pi/180.*value)
    
    def angle_to_pix(self, value):
        return self.screen['resolution'][0]/self.screen['width']*\
            self.angle_to_cm(value)

                           
    ################################
    #  ---     Experiment      --- #
    ################################

    def init_experiment(self, protocol, keys, run_type='static'):

        self.experiment, self.PATTERNS = {}, []

        if protocol['Presentation']=='Single-Stimulus':
            for key in protocol:
                if key.split(' (')[0] in keys:
                    self.experiment[key.split(' (')[0]] = [protocol[key]]
                    self.experiment['index'] = [0]
                    self.experiment['frame_run_type'] = [run_type]
                    self.experiment['index'] = [0]
                    self.experiment['time_start'] = [protocol['presentation-prestim-period']]
                    self.experiment['time_stop'] = [protocol['presentation-duration']+protocol['presentation-prestim-period']]
                    self.experiment['time_duration'] = [protocol['presentation-duration']]
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
            self.experiment['time_start'], self.experiment['time_stop'], self.experiment['time_duration'] = [], [], []
            self.experiment['frame_run_type'] = []

            index_no_repeat = np.arange(len(FULL_VECS[key]))

            # SHUFFLING IF NECESSARY
            if (protocol['Presentation']=='Randomized-Sequence'):
                np.random.seed(protocol['shuffling-seed'])
                np.random.shuffle(index_no_repeat)
                
            Nrepeats = max([1,protocol['N-repeat']])
            index = np.concatenate([index_no_repeat for r in range(Nrepeats)])
            repeat = np.concatenate([r+0*index_no_repeat for r in range(Nrepeats)])

            for n, i in enumerate(index[protocol['starting-index']:]):
                for key in keys:
                    self.experiment[key].append(FULL_VECS[key][i])
                self.experiment['index'].append(i)
                self.experiment['repeat'].append(repeat[n+protocol['starting-index']])
                self.experiment['time_start'].append(protocol['presentation-prestim-period']+\
                                                     n*protocol['presentation-duration']+n*protocol['presentation-interstim-period'])
                self.experiment['time_stop'].append(protocol['presentation-prestim-period']+\
                                                     (n+1)*protocol['presentation-duration']+n*protocol['presentation-interstim-period'])
                self.experiment['time_duration'].append(protocol['presentation-duration'])
                self.experiment['frame_run_type'].append(run_type)

    # the close function
    def close(self):
        self.win.close()

    def quit(self):
        core.quit()

    # screen at start
    def start_screen(self, parent):
        if not parent.stop_flag:
            self.blank_start.draw()
            self.off.draw()
            try:
                self.win.flip()
                if self.store_frame:
                    self.win.getMovieFrame() # we store the last frame
            except AttributeError:
                pass
            clock.wait(self.protocol['presentation-prestim-period'])

    # screen at end
    def end_screen(self, parent):
        if not parent.stop_flag:
            self.blank_end.draw()
            self.off.draw()
            try:
                self.win.flip()
                if self.store_frame:
                    self.win.getMovieFrame() # we store the last frame
            except AttributeError:
                pass
            clock.wait(self.protocol['presentation-poststim-period'])

    # screen for interstim
    def inter_screen(self, parent):
        if not parent.stop_flag:
            self.blank_inter.draw()
            self.off.draw()
            try:
                self.win.flip()
                if self.store_frame:
                    self.win.getMovieFrame() # we store the last frame
            except AttributeError:
                pass
            clock.wait(self.protocol['presentation-interstim-period'])

    # blinking in one corner
    def add_monitoring_signal(self, new_t, start):
        """ Pulses of length Ton at the times : [0, 0.5, 1, 2, 3, 4, ...] """
        if (int(1e3*new_t-1e3*start)<self.Tfull) and (int(1e3*new_t-1e3*start)%self.Tfull_first<self.Ton):
            self.on.draw()
        elif int(1e3*new_t-1e3*start)%self.Tfull<self.Ton:
            self.on.draw()
        else:
            self.off.draw()

    def add_monitoring_signal_sp(self, new_t, start):
        """ Single pulse monitoring signal (see array_run) """
        if (int(1e3*new_t-1e3*start)<self.Ton):
            self.on.draw()
        else:
            self.off.draw()

    ##########################################################
    #############      PRESENTING STIMULI    #################
    ##########################################################

    
    #####################################################
    # showing a single static pattern
    def single_static_pattern_presentation(self, parent, index):
        start = clock.getTime()
        patterns = self.get_patterns(index)
        while ((clock.getTime()-start)<self.experiment['time_duration'][index]) and not parent.stop_flag:
            for pattern in patterns:
                pattern.draw()
            if (int(1e3*clock.getTime()-1e3*start)<self.Tfull) and\
               (int(1e3*clock.getTime()-1e3*start)%self.Tfull_first<self.Ton):
                self.on.draw()
            elif int(1e3*clock.getTime()-1e3*start)%self.Tfull<self.Ton:
                self.on.draw()
            else:
                self.off.draw()
            try:
                self.win.flip()
            except AttributeError:
                pass

    #######################################################
    # showing a single dynamic pattern with a phase advance
    def single_dynamic_grating_presentation(self, parent, index):
        start, prev_t = clock.getTime(), clock.getTime()
        patterns = self.get_patterns(index)
        self.speed = self.experiment['speed'][index]
        while ((clock.getTime()-start)<self.experiment['time_duration'][index]) and not parent.stop_flag:
            new_t = clock.getTime()
            for pattern in patterns:
                pattern.setPhase(self.speed*(new_t-prev_t), '+') # advance phase
                pattern.draw()
            self.add_monitoring_signal(new_t, start)
            prev_t = new_t
            try:
                self.win.flip()
            except AttributeError:
                pass
        self.win.getMovieFrame() # we store the last frame

    #####################################################
    # adding a run purely define by an array (time, x, y), see e.g. sparse_noise initialization
    def single_array_presentation(self, parent, index):
        pattern = visual.ImageStim(self.win,
                                   image=self.gamma_corrected_lum(self.get_frame(index)),
                                   units='pix', size=self.win.size)
        start = clock.getTime()
        while ((clock.getTime()-start)<(self.experiment['time_duration'][index])) and not parent.stop_flag:
            pattern.draw()
            self.add_monitoring_signal_sp(clock.getTime(), start)
            try:
                self.win.flip()
            except AttributeError:
                pass


    #####################################################
    # adding a run purely define by an array (time, x, y), see e.g. sparse_noise initialization
    def single_array_sequence_presentation(self, parent, index):
        time_indices, frames = self.get_frames_sequence(index)
        FRAMES = []
        for frame in frames:
            FRAMES.append(visual.ImageStim(self.win,
                                           image=self.gamma_corrected_lum(frame),
                                           units='pix', size=self.win.size))
        start = clock.getTime()
        while ((clock.getTime()-start)<(self.experiment['time_duration'][index])) and not parent.stop_flag:
            iframe = int((clock.getTime()-start)*self.frame_refresh)
            FRAMES[time_indices[iframe]].draw()
            self.add_monitoring_signal(clock.getTime(), start)
            try:
                self.win.flip()
            except AttributeError:
                pass


    def single_episode_run(self, parent, index):
        
        if self.experiment['frame_run_type'][index]=='drifting':
            self.single_dynamic_grating_presentation(parent, index)
        elif self.experiment['frame_run_type'][index]=='image':
            self.single_array_presentation(parent, index)
        elif self.experiment['frame_run_type'][index]=='images_sequence':
            self.single_array_sequence_presentation(parent, index)
        else: # static by defaults
            self.single_static_pattern_presentation(parent, index)
            
        # we store the last frame if needed
        if self.store_frame:
            self.win.getMovieFrame() 

        
    ## FINAL RUN FUNCTION
    def run(self, parent):
        self.start_screen(parent)
        for i in range(len(self.experiment['index'])):
            if stop_signal(parent):
                break
            print('Running protocol of index %i/%i' % (i+1, len(self.experiment['index'])))
            self.single_episode_run(parent, i)
            if i<(len(self.experiment['index'])-1):
                self.inter_screen(parent)
        self.end_screen(parent)
        if not parent.stop_flag and hasattr(parent, 'statusBar'):
            parent.statusBar.showMessage('stimulation over !')
        
        if self.store_frame:
            self.win.saveMovieFrames(os.path.join(str(parent.datafolder.get()),
                                                  'screen-frames', 'frame.tiff'))
        

        
#####################################################
##  ----         MULTI-PROTOCOLS            --- #####           
#####################################################

class multiprotocol(visual_stim):

    def __init__(self, protocol):
        
        super().__init__(protocol)

        if 'movie_refresh_freq' not in protocol:
            protocol['movie_refresh_freq'] = 30.
        if 'appearance_threshold' not in protocol:
            protocol['appearance_threshold'] = 2.5 # 
        self.frame_refresh = protocol['movie_refresh_freq']

        
        self.STIM = []
        i=1
        while 'Protocol-%i'%i in protocol:
            Ppath = os.path.join(protocol['protocol-folder'], protocol['Protocol-%i'%i])
            with open(Ppath, 'r') as fp:
                subprotocol = json.load(fp)
                subprotocol['screen'] = protocol['screen']
                subprotocol['no-window'] = True
                self.STIM.append(build_stim(subprotocol))
            i+=1

        self.experiment = {'time_duration':[],
                           'protocol_id':[]}
        # we initialize the keys
        for stim in self.STIM:
            for key in stim.experiment:
                self.experiment[key] = []
        # then we iterate over values
        for IS, stim in enumerate(self.STIM):
            for i in range(len(stim.experiment['index'])):
                for key in self.experiment:
                    if key in stim.experiment:
                        self.experiment[key].append(stim.experiment[key][i])
                    elif key not in ['protocol_id', 'time_duration']:
                        self.experiment[key].append(None)
                self.experiment['protocol_id'].append(IS)
                self.experiment['time_duration'].append(stim.experiment['time_stop'][i]-stim.experiment['time_start'][i])
        # we rebuild time
        self.experiment['time_start'][0] = protocol['presentation-prestim-period']
        self.experiment['time_stop'][0] = protocol['presentation-prestim-period']+self.experiment['time_duration'][0]
        for i in range(1, len(self.experiment['index'])):
            self.experiment['time_start'][i] = self.experiment['time_stop'][i-1]+protocol['presentation-interstim-period']
            self.experiment['time_stop'][i] = self.experiment['time_start'][i]+self.experiment['time_duration'][i]
        print(self.experiment)
        print(self.protocol)

        
    # functions implemented in child class
    def get_frame(self, index):
        return self.STIM[self.experiment['protocol_id'][index]].get_frame(index, parent=self)
    def get_patterns(self, index):
        return self.STIM[self.experiment['protocol_id'][index]].get_patterns(index, parent=self)
    def get_frames_sequence(self, index):
        return self.STIM[self.experiment['protocol_id'][index]].get_frames_sequence(index, parent=self)


#####################################################
##  ----   PRESENTING VARIOUS LIGHT LEVELS  --- #####           
#####################################################

class light_level_single_stim(visual_stim):

    def __init__(self, protocol):
        
        super().__init__(protocol)
        super().init_experiment(protocol, ['light-level'], run_type='static')
            
    def get_patterns(self, index, parent=None):
        if parent is not None:
            cls = parent
        else:
            cls = self
        return [visual.GratingStim(win=cls.win,
                                   size=10000, pos=[0,0], sf=0, units='pix',
                                   color=cls.gamma_corrected_lum(cls.experiment['light-level'][index]))]

            
#####################################################
##  ----   PRESENTING FULL FIELD GRATINGS   --- #####           
#####################################################

class full_field_grating_stim(visual_stim):

    def __init__(self, protocol):
        super().__init__(protocol)
        super().init_experiment(protocol, ['spatial-freq', 'angle', 'contrast'], run_type='static')

    def get_patterns(self, index, parent=None):
        if parent is not None:
            cls = parent
        else:
            cls = self
        return [visual.GratingStim(win=cls.win,
                                   size=10000, pos=[0,0], units='pix',
                                   sf=cls.angle_to_pix(cls.experiment['spatial-freq'][index]),
                                   ori=cls.experiment['angle'][index],
                                   contrast=cls.gamma_corrected_contrast(cls.experiment['contrast'][index]))]
                                 
            
class drifting_full_field_grating_stim(visual_stim):

    def __init__(self, protocol):
        super().__init__(protocol)
        super().init_experiment(protocol, ['spatial-freq', 'angle', 'contrast', 'speed'], run_type='drifting')

    def get_patterns(self, index, parent=None):
        if parent is not None:
            cls = parent
        else:
            cls = self
        return [visual.GratingStim(win=cls.win,
                                   size=10000, pos=[0,0], units='pix',
                                   sf=1./cls.angle_to_pix(1./cls.experiment['spatial-freq'][index]),
                                   ori=cls.experiment['angle'][index],
                                   contrast=cls.gamma_corrected_contrast(cls.experiment['contrast'][index]))]

        
#####################################################
##  ----    PRESENTING CENTERED GRATINGS    --- #####           
#####################################################

class center_grating_stim(visual_stim):
    
    def __init__(self, protocol):
        super().__init__(protocol)
        super().init_experiment(protocol, ['x-center', 'y-center', 'radius','spatial-freq', 'angle', 'contrast'], run_type='static')

    def get_patterns(self, index, parent=None):
        if parent is not None:
            cls = parent
        else:
            cls = self
        return [visual.GratingStim(win=cls.win,
                                   pos=[cls.angle_to_pix(cls.experiment['x-center'][index]),
                                        cls.angle_to_pix(cls.experiment['y-center'][index])],
                                   sf=1./cls.angle_to_pix(1./cls.experiment['spatial-freq'][index]),
                                   size= 2*cls.angle_to_pix(cls.experiment['radius'][index]),
                                   ori=cls.experiment['angle'][index], units='pix',
                                   mask='circle',
                                   contrast=cls.gamma_corrected_contrast(cls.experiment['contrast'][index]))]

class drifting_center_grating_stim(visual_stim):
    
    def __init__(self, protocol):
        super().__init__(protocol)
        super().init_experiment(protocol, ['x-center', 'y-center', 'radius','spatial-freq', 'angle', 'contrast', 'speed', 'bg-color'], run_type='static')

    def get_patterns(self, index, parent=None):
        if parent is not None:
            cls = parent
        else:
            cls = self
        return [visual.GratingStim(win=cls.win,
                                   pos=[cls.angle_to_pix(cls.experiment['x-center'][index]),
                                        cls.angle_to_pix(cls.experiment['y-center'][index])],
                                   sf=1./cls.angle_to_pix(1./cls.experiment['spatial-freq'][index]),
                                   size= 2*cls.angle_to_pix(cls.experiment['radius'][index]),
                                   ori=cls.experiment['angle'][index], units='pix',
                                   mask='circle',
                                   contrast=cls.gamma_corrected_contrast(cls.experiment['contrast'][index]))]


#####################################################
##  ----    PRESENTING OFF-CENTERED GRATINGS    --- #####           
#####################################################

class off_center_grating_stim(visual_stim):
    
    def __init__(self, protocol):
        super().__init__(protocol)
        super().init_experiment(protocol, ['x-center', 'y-center', 'radius','spatial-freq', 'angle', 'contrast', 'bg-color'], run_type='static')

        
    def get_patterns(self, index, parent=None):
        if parent is not None:
            cls = parent
        else:
            cls = self
        return [visual.GratingStim(win=cls.win,
                                   size=10000, pos=[0,0], units='pix',
                                   sf=cls.experiment['spatial-freq'][index],
                                   ori=cls.experiment['angle'][index],
                                   contrast=cls.gamma_corrected_contrast(cls.experiment['contrast'][index])),
                visual.GratingStim(win=cls.win,
                                   pos=[cls.angle_to_pix(cls.experiment['x-center'][index]),
                                        cls.angle_to_pix(cls.experiment['y-center'][index])],
                                   sf=1./cls.angle_to_pix(1./cls.experiment['spatial-freq'][index]),
                                   size= 2*cls.angle_to_pix(cls.experiment['radius'][index]),
                                   mask='circle', units='pix',
                                   color=cls.gamma_corrected_lum(cls.experiment['bg-color'][index]))]

            
class drifting_off_center_grating_stim(visual_stim):
    
    def __init__(self, protocol):
        super().__init__(protocol)
        super().init_experiment(protocol, ['x-center', 'y-center', 'radius','spatial-freq', 'angle', 'contrast', 'bg-color', 'speed'], run_type='drifting')

    def get_patterns(self, index, parent=None):
        if parent is not None:
            cls = parent
        else:
            cls = self
        return [\
                # Surround grating
                visual.GratingStim(win=cls.win,
                                   size=1000, pos=[0,0],
                                   sf=cls.experiment['spatial-freq'][index],
                                   ori=cls.experiment['angle'][index],
                                   contrast=cls.gamma_corrected_contrast(cls.experiment['contrast'][index])),
                # + center Mask
                visual.GratingStim(win=cls.win,
                                   pos=[cls.experiment['x-center'][index], cls.experiment['y-center'][index]],
                                   size=cls.experiment['radius'][index],
                                   mask='circle', sf=0, contrast=0,
                                   color=cls.gamma_corrected_lum(cls.experiment['bg-color'][index]))]


#####################################################
##  ----    PRESENTING SURROUND GRATINGS    --- #####           
#####################################################
        
class surround_grating_stim(visual_stim):

    def __init__(self, protocol):
        super().__init__(protocol)
        super().init_experiment(protocol, ['x-center', 'y-center', 'radius-start', 'radius-end','spatial-freq', 'angle', 'contrast', 'bg-color'], run_type='static')

    def get_patterns(self, index, parent=None):
        if parent is not None:
            cls = parent
        else:
            cls = self
        return [\
                visual.GratingStim(win=cls.win,
                                   size=1000, pos=[0,0], sf=0,
                                   color=cls.gamma_corrected_lum(cls.experiment['bg-color'][index])),
                visual.GratingStim(win=cls.win,
                                   pos=[cls.experiment['x-center'][index], cls.experiment['y-center'][index]],
                                   size=cls.experiment['radius-end'][index],
                                   mask='circle', 
                                   sf=cls.experiment['spatial-freq'][index],
                                   ori=cls.experiment['angle'][index],
                                   contrast=cls.gamma_corrected_contrast(cls.experiment['contrast'][index])),
                visual.GratingStim(win=cls.win,
                                   pos=[cls.experiment['x-center'][index], cls.experiment['y-center'][index]],
                                   size=cls.experiment['radius-start'][index],
                                   mask='circle', sf=0,
                                   color=cls.gamma_corrected_lum(cls.experiment['bg-color'][index]))]
    

class drifting_surround_grating_stim(visual_stim):

    def __init__(self, protocol):
        super().__init__(protocol)
        super().init_experiment(protocol, ['x-center', 'y-center', 'radius-start', 'radius-end','spatial-freq', 'angle', 'contrast', 'bg-color', 'speed'], run_type='drifting')

    def get_patterns(self, index, parent=None):
        if parent is not None:
            cls = parent
        else:
            cls = self
        return [\
                visual.GratingStim(win=cls.win,
                                   size=1000, pos=[0,0], sf=0,contrast=0,
                                   color=cls.gamma_corrected_lum(cls.experiment['bg-color'][index])),
                visual.GratingStim(win=cls.win,
                                   pos=[cls.experiment['x-center'][index], cls.experiment['y-center'][index]],
                                   size=cls.experiment['radius-end'][index],
                                   mask='circle', 
                                   sf=cls.experiment['spatial-freq'][index],
                                   ori=cls.experiment['angle'][index],
                                   contrast=cls.gamma_corrected_contrast(cls.experiment['contrast'][index])),
                visual.GratingStim(win=cls.win,
                                   pos=[cls.experiment['x-center'][index], cls.experiment['y-center'][index]],
                                   size=cls.experiment['radius-start'][index],
                                   mask='circle', sf=0,contrast=0,
                                   color=cls.gamma_corrected_lum(cls.experiment['bg-color'][index]))]
        

#####################################################
##  -- PRESENTING APPEARING GAUSSIAN BLOBS  --  #####           
#####################################################

class gaussian_blobs(visual_stim):
    
    def __init__(self, protocol):

        if 'movie_refresh_freq' not in protocol:
            protocol['movie_refresh_freq'] = 30.
        if 'appearance_threshold' not in protocol:
            protocol['appearance_threshold'] = 2.5 # 
        self.frame_refresh = protocol['movie_refresh_freq']
        
        super().__init__(protocol)
        super().init_experiment(self.protocol,
                                ['x-center', 'y-center', 'radius','center-time', 'extent-time', 'contrast', 'bg-color'],
                                run_type='images_sequence')
        
            
    def get_frames_sequence(self, index, parent=None):
        if parent is not None:
            cls = parent
        else:
            cls = self
        """
        Generator creating a random number of chunks (but at most max_chunks) of length chunk_length containing
        random samples of sin([0, 2pi]).
        """
        x, z = cls.angle_meshgrid()
        
        bg = np.ones(cls.screen['resolution'])*cls.experiment['bg-color'][index]
        interval = cls.experiment['time_stop'][index]-cls.experiment['time_start'][index]

        contrast = cls.experiment['contrast'][index]
        xcenter, zcenter = cls.experiment['x-center'][index], cls.experiment['y-center'][index]
        radius = cls.experiment['radius'][index]
        bg_color = cls.experiment['bg-color'][index]
        
        t0, sT = cls.experiment['center-time'][index], cls.experiment['extent-time'][index]
        itstart = np.max([0, int((t0-cls.protocol['appearance_threshold']*sT)*cls.protocol['movie_refresh_freq'])])
        itend = np.min([int(interval*cls.protocol['movie_refresh_freq']),
                        int((t0+cls.protocol['appearance_threshold']*sT)*cls.protocol['movie_refresh_freq'])])

        times, FRAMES = np.zeros(int(1.2*interval*cls.protocol['movie_refresh_freq']), dtype=int), []
        # the pre-time
        FRAMES.append(2*bg_color-1.+0.*x)
        times[:itstart] = 0
        for iframe, it in enumerate(np.arange(itstart, itend)):
            img = 2*(np.exp(-((x-xcenter)**2+(z-zcenter)**2)/2./radius**2)*\
                     contrast*np.exp(-(it/cls.protocol['movie_refresh_freq']-t0)**2/2./sT**2)+bg_color)-1.
            FRAMES.append(img)
            times[it] = iframe
        # the post-time
        FRAMES.append(2*bg_color-1.+0.*x)
        times[itend:] = len(FRAMES)-1
            
        return times, FRAMES


#####################################################
##  ----    PRESENTING NATURAL IMAGES       --- #####
#####################################################

NI_directory = os.path.join(str(pathlib.Path(__file__).resolve().parents[1]), 'NI_bank')
        
class natural_image(visual_stim):

    def __init__(self, protocol):
        super().__init__(protocol)
        super().init_experiment(protocol, ['Image-ID'], run_type='image')

    def get_frame(self, index, parent=None):
        if parent is not None:
            cls = parent
        else:
            cls = self
        filename = os.listdir(NI_directory)[int(cls.experiment['Image-ID'][index])]
        img = load(os.path.join(NI_directory, filename))
        return 2*img_after_hist_normalization(img).T-1.

            
#####################################################
##  --    WITH VIRTUAL SCENE EXPLORATION    --- #####
#####################################################


def generate_VSE(duration=5,
                 mean_saccade_duration=2.,# in s
                 std_saccade_duration=1.,# in s
                 saccade_amplitude=100, # in pixels, TO BE PUT IN DEGREES
                 seed=0):
    """
    to do: clean up the VSE generator
    """
    print('generating Virtual-Scene-Exploration [...]')
    
    np.random.seed(seed)
    
    tsaccades = np.cumsum(np.clip(mean_saccade_duration+np.abs(np.random.randn(int(1.5*duration/mean_saccade_duration))*std_saccade_duration),
                                  mean_saccade_duration/2., 1.5*mean_saccade_duration))

    x = np.array(np.clip((np.random.randn(len(tsaccades))+1)*saccade_amplitude, 0, 2*saccade_amplitude), dtype=int)
    y = np.array(np.clip((np.random.randn(len(tsaccades))+1)*saccade_amplitude, 0, 2*saccade_amplitude), dtype=int)
    
    return {'t':np.array([0]+list(tsaccades)),
            'x':np.array([0]+list(x)),
            'y':np.array([0]+list(y)),
            'max_amplitude':saccade_amplitude}

            

class natural_image_vse(visual_stim):

    
    def __init__(self, protocol):

        super().__init__(protocol)
        super().init_experiment(protocol, ['Image-ID', 'VSE-seed',
                                           'mean-saccade-duration', 'std-saccade-duration',
                                           'vary-VSE-with-Image', 'saccade-amplitude'],
                                run_type='images_sequence')

        if 'movie_refresh_freq' not in protocol:
            protocol['movie_refresh_freq'] = 30.
        self.frame_refresh = protocol['movie_refresh_freq']
        
        
    def get_frames_sequence(self, index, parent=None):
        if parent is not None:
            cls = parent
        else:
            cls = self

        if cls.experiment['vary-VSE-with-Image'][index]==1:
            seed = int(cls.experiment['VSE-seed'][index]+1000*cls.experiment['Image-ID'][index])
        else:
            seed = int(cls.experiment['VSE-seed'][index])

        vse = generate_VSE(duration=cls.protocol['presentation-duration'],
                           mean_saccade_duration=cls.experiment['mean-saccade-duration'][index],
                           std_saccade_duration=cls.experiment['std-saccade-duration'][index],
                           saccade_amplitude=cls.angle_to_pix(cls.experiment['saccade-amplitude'][index]), # in pixels, TO BE PUT IN DEGREES
                           seed=seed)

        filename = os.listdir(NI_directory)[int(cls.experiment['Image-ID'][index])]
        img0 = load(os.path.join(NI_directory, filename))
        img = 2*img_after_hist_normalization(img0)-1 # normalization + gamma_correction
        sx, sy = img.shape
            
        interval = cls.experiment['time_stop'][index]-cls.experiment['time_start'][index]
        times, FRAMES = np.zeros(int(1.2*interval*cls.protocol['movie_refresh_freq']), dtype=int), []
        Times = np.arange(int(1.2*interval*cls.protocol['movie_refresh_freq']))/cls.protocol['movie_refresh_freq']

        for i, t in enumerate(vse['t']):
            ix, iy = int(vse['x'][i]), int(vse['y'][i])
            new_im = np.zeros(img.shape)
            new_im[ix:,iy:] = img[:sx-ix,:sy-iy]
            new_im[:ix,:] = img[sx-ix:,:]
            new_im[:,:iy] = img[:,sy-iy:]
            new_im[:ix,:iy] = img[sx-ix:,sy-iy:]
            FRAMES.append(new_im.T)
            times[Times>=t] = int(i)
            
        return times, FRAMES

    
#####################################################
##  ----    PRESENTING BINARY NOISE         --- #####
#####################################################

class sparse_noise(visual_stim):
    
    def __init__(self, protocol):

        super().__init__(protocol)
        super().init_experiment(protocol,
                                ['square-size', 'sparseness', 'mean-refresh-time', 'jitter-refresh-time'], run_type='image')

        self.noise_gen = sparse_noise_generator(duration=protocol['presentation-duration'],
                                                screen=self.screen,
                                                sparseness=protocol['sparseness (%)']/100.,
                                                square_size=protocol['square-size (deg)'],
                                                bg_color=protocol['bg-color (lum.)'],
                                                contrast=protocol['contrast (norm.)'],
                                                noise_mean_refresh_time=protocol['mean-refresh-time (s)'],
                                                noise_rdm_jitter_refresh_time=protocol['jitter-refresh-time (s)'],
                                                seed=protocol['noise-seed (#)'])

        self.experiment['time_start'] = self.noise_gen.events[:-1]
        self.experiment['time_stop'] = self.noise_gen.events[1:]
        
    def get_frame(self, index, parent=None):
        if parent is not None:
            cls = parent
        else:
            cls = self
        return self.noise_gen.get_frame(index).T
            

class dense_noise(visual_stim):

    def __init__(self, protocol):

        super().__init__(protocol)
        super().init_experiment(protocol,
                                ['square-size', 'sparseness', 'mean-refresh-time', 'jitter-refresh-time'], run_type='image')

        self.noise_gen = dense_noise_generator(duration=protocol['presentation-duration'],
                                                screen=self.screen,
                                                square_size=protocol['square-size (deg)'],
                                                contrast=protocol['contrast (norm.)'],
                                                noise_mean_refresh_time=protocol['mean-refresh-time (s)'],
                                                noise_rdm_jitter_refresh_time=protocol['jitter-refresh-time (s)'],
                                                seed=protocol['noise-seed (#)'])

        self.experiment['time_start'] = self.noise_gen.events[:-1]
        self.experiment['time_stop'] = self.noise_gen.events[1:]
        
    def get_frame(self, index, parent=None):
        if parent is not None:
            cls = parent
        else:
            cls = self
        return self.noise_gen.get_frame(index).T
            

if __name__=='__main__':

    import json, tempfile
    from pathlib import Path
    
    with open('exp/protocols/multiprotocols.json', 'r') as fp:
        protocol = json.load(fp)

    class df:
        def __init__(self):
            pass
        def get(self):
            Path(os.path.join(tempfile.gettempdir(), 'screen-frames')).mkdir(parents=True, exist_ok=True)
            return tempfile.gettempdir()
        
    class dummy_parent:
        def __init__(self):
            self.stop_flag = False
            self.datafolder = df()

    stim = build_stim(protocol)
    parent = dummy_parent()
    stim.run(parent)
    stim.close()
