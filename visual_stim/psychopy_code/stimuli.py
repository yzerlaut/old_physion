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
    if (protocol['Stimulus']=='light-level'):
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
        
        if 'screen' not in self.protocol:
            self.protocol['screen'] = 'Dell-P2018H'

        if self.protocol['Setup']=='demo-mode' or demo==True:
            # Everything is scaled-down by a factor 2
            self.monitor = monitors.Monitor('testMonitor')
            self.screen0 = SCREENS[self.protocol['screen']]
            self.screen = {}
            for key in self.screen0:
                self.screen[key] = self.screen0[key]
            # for key in ['resolution', 'distance_from_eye', 'width']:
            #     self.screen[key] = np.array(self.screen0[key])/1.6 # SCALE FACTOR HERE
            self.win = visual.Window(self.screen['resolution'], monitor=self.monitor,
                                     units='pix', color=-1) #create a window
        else:
            self.screen = SCREENS[self.protocol['screen']]
            self.monitor = monitors.Monitor(self.protocol['screen'])
            self.monitor.setDistance(self.screen['distance_from_eye'])
            self.win = visual.Window(self.screen['resolution'], monitor=self.monitor,
                                     screen=self.screen['screen_id'], fullscr=True,
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
        return np.meshgrid(np.arange(self.screen['resolution'][0],
                           np.arange(self.screen['resolution'][1])))
    
    def cm_to_angle(self, value):
        return 180./np.pi*np.arctan(value/self.screen['distance_from_eye'])
    
    def pix_to_angle(self, value):
        return self.cm_to_angle(value/self.screen['resolution'][0]*self.screen['width'])
    
    def angle_meshgrid(self):
        x = np.linspace(self.cm_to_angle(-self.screen['width']/2.),
                        self.cm_to_angle(self.screen['width']/2.),
                        self.screen['resolution'][0])
        z = np.linspace(self.cm_to_angle(-self.screen['height']/2.),
                        self.cm_to_angle(self.screen['height']/2.),
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
            self.experiment['run_type'] = []

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
                self.experiment['run_type'].append(run_type)

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
        while ((clock.getTime()-start)<self.protocol['presentation-duration']) and not parent.stop_flag:
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
        while ((clock.getTime()-start)<self.protocol['presentation-duration']) and not parent.stop_flag:
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
        while ((clock.getTime()-start)<(self.experiment['time_stop'][index]-\
                                        self.experiment['time_start'][index])) and not parent.stop_flag:
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
        while ((clock.getTime()-start)<(self.experiment['time_stop'][index]-\
                                        self.experiment['time_start'][index])) and not parent.stop_flag:
            index = int((clock.getTime()-start)*self.frame_refresh)
            PATTERNS[time_indices[index]].draw()
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
            print('skdjfskudfh')
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

        
    # def static_run(self, parent):
    #     self.start_screen(parent)
    #     for i in range(len(self.experiment['index'])):
    #         if stop_signal(parent):
    #             break
    #         print('Running protocol of index %i/%i' % (i+1, len(self.experiment['index'])))
    #         self.single_static_patterns_presentation(parent, i)
    #         if self.store_frame:
    #             self.win.getMovieFrame() # we store the last frame
    #         if self.protocol['Presentation']!='Single-Stimulus':
    #             self.inter_screen(parent)
    #     self.end_screen(parent)
    #     if not parent.stop_flag and hasattr(parent, 'statusBar'):
    #         parent.statusBar.showMessage('stimulation over !')
    #     if self.store_frame:
    #         self.win.saveMovieFrames(os.path.join(str(parent.datafolder.get()),
    #                                           'screen-frames', 'frame.tiff'))

            
    # def drifting_run(self, parent):
    #     self.start_screen(parent)
    #     for i in range(len(self.experiment['index'])):
    #         if stop_signal(parent):
    #             break
    #         print('Running protocol of index %i/%i' % (i+1, len(self.experiment['index'])))
    #         self.speed = self.experiment['speed'][i] # conversion to pixels
    #         self.single_dynamic_gratings_presentation(parent, i)
    #         if self.protocol['Presentation']!='Single-Stimulus':
    #             self.inter_screen(parent)
    #     self.end_screen(parent)
    #     if not parent.stop_flag and hasattr(parent, 'statusBar'):
    #         parent.statusBar.showMessage('stimulation over !')
    #     self.win.saveMovieFrames(os.path.join(str(parent.datafolder.get()),
    #                                           'screen-frames', 'frame.tiff'))

    # #####################################################
    # # adding a Virtual-Scene-Exploration on top of an image stim
    # def single_VSE_image_presentation(self, parent, index):
    #     start, prev_t = clock.getTime(), clock.getTime()
    #     while ((clock.getTime()-start)<self.protocol['presentation-duration']) and not parent.stop_flag:
    #         new_t = clock.getTime()
    #         i0 = np.min(np.argwhere(self.VSEs[index]['t']>(new_t-start)))
    #         self.PATTERNS[index][i0].draw()
    #         self.add_monitoring_signal(new_t, start)
    #         prev_t = new_t
    #         try:
    #             self.win.flip()
    #         except AttributeError:
    #             pass
    #     self.win.getMovieFrame() # we store the last frame

    # def vse_run(self, parent):
    #     self.start_screen(parent)
    #     for i in range(len(self.experiment['index'])):
    #         if stop_signal(parent):
    #             break
    #         print('Running protocol of index %i/%i' % (i+1, len(self.experiment['index'])))
    #         self.single_VSE_image_presentation(parent, i)
    #         if self.protocol['Presentation']!='Single-Stimulus':
    #             self.inter_screen(parent)
    #     self.end_screen(parent)
    #     if not parent.stop_flag and hasattr(parent, 'statusBar'):
    #         parent.statusBar.showMessage('stimulation over !')
    #     self.win.saveMovieFrames(os.path.join(str(parent.datafolder.get()),
    #                                           'screen-frames', 'frame.tiff'))
        

    # def array_run(self, parent):
    #     self.start_screen(parent)
    #     for i in range(len(self.experiment['time_start'])):
    #         if stop_signal(parent):
    #             break
    #         print('Running frame of index %i/%i' % (i+1, len(self.experiment['time_start'])))
    #         self.single_array_presentation(parent, i)
    #         if self.protocol['Presentation']!='Single-Stimulus':
    #             self.inter_screen(parent)
    #     self.end_screen(parent)
    #     if not parent.stop_flag and hasattr(parent, 'statusBar'):
    #         parent.statusBar.showMessage('stimulation over !')
    #     # self.win.saveMovieFrames(os.path.join(str(parent.datafolder.get()),
    #     #                                       'screen-frames', 'frame.tiff'))
            
    # ## FINAL RUN FUNCTION
    # def run(self, parent):
    #     if len(self.protocol['Stimulus'].split('drifting'))>1:
    #         return self.drifting_run(parent)
    #     elif len(self.protocol['Stimulus'].split('VSE'))>1:
    #         return self.vse_run(parent)
    #     elif len(self.protocol['Stimulus'].split('noise'))>1:
    #         return self.array_run(parent)
    #     else:
    #         return self.static_run(parent)
        
#####################################################
##  ----   PRESENTING VARIOUS LIGHT LEVELS  --- #####           
#####################################################

class light_level_single_stim(visual_stim):

    def __init__(self, protocol):
        
        super().__init__(protocol)
        super().init_experiment(protocol, ['light-level'])
        
            
    def get_patterns(self, index):
        return [visual.GratingStim(win=self.win,
                                   size=10000, pos=[0,0], sf=0, units='pix',
                                   color=self.gamma_corrected_lum(self.experiment['light-level'][index]))]
            
#####################################################
##  ----   PRESENTING FULL FIELD GRATINGS   --- #####           
#####################################################

class full_field_grating_stim(visual_stim):

    def __init__(self, protocol):
        super().__init__(protocol)
        super().init_experiment(protocol, ['spatial-freq', 'angle', 'contrast'])

    def get_patterns(self, index):
        return [visual.GratingStim(win=self.win,
                                   size=10000, pos=[0,0], units='pix',
                                   sf=self.angle_to_pix(self.experiment['spatial-freq'][index]),
                                   ori=self.experiment['angle'][index],
                                   contrast=self.gamma_corrected_contrast(self.experiment['contrast'][index]))]
                                 
            
class drifting_full_field_grating_stim(visual_stim):

    def __init__(self, protocol):
        super().__init__(protocol)
        super().init_experiment(protocol, ['spatial-freq', 'angle', 'contrast', 'speed'])

    def get_patterns(self, index):
        return [visual.GratingStim(win=self.win,
                                   size=10000, pos=[0,0], units='pix',
                                   sf=1./self.angle_to_pix(1./self.experiment['spatial-freq'][index]),
                                   ori=self.experiment['angle'][index],
                                   contrast=self.gamma_corrected_contrast(self.experiment['contrast'][index]))]

        
#####################################################
##  ----    PRESENTING CENTERED GRATINGS    --- #####           
#####################################################

class center_grating_stim(visual_stim):
    
    def __init__(self, protocol):
        super().__init__(protocol)
        super().init_experiment(protocol, ['x-center', 'y-center', 'radius','spatial-freq', 'angle', 'contrast'])

    def get_patterns(self, index):
        return [visual.GratingStim(win=self.win,
                                   pos=[self.angle_to_pix(self.experiment['x-center'][index]),
                                        self.angle_to_pix(self.experiment['y-center'][index])],
                                   sf=1./self.angle_to_pix(1./self.experiment['spatial-freq'][index]),
                                   size= 2*self.angle_to_pix(self.experiment['radius'][index]),
                                   ori=self.experiment['angle'][index], units='pix',
                                   mask='circle',
                                   contrast=self.gamma_corrected_contrast(self.experiment['contrast'][index]))]

class drifting_center_grating_stim(visual_stim):
    
    def __init__(self, protocol):
        super().__init__(protocol)
        super().init_experiment(protocol, ['x-center', 'y-center', 'radius','spatial-freq', 'angle', 'contrast', 'speed', 'bg-color'])

    def get_patterns(self, index):
        return [visual.GratingStim(win=self.win,
                                   pos=[self.angle_to_pix(self.experiment['x-center'][index]),
                                        self.angle_to_pix(self.experiment['y-center'][index])],
                                   sf=1./self.angle_to_pix(1./self.experiment['spatial-freq'][index]),
                                   size= 2*self.angle_to_pix(self.experiment['radius'][index]),
                                   ori=self.experiment['angle'][index], units='pix',
                                   mask='circle',
                                   contrast=self.gamma_corrected_contrast(self.experiment['contrast'][index]))]


#####################################################
##  ----    PRESENTING OFF-CENTERED GRATINGS    --- #####           
#####################################################

class off_center_grating_stim(visual_stim):
    
    def __init__(self, protocol):
        super().__init__(protocol)
        super().init_experiment(protocol, ['x-center', 'y-center', 'radius','spatial-freq', 'angle', 'contrast', 'bg-color'])

        
    def get_patterns(self, index):
        return [visual.GratingStim(win=self.win,
                                   size=10000, pos=[0,0], units='pix',
                                   sf=self.experiment['spatial-freq'][index],
                                   ori=self.experiment['angle'][index],
                                   contrast=self.gamma_corrected_contrast(self.experiment['contrast'][index])),
                visual.GratingStim(win=self.win,
                                   pos=[self.angle_to_pix(self.experiment['x-center'][index]),
                                        self.angle_to_pix(self.experiment['y-center'][index])],
                                   sf=1./self.angle_to_pix(1./self.experiment['spatial-freq'][index]),
                                   size= 2*self.angle_to_pix(self.experiment['radius'][index]),
                                   mask='circle', units='pix',
                                   color=self.gamma_corrected_lum(self.experiment['bg-color'][index]))]

            
class drifting_off_center_grating_stim(visual_stim):
    
    def __init__(self, protocol):
        super().__init__(protocol)
        super().init_experiment(protocol, ['x-center', 'y-center', 'radius','spatial-freq', 'angle', 'contrast', 'bg-color', 'speed'])

    def get_patterns(self, index):
        return [\
                # Surround grating
                visual.GratingStim(win=self.win,
                                   size=1000, pos=[0,0],
                                   sf=self.experiment['spatial-freq'][index],
                                   ori=self.experiment['angle'][index],
                                   contrast=self.gamma_corrected_contrast(self.experiment['contrast'][index])),
                # + center Mask
                visual.GratingStim(win=self.win,
                                   pos=[self.experiment['x-center'][index], self.experiment['y-center'][index]],
                                   size=self.experiment['radius'][index],
                                   mask='circle', sf=0, contrast=0,
                                   color=self.gamma_corrected_lum(self.experiment['bg-color'][index]))]


#####################################################
##  ----    PRESENTING SURROUND GRATINGS    --- #####           
#####################################################
        
class surround_grating_stim(visual_stim):

    def __init__(self, protocol):
        super().__init__(protocol)
        super().init_experiment(protocol, ['x-center', 'y-center', 'radius-start', 'radius-end','spatial-freq', 'angle', 'contrast', 'bg-color'])

    def get_patterns(self, index):
        return [\
                visual.GratingStim(win=self.win,
                                   size=1000, pos=[0,0], sf=0,
                                   color=self.gamma_corrected_lum(self.experiment['bg-color'][index])),
                visual.GratingStim(win=self.win,
                                   pos=[self.experiment['x-center'][index], self.experiment['y-center'][index]],
                                   size=self.experiment['radius-end'][index],
                                   mask='circle', 
                                   sf=self.experiment['spatial-freq'][index],
                                   ori=self.experiment['angle'][index],
                                   contrast=self.gamma_corrected_contrast(self.experiment['contrast'][index])),
                visual.GratingStim(win=self.win,
                                   pos=[self.experiment['x-center'][index], self.experiment['y-center'][index]],
                                   size=self.experiment['radius-start'][index],
                                   mask='circle', sf=0,
                                   color=self.gamma_corrected_lum(self.experiment['bg-color'][index]))]
    

class drifting_surround_grating_stim(visual_stim):

    def __init__(self, protocol):
        super().__init__(protocol)
        super().init_experiment(protocol, ['x-center', 'y-center', 'radius-start', 'radius-end','spatial-freq', 'angle', 'contrast', 'bg-color', 'speed'])

    def get_patterns(self, index):
        return [\
                visual.GratingStim(win=self.win,
                                   size=1000, pos=[0,0], sf=0,contrast=0,
                                   color=self.gamma_corrected_lum(self.experiment['bg-color'][index])),
                visual.GratingStim(win=self.win,
                                   pos=[self.experiment['x-center'][index], self.experiment['y-center'][index]],
                                   size=self.experiment['radius-end'][index],
                                   mask='circle', 
                                   sf=self.experiment['spatial-freq'][index],
                                   ori=self.experiment['angle'][index],
                                   contrast=self.gamma_corrected_contrast(self.experiment['contrast'][index])),
                visual.GratingStim(win=self.win,
                                   pos=[self.experiment['x-center'][index], self.experiment['y-center'][index]],
                                   size=self.experiment['radius-start'][index],
                                   mask='circle', sf=0,contrast=0,
                                   color=self.gamma_corrected_lum(self.experiment['bg-color'][index]))]
        

#####################################################
##  -- PRESENTING APPEARING GAUSSIAN BLOBS  --  #####           
#####################################################

class gaussian_blobs(visual_stim):
    
    def __init__(self, protocol):

        if 'movie_refresh_freq' not in protocol:
            protocol['movie_refresh_freq'] = 30.
        if 'appearance_threshold' not in protocol:
            protocol['appearance_threshold'] = 2.5 # 
        
        super().__init__(protocol)
        super().init_experiment(self.protocol,
                                ['x-center', 'y-center', 'radius','center-time', 'extent-time', 'contrast', 'bg-color'])
            
    def get_frames_sequence(self, index):
        """
        Generator creating a random number of chunks (but at most max_chunks) of length chunk_length containing
        random samples of sin([0, 2pi]).
        """
        # xp, zp = self.pixel_meshgrid()
        # x, z = self.horizontal_pix_to_angle(xp), self.vertical_pix_to_angle(zp)
        # # prestim
        # for i in range(int(self.protocol['presentation-prestim-period']*self.protocol['movie_refresh_freq'])):
        #     yield np.ones(x.shape)*(2*self.protocol['presentation-prestim-screen']-1)
        # for episode, start in enumerate(self.experiment['time_start']):

        bg = np.ones(self.screen['resolution'])*self.experiment['bg-color'][index]
        interval = self.experiment['time_stop'][index]-self.experiment['time_start'][index]

        contrast = self.experiment['contrast'][index]
        xcenter, zcenter = self.experiment['x-center'][index], self.experiment['y-center'][index]
        radius = self.experiment['radius'][index]
        bg_color = self.experiment['bg-color'][index]
        
        times = np.zeros(int(1.2*interval*self.protocol['movie_refresh_freq']))
        t0, sT = self.experiment['center-time'][index], self.experiment['extent-time'][index]
        itstart = np.max([0, int((t0-self.protocol['appearance_threshold']*sT)*self.protocol['movie_refresh_freq'])])
        itend = np.min([int(interval*self.protocol['movie_refresh_freq']),
                        int((t0+self.protocol['appearance_threshold']*sT)*self.protocol['movie_refresh_freq'])])
        for if, it in enumerate(np.arange(itstart, itend)):
            img = 2*(np.exp(-((x-xcenter)**2+(z-zcenter)**2)/2./radius**2)*\
                     contrast*np.exp(-(it/self.protocol['movie_refresh_freq']-t0)**2/2./sT**2)+bg_color)-1.
            FRAMES.append(frame)
            times[it] = if+1
            
        return times, FRAMES


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
        
    def get_frame(self, index):
        return self.noise_gen.get_frame(index).T
            

class dense_noise(visual_stim):

    def __init__(self, protocol):

        super().__init__(protocol)
        super().init_experiment(protocol,
            ['square-size', 'sparseness', 'mean-refresh-time', 'jitter-refresh-time'])

        self.noise_gen = dense_noise_generator(duration=protocol['presentation-duration'],
                                                screen=self.screen,
                                                square_size=protocol['square-size (deg)'],
                                                contrast=protocol['contrast (norm.)'],
                                                noise_mean_refresh_time=protocol['mean-refresh-time (s)'],
                                                noise_rdm_jitter_refresh_time=protocol['jitter-refresh-time (s)'],
                                                seed=protocol['noise-seed (#)'])

        self.experiment['time_start'] = self.noise_gen.events[:-1]
        self.experiment['time_stop'] = self.noise_gen.events[1:]
        
    def get_frame(self, index):
        return self.noise_gen.get_frame(index).T
            


if __name__=='__main__':

    import json, tempfile
    from pathlib import Path
    
    with open('exp/protocols/center-gratings.json', 'r') as fp:
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

    protocol['Setup']='demo-mode'
    stim = build_stim(protocol)
    parent = dummy_parent()
    stim.run(parent)
    stim.close()
