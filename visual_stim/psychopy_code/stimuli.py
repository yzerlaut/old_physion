from psychopy import visual, core, event, clock, monitors # some libraries from PsychoPy
import numpy as np
import itertools, os, sys, pathlib, subprocess, time
 
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from screens import SCREENS
from psychopy_code.noise import sparse_noise_generator, build_dense_noise
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
        parent.statusBar.showMessage('stimulation stopped !')
        return True
    else:
        return False


class visual_stim:

    def __init__(self, protocol,
                 screen='Lilliput',
                 screen_id = 1,
                 screen_size = np.array([1280, 768]),
                 monitoring_square = {'size':8,
                                      'x':-24,
                                      'y':-13.5,
                                      'color-on':1,
                                      'color-off':-1,
                                      'time-on':0.2, 'time-off':0.8},
                 gamma_correction= {'k':1.03,
                                    'gamma':1.77}):
        """
        """
        self.protocol = protocol
        self.monitoring_square = monitoring_square
        self.screen = screen
        # gamma_correction
        self.k, self.gamma = gamma_correction['k'], gamma_correction['gamma']
        
        if self.protocol['Setup']=='demo-mode':
            self.monitor = monitors.Monitor('testMonitor')
            self.win = visual.Window(screen_size, monitor=self.monitor,
                                     units='deg', color=-1) #create a window
        else:
            self.monitor = monitors.Monitor(screen)
            self.win = visual.Window(screen_size, monitor=self.monitor,
                                     screen=screen_id, fullscr=True, units='deg', color=-1)
            
        # blank screens
        self.blank_start = visual.GratingStim(win=self.win, size=1000, pos=[0,0], sf=0,
                                color=self.protocol['presentation-prestim-screen'])
        if 'presentation-interstim-screen' in self.protocol:
            self.blank_inter = visual.GratingStim(win=self.win, size=1000, pos=[0,0], sf=0,
                                                  color=self.protocol['presentation-interstim-screen'])
        self.blank_end = visual.GratingStim(win=self.win, size=1000, pos=[0,0], sf=0,
                                color=self.protocol['presentation-poststim-screen'])

        # monitoring signal
        self.on = visual.GratingStim(win=self.win, size=self.monitoring_square['size'],
                                         pos=[self.monitoring_square['x'], self.monitoring_square['y']],
                                         sf=0, color=self.monitoring_square['color-on'])
        self.off = visual.GratingStim(win=self.win, size=self.monitoring_square['size'],
                                          pos=[self.monitoring_square['x'], self.monitoring_square['y']],
                                          sf=0, color=self.monitoring_square['color-off'])
        
        # initialize the times for the monitoring signals
        self.Ton = int(1e3*self.monitoring_square['time-on'])
        self.Toff = int(1e3*self.monitoring_square['time-off'])
        self.Tfull, self.Tfull_first = int(self.Ton+self.Toff), int((self.Ton+self.Toff)/2.)

    # Gamma correction 
    def gamma_corrected_lum(self, level):
        return 2*np.power(((level+1.)/2./self.k), 1./self.gamma)-1.
    def gamma_corrected_contrast(self, contrast):
        return np.power(contrast/self.k, 1./self.gamma)
    
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
                self.win.getMovieFrame() # we store the last frame
            except AttributeError:
                pass
            clock.wait(self.protocol['presentation-interstim-period'])

            # blinking in bottom-left corner
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
            
    #####################################################
    # showing a single static pattern
    def single_static_patterns_presentation(self, parent, index):
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

    def static_run(self, parent):
        self.start_screen(parent)
        for i in range(len(self.experiment['index'])):
            if stop_signal(parent):
                break
            print('Running protocol of index %i/%i' % (i+1, len(self.experiment['index'])))
            self.single_static_patterns_presentation(parent, i)
            self.win.getMovieFrame() # we store the last frame
            if self.protocol['Presentation']!='Single-Stimulus':
                self.inter_screen(parent)
        self.end_screen(parent)
        if not parent.stop_flag:
            parent.statusBar.showMessage('stimulation over !')
        self.win.saveMovieFrames(os.path.join(parent.datafolder,
                                              'screen-frames', 'frame.tiff'))

            
    #####################################################
    # showing a single dynamic pattern with a phase advance
    def single_dynamic_gratings_presentation(self, parent, index):
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

    def drifting_run(self, parent):
        self.start_screen(parent)
        for i in range(len(self.experiment['index'])):
            if stop_signal(parent):
                break
            print('Running protocol of index %i/%i' % (i+1, len(self.experiment['index'])))
            self.speed = self.experiment['speed'][i]
            self.single_dynamic_gratings_presentation(parent, i)
            if self.protocol['Presentation']!='Single-Stimulus':
                self.inter_screen(parent)
        self.end_screen(parent)
        if not parent.stop_flag:
            parent.statusBar.showMessage('stimulation over !')
        self.win.saveMovieFrames(os.path.join(parent.datafolder,
                                              'screen-frames', 'frame.tiff'))

    #####################################################
    # adding a Virtual-Scene-Exploration on top of an image stim
    def single_VSE_image_presentation(self, parent, index):
        start, prev_t = clock.getTime(), clock.getTime()
        while ((clock.getTime()-start)<self.protocol['presentation-duration']) and not parent.stop_flag:
            new_t = clock.getTime()
            i0 = np.min(np.argwhere(self.VSEs[index]['t']>(new_t-start)))
            self.PATTERNS[index][i0].draw()
            self.add_monitoring_signal(new_t, start)
            prev_t = new_t
            try:
                self.win.flip()
            except AttributeError:
                pass
        self.win.getMovieFrame() # we store the last frame

    def vse_run(self, parent):
        self.start_screen(parent)
        for i in range(len(self.experiment['index'])):
            if stop_signal(parent):
                break
            print('Running protocol of index %i/%i' % (i+1, len(self.experiment['index'])))
            self.single_VSE_image_presentation(parent, i)
            if self.protocol['Presentation']!='Single-Stimulus':
                self.inter_screen(parent)
        self.end_screen(parent)
        if not parent.stop_flag:
            parent.statusBar.showMessage('stimulation over !')
        self.win.saveMovieFrames(os.path.join(parent.datafolder,
                                              'screen-frames', 'frame.tiff'))
        
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
        # self.win.getMovieFrame() # we store the last frame

    def array_run(self, parent):
        self.start_screen(parent)
        for i in range(len(self.experiment['time_start'])):
            if stop_signal(parent):
                break
            print('Running frame of index %i/%i' % (i+1, len(self.experiment['time_start'])))
            self.single_array_presentation(parent, i)
            if self.protocol['Presentation']!='Single-Stimulus':
                self.inter_screen(parent)
        self.end_screen(parent)
        if not parent.stop_flag:
            parent.statusBar.showMessage('stimulation over !')
        # self.win.saveMovieFrames(os.path.join(parent.datafolder,
        #                                       'screen-frames', 'frame.tiff'))
        
    # #####################################################
    # # adding a run purely define by an array (time, x, y), see e.g. sparse_noise initialization
    # def array_run(self, parent):
    #     # start screen
    #     self.start_screen(parent)
    #     # stimulation
    #     start, prev_t = clock.getTime(), clock.getTime()
    #     while ((clock.getTime()-start)<self.protocol['presentation-duration']) and not parent.stop_flag:
    #         if stop_signal(parent):
    #             break
    #         new_t = clock.getTime()
    #         try:
    #             it = np.argwhere((self.STIM['t'][1:]>=(new_t-start)) & (self.STIM['t'][:-1]<(new_t-start))).flatten()[0]
    #             pattern = visual.ImageStim(self.win,
    #                                        image=self.gamma_corrected_lum(self.STIM['array'][it,:,:].T),
    #                                        units='pix', size=self.win.size)
    #             pattern.draw()
                
    #         except BaseException as e:
    #             print('time not matching')
    #             print(np.argwhere((self.STIM['t'][1:]>=(new_t-start)) & (self.STIM['t'][:-1]<(new_t-start))))
    #         self.add_monitoring_signal(new_t, start)
    #         prev_t = new_t
    #         try:
    #             self.win.flip()
    #         except AttributeError:
    #             pass
    #     self.end_screen(parent)
    #     if not parent.stop_flag:
    #         parent.statusBar.showMessage('stimulation over !')

            
    ## FINAL RUN FUNCTION
    def run(self, parent):
        if len(self.protocol['Stimulus'].split('drifting'))>1:
            return self.drifting_run(parent)
        elif len(self.protocol['Stimulus'].split('VSE'))>1:
            return self.vse_run(parent)
        elif len(self.protocol['Stimulus'].split('noise'))>1:
            return self.array_run(parent)
        else:
            return self.static_run(parent)
        
#####################################################
##  ----   PRESENTING VARIOUS LIGHT LEVELS  --- #####           
#####################################################

class light_level_single_stim(visual_stim):

    def __init__(self, protocol):
        
        super().__init__(protocol)
        super().init_experiment(protocol, ['light-level'])
        
        # then manually building patterns
        for i in range(len(self.experiment['index'])):
            self.PATTERNS.append([\
            visual.GratingStim(win=self.win,
                               size=1000, pos=[0,0], sf=0,
                               color=self.gamma_corrected_lum(self.experiment['light-level'][i]))])
            
    def get_patterns(self, index):
        return [visual.GratingStim(win=self.win,
                                   size=1000, pos=[0,0], sf=0,
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
                                   size=1000, pos=[0,0],
                                   sf=self.experiment['spatial-freq'][index],
                                   ori=self.experiment['angle'][index],
                                   contrast=self.gamma_corrected_contrast(self.experiment['contrast'][index]))]
                                 
            
class drifting_full_field_grating_stim(visual_stim):

    def __init__(self, protocol):
        super().__init__(protocol)
        super().init_experiment(protocol, ['spatial-freq', 'angle', 'contrast', 'speed'])

    def get_patterns(self, index):
        return [visual.GratingStim(win=self.win,
                                   size=1000, pos=[0,0],
                                   sf=self.experiment['spatial-freq'][index],
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
                                   pos=[self.experiment['x-center'][index], self.experiment['y-center'][index]],
                                   size=self.experiment['radius'][index], mask='circle',
                                   sf=self.experiment['spatial-freq'][index],
                                   ori=self.experiment['angle'][index],
                                   contrast=self.gamma_corrected_contrast(self.experiment['contrast'][index]))]

class drifting_center_grating_stim(visual_stim):
    
    def __init__(self, protocol):
        super().__init__(protocol)
        super().init_experiment(protocol, ['x-center', 'y-center', 'radius','spatial-freq', 'angle', 'contrast', 'speed', 'bg-color'])

    def get_patterns(self, index):
        return [visual.GratingStim(win=self.win,
                                   pos=[self.experiment['x-center'][index], self.experiment['y-center'][index]],
                                   size=self.experiment['radius'][index], mask='circle',
                                   sf=self.experiment['spatial-freq'][index],
                                   ori=self.experiment['angle'][index],
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
                                   size=1000, pos=[0,0],
                                   sf=self.experiment['spatial-freq'][index],
                                   ori=self.experiment['angle'][index],
                                   contrast=self.gamma_corrected_contrast(self.experiment['contrast'][index])),
                visual.GratingStim(win=self.win,
                                   pos=[self.experiment['x-center'][index], self.experiment['y-center'][index]],
                                   size=self.experiment['radius'][index],
                                   mask='circle', sf=0,
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
                                                screen=SCREENS[self.screen],
                                                square_size=protocol['square-size (deg)'],
                                                noise_mean_refresh_time=protocol['mean-refresh-time (s)'],
                                                noise_rdm_jitter_refresh_time=protocol['jitter-refresh-time (s)'],
                                                seed=protocol['noise-seed (#)'])

        print(self.Ton, self.Toff)
        self.experiment['time_start'] = self.noise_gen.events[:-1]
        self.experiment['time_stop'] = self.noise_gen.events[:-1]+self.noise_gen.durations
        
    def get_frame(self, index):
        return self.noise_gen.get_frame(index)
            

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
            


if __name__=='__main__':

    import json
    with open('protocol.json', 'r') as fp:
        protocol = json.load(fp)
    
    stim = light_level_single_stim(protocol)
    stim.run()
    stim.close()
