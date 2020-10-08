from psychopy import visual, core, event, clock, monitors # some libraries from PsychoPy
import numpy as np
import itertools, os, sys, pathlib, subprocess, time
 
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from psychopy_code.noise import build_sparse_noise, build_dense_noise

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
    elif (protocol['Stimulus']=='Natural-Image+VEM'):
        return natural_image_vem(protocol)
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
                 screen_size = np.array([1280, 768]),
                 monitoring_square = {'size':6,
                                      'x':-19.5,
                                      'y':-11.5,
                                      'color-on':1,
                                      'color-off':-1,
                                      'time-on':0.2, 'time-off':0.8},
                 # gamma_correction= {'k':1.03,
                 #                    'gamma':1.52}):
                 gamma_correction= {'k':1.03,
                                    'gamma':1.77}):
        """
        """
        self.protocol = protocol
        self.monitoring_square = monitoring_square
        # gamma_correction
        self.k, self.gamma = gamma_correction['k'], gamma_correction['gamma']
        
        if self.protocol['Setup']=='demo-mode':
            self.monitor = monitors.Monitor('testMonitor')
            self.win = visual.Window(screen_size, monitor=self.monitor,
                                     units='deg', color=-1) #create a window
        else:
            self.monitor = monitors.Monitor('Lilliput')
            self.win = visual.Window(screen_size, monitor=self.monitor,
                                     screen=1, fullscr=True, units='deg', color=-1)
            
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
                                                     (n+1)*protocol['presentation-duration']+(n+1)*protocol['presentation-interstim-period'])

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
        if (int(1e3*new_t-1e3*start)<self.Tfull) and\
           (int(1e3*new_t-1e3*start)%self.Tfull_first<self.Ton):
            self.on.draw()
        elif int(1e3*new_t-1e3*start)%self.Tfull<self.Ton:
            self.on.draw()
        else:
            self.off.draw()
            
    #####################################################
    # showing a single static pattern
    def single_static_patterns_presentation(self, parent, index):
        start = clock.getTime()
        while ((clock.getTime()-start)<self.protocol['presentation-duration']) and not parent.stop_flag:
            for pattern in self.PATTERNS[index]:
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
        self.win.saveMovieFrames(os.path.join(parent.datafolder, 'screen-frames', 'frame.tiff'))

            
    #####################################################
    # showing a single dynamic pattern with a phase advance
    def single_dynamic_gratings_presentation(self, parent, index):
        start, prev_t = clock.getTime(), clock.getTime()
        while ((clock.getTime()-start)<self.protocol['presentation-duration']) and not parent.stop_flag:
            new_t = clock.getTime()
            for pattern in self.PATTERNS[index]:
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
        self.win.saveMovieFrames(os.path.join(parent.datafolder, 'screen-frames', 'frame.tiff'))
        
    #####################################################
    # adding a run purely define by an array (time, x, y), see e.g. sparse_noise initialization
    def array_run(self, parent):
        # start screen
        self.start_screen(parent)
        # stimulation
        start, prev_t = clock.getTime(), clock.getTime()
        while ((clock.getTime()-start)<self.protocol['presentation-duration']) and not parent.stop_flag:
            if stop_signal(parent):
                break
            new_t = clock.getTime()
            try:
                it = np.argwhere((self.STIM['t'][1:]>=(new_t-start)) & (self.STIM['t'][:-1]<(new_t-start))).flatten()[0]
                self.PATTERNS[it].draw()
            except BaseException as e:
                print('time not matching')
                print(np.argwhere((self.STIM['t'][1:]>=(new_t-start)) & (self.STIM['t'][:-1]<(new_t-start))))
            self.add_monitoring_signal(new_t, start)
            prev_t = new_t
            try:
                self.win.flip()
            except AttributeError:
                pass
        self.end_screen(parent)
        if not parent.stop_flag:
            parent.statusBar.showMessage('stimulation over !')

            
    ## FINAL RUN FUNCTION
    def run(self, parent):
        if len(self.protocol['Stimulus'].split('drifting'))>1:
            return self.drifting_run(parent)
        elif len(self.protocol['Stimulus'].split('VEM'))>1:
            return self.vem_run(parent)
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
            
#####################################################
##  ----   PRESENTING FULL FIELD GRATINGS   --- #####           
#####################################################

class full_field_grating_stim(visual_stim):

    def __init__(self, protocol):
        super().__init__(protocol)
        super().init_experiment(protocol, ['spatial-freq', 'angle', 'contrast'])

        # then manually building patterns
        for i in range(len(self.experiment['index'])):
            self.PATTERNS.append([\
                                  visual.GratingStim(win=self.win,
                                                     size=1000, pos=[0,0],
                                                     sf=self.experiment['spatial-freq'][i],
                                                     ori=self.experiment['angle'][i],
                                                     contrast=self.gamma_corrected_contrast(self.experiment['contrast'][i]))])

            
class drifting_full_field_grating_stim(visual_stim):

    def __init__(self, protocol):
        super().__init__(protocol)
        super().init_experiment(protocol, ['spatial-freq', 'angle', 'contrast', 'speed'])

        # then manually building patterns
        for i in range(len(self.experiment['index'])):
            self.PATTERNS.append([\
                                  visual.GratingStim(win=self.win,
                                                     size=1000, pos=[0,0],
                                                     sf=self.experiment['spatial-freq'][i],
                                                     ori=self.experiment['angle'][i],
                                                     contrast=self.gamma_corrected_contrast(self.experiment['contrast'][i]))])

        
#####################################################
##  ----    PRESENTING CENTERED GRATINGS    --- #####           
#####################################################

class center_grating_stim(visual_stim):
    
    def __init__(self, protocol):
        super().__init__(protocol)
        super().init_experiment(protocol, ['x-center', 'y-center', 'radius','spatial-freq', 'angle', 'contrast'])

        # then manually building patterns
        for i in range(len(self.experiment['index'])):
            self.PATTERNS.append([\
                                  visual.GratingStim(win=self.win,
                                                     pos=[self.experiment['x-center'][i], self.experiment['y-center'][i]],
                                                     size=self.experiment['radius'][i], mask='circle',
                                                     sf=self.experiment['spatial-freq'][i],
                                                     ori=self.experiment['angle'][i],
                                                     contrast=self.gamma_corrected_contrast(self.experiment['contrast'][i]))])

class drifting_center_grating_stim(visual_stim):
    
    def __init__(self, protocol):
        super().__init__(protocol)
        super().init_experiment(protocol, ['x-center', 'y-center', 'radius','spatial-freq', 'angle', 'contrast', 'speed'])

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
        from .preprocess_NI import load, img_after_hist_normalization
        
        super().__init__(protocol)
        super().init_experiment(protocol, ['Image-ID'])

        
        for i in range(len(self.experiment['index'])):
            filename = os.listdir(NI_directory)[int(self.experiment['Image-ID'][i])]
            img = load(os.path.join(NI_directory, filename))
            img = 2*self.gamma_corrected_contrast(img_after_hist_normalization(img))-1 # normalization + gamma_correction
            # rescaled_img = adapt_to_screen_resolution(img, (SCREEN[0], SCREEN[1]))

            self.PATTERNS.append([visual.ImageStim(self.win, image=img.T,
                                                   units='pix', size=self.win.size)])


def generate_VEM(duration=5,
                 saccade_period=0.5, saccade_amplitude=50.,
                 microsaccade_period=0.05, microsaccade_amplitude=10.,
                 seed=0):

    np.random.seed(seed)
    
    t, x, y = [], [], []

    print('generating Virtual-Eye-Movement [...]')
    
    for tt in np.cumsum(np.random.exponential(saccade_period, size=int(duration/saccade_period))):
        
        t.append(tt)
        x.append(saccade_amplitude*np.random.randn())
        y.append(saccade_amplitude*np.random.randn())

    return {'t':np.array([0]+t),
            'x':np.array([0]+x),
            'y':np.array([0]+y)}
            

            
class natural_image_vem(visual_stim):

    def __init__(self, protocol):

        # from visual_stim.psychopy_code.preprocess_NI import load, img_after_hist_normalization
        from .preprocess_NI import load, img_after_hist_normalization
        
        super().__init__(protocol)
        super().init_experiment(protocol, ['Image-ID', 'VEM-seed'])

        self.VEMs = []
        for i in range(len(self.experiment['index'])):

            self.VEMs.append(generate_VEM(seed=int(self.experiment['VEM-seed'][i])))
            
            filename = os.listdir(NI_directory)[int(self.experiment['Image-ID'][i])]
            img = load(os.path.join(NI_directory, filename))
            img = 2*self.gamma_corrected_contrast(img_after_hist_normalization(img))-1 # normalization + gamma-correction
            # rescaled_img = adapt_to_screen_resolution(img, (SCREEN[0], SCREEN[1]))

            self.PATTERNS.append([visual.ImageStim(self.win, image=img.T,
                                                   units='pix', size=self.win.size)])
    

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
        
        for i in range(len(self.STIM['t'])-1):
            self.PATTERNS.append(visual.ImageStim(self.win,
                                                  image=self.gamma_corrected_lum(self.STIM['array'][i,:,:].T),
                                                  units='pix', size=self.win.size))

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

        for i in range(len(self.STIM['t'])-1):
            self.PATTERNS.append(visual.ImageStim(self.win,
                                                  image=self.gamma_corrected_lum(self.STIM['array'][i,:,:].T),
                                                  units='pix', size=self.win.size))
            
        self.experiment = {'refresh-times':self.STIM['t']}
            


if __name__=='__main__':

    import json
    with open('protocol.json', 'r') as fp:
        protocol = json.load(fp)
    
    stim = light_level_single_stim(protocol)
    stim.run()
    stim.close()
