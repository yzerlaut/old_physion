from psychopy import visual, core, event, clock #import some libraries from PsychoPy
import numpy as np
import itertools

SCREEN = [800,600]
MONITOR = "testMonitor"
UNITS = "deg"
MONITORING_SQUARE = {'size':2, 'x':11, 'y':-8, 'color-on':1, 'color-off':-1,
                     'time-on':0.2, 'time-off':0.8}

def build_stim(protocol):
    """
    """
    if (protocol['Presentation']=='Single-Stimulus'):
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
            return full_field_drifting_grating_stim(protocol)
        elif (protocol['Stimulus']=='drifting-center-grating'):
            return center_drifting_grating_stim(protocol)
        elif (protocol['Stimulus']=='drifting-off-center-grating'):
            return off_center_drifting_grating_stim(protocol)
        elif (protocol['Stimulus']=='drifting-surround-grating'):
            return surround_drifting_grating_stim(protocol)
        else:
            print('Protocol not recognized !')
            return None
    else:
        if (protocol['Stimulus']=='light-level'):
            stim = None # SET_of_light_level_single_stim(protocol)
        elif (protocol['Stimulus']=='full-field-grating'):
            stim = SET_of_full_field_grating_stim(protocol)
        elif (protocol['Stimulus']=='center-grating'):
            stim = None # SET_of_center_grating_stim(protocol)
        elif (protocol['Stimulus']=='off-center-grating'):
            stim = None # SET_of_off_center_grating_stim(protocol)
        elif (protocol['Stimulus']=='surround-grating'):
            stim = None # SET_of_surround_grating_stim(protocol)
        elif (protocol['Stimulus']=='drifting-full-field-grating'):
            stim = SET_of_full_field_drifting_grating_stim(protocol)
        # elif (protocol['Stimulus']=='drifting-center-grating'):
        #     return SET_of_center_drifting_grating_stim(protocol)
        # elif (protocol['Stimulus']=='drifting-off-center-grating'):
        #     return SET_of_off_center_drifting_grating_stim(protocol)
        # elif (protocol['Stimulus']=='drifting-surround-grating'):
        #     return SET_of_surround_drifting_grating_stim(protocol)
        else:
            print('Protocol not recognized !')
            stim = None

        # SHUFFLING IF NECESSARY
        if (protocol['Presentation']=='Randomized-Sequence'):
            np.random.seed(protocol['shuffling-seed'])
            np.random.shuffle(stim.experiment['index'])
            
        return stim
            
        

def stop_signal(parent):
    if (len(event.getKeys())>0) or (parent.stop_flag):
        parent.stop_flag = True
        parent.statusBar.showMessage('stimulation stopped !')
        return True
    else:
        return False


class visual_stim:

    def __init__(self, protocol):
        """
        """
        self.protocol = protocol
        self.win = visual.Window(SCREEN, monitor=MONITOR, units=UNITS) #create a window
        # blank screens
        self.blank_start = visual.GratingStim(win=self.win, size=1000, pos=[0,0], sf=0,
                                color=self.protocol['presentation-prestim-screen'])
        if 'presentation-interstim-screen' in self.protocol:
            self.blank_inter = visual.GratingStim(win=self.win, size=1000, pos=[0,0], sf=0,
                                                  color=self.protocol['presentation-interstim-screen'])
        self.blank_end = visual.GratingStim(win=self.win, size=1000, pos=[0,0], sf=0,
                                color=self.protocol['presentation-poststim-screen'])

        # monitoring signal
        self.on = visual.GratingStim(win=self.win, size=MONITORING_SQUARE['size'],
                                         pos=[MONITORING_SQUARE['x'], MONITORING_SQUARE['y']],
                                         sf=0, color=MONITORING_SQUARE['color-on'])
        self.off = visual.GratingStim(win=self.win, size=MONITORING_SQUARE['size'],
                                          pos=[MONITORING_SQUARE['x'], MONITORING_SQUARE['y']],
                                          sf=0, color=MONITORING_SQUARE['color-off'])
        
        # initialize the times for the monitoring signals
        self.Ton = int(1e3*MONITORING_SQUARE['time-on'])
        self.Toff = int(1e3*MONITORING_SQUARE['time-off'])
        self.Tfull, self.Tfull_first = int(self.Ton+self.Toff), int((self.Ton+self.Toff)/2.)

        
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
            self.win.flip()
            clock.wait(self.protocol['presentation-prestim-period'])

    # screen at end
    def end_screen(self, parent):
        if not parent.stop_flag:
            self.blank_end.draw()
            self.off.draw()
            self.win.flip()
            clock.wait(self.protocol['presentation-poststim-period'])

    # screen for interstim
    def inter_screen(self, parent):
        if not parent.stop_flag:
            self.blank_inter.draw()
            self.off.draw()
            self.win.flip()
            clock.wait(self.protocol['presentation-interstim-period'])
            
    # showing a single static pattern
    def single_static_patterns_presentation(self, parent, PATTERNS, duration):
        start = clock.getTime()
        while ((clock.getTime()-start)<duration) and not parent.stop_flag:
            for pattern in PATTERNS:
                pattern.draw()
            if (int(1e3*clock.getTime()-1e3*start)<self.Tfull) and\
               (int(1e3*clock.getTime()-1e3*start)%self.Tfull_first<self.Ton):
                self.on.draw()
            elif int(1e3*clock.getTime()-1e3*start)%self.Tfull<self.Ton:
                self.on.draw()
            else:
                self.off.draw()
            self.win.flip()

    # showing a single dynamic pattern with a phase advance
    def single_dynamic_gratings_presentation(self, parent, PATTERNS, duration):
        start, prev_t = clock.getTime(), clock.getTime()
        while ((prev_t-start)<duration) and not parent.stop_flag:
            new_t = clock.getTime()
            for pattern in PATTERNS:
                pattern.setPhase(self.speed*(new_t-prev_t), '+') # advance phase
                pattern.draw()
            self.add_monitoring_signal(new_t, start)
            prev_t = new_t
            self.win.flip()
        
    def add_monitoring_signal(self, new_t, start):
        if (int(1e3*new_t-1e3*start)<self.Tfull) and\
           (int(1e3*new_t-1e3*start)%self.Tfull_first<self.Ton):
            self.on.draw()
        elif int(1e3*new_t-1e3*start)%self.Tfull<self.Ton:
            self.on.draw()
        else:
            self.off.draw()
        
#####################################################
##  ----   PRESENTING VARIOUS LIGHT LEVELS  --- #####           
#####################################################

class light_level_single_stim(visual_stim):

    def __init__(self, protocol):
        super().__init__(protocol)
        self.grating = visual.GratingStim(win=self.win,
                                          size=1000, pos=[0,0], sf=0,
                                          color=protocol['light-level (lum.)'])

    def run(self, parent):
        self.start_screen(parent)
        self.single_static_patterns_presentation(parent,
                                                 [self.grating], self.protocol['presentation-duration'])
        self.end_screen(parent)
        parent.statusBar.showMessage('stimulation over !')

        
#####################################################
##  ----   PRESENTING FULL FIELD GRATINGS   --- #####           
#####################################################

class full_field_grating_stim(visual_stim):

    def __init__(self, protocol):
        super().__init__(protocol)
        self.grating = visual.GratingStim(win=self.win,
                                          size=1000, pos=[0,0],
                                          sf=protocol['spatial-freq (cycle/deg)'],
                                          ori=protocol['angle (deg)'],
                                          contrast=protocol['contrast (norm.)'])

    def run(self, parent):
        self.start_screen(parent)
        self.single_static_patterns_presentation(parent,
                                                 [self.grating], self.protocol['presentation-duration'])
        self.end_screen(parent)
        parent.statusBar.showMessage('stimulation over !')
        

class SET_of_full_field_grating_stim(visual_stim):

    def __init__(self, protocol):
        super().__init__(protocol)
        # spatial freq
        if protocol['N-spatial-freq']>1:
            self.SF = np.linspace(protocol['spatial-freq-1'], protocol['spatial-freq-2'],
                                  protocol['N-spatial-freq'])
        else:
            self.SF = np.array([protocol['spatial-freq-1']])
        # angle
        if protocol['N-angle']>1:
            self.Angle = np.linspace(protocol['angle-1'], protocol['angle-2'],
                                  protocol['N-angle'])
        else:
            self.Angle = np.array([protocol['angle-1']])
        # contrast
        if protocol['N-contrast']>1:
            self.Contrast = np.linspace(protocol['contrast-1'], protocol['contrast-2'],
                                  protocol['N-contrast'])
        else:
            self.Contrast = np.array([protocol['contrast-1']])

        self.PATTERNS = []
        self.experiment ={'spatial-freq':[], 'angle':[], 'contrast':[]}
        for sf, a, c in itertools.product(self.SF, self.Angle, self.Contrast):
            self.PATTERNS.append([visual.GratingStim(win=self.win,
                                                    size=1000, pos=[0,0],
                                                     sf=sf, ori=a, contrast=c)])
            self.experiment['spatial-freq'].append(sf)
            self.experiment['angle'].append(a)
            self.experiment['contrast'].append(c)

        self.experiment['index'] = np.arange(len(self.PATTERNS))
        
        
    def run(self, parent):
        self.start_screen(parent)
        for i in self.experiment['index']:
            if stop_signal(parent):
                break
            self.single_static_patterns_presentation(parent,
                                                         self.PATTERNS[i],
                                                         self.protocol['presentation-duration'])
            self.inter_screen(parent)
        self.end_screen(parent)
        parent.statusBar.showMessage('stimulation over !')
        
        
#####################################################
##  ----    PRESENTING CENTERED GRATINGS    --- #####           
#####################################################

class center_grating_stim(visual_stim):

    def __init__(self, protocol):
        super().__init__(protocol)
        self.grating = visual.GratingStim(win=self.win,
                                          size=protocol['radius (deg)'], mask='circle',
                                          pos=[protocol['x-center (deg)'], protocol['y-center (deg)']],
                                          sf=protocol['spatial-freq (cycle/deg)'],
                                          ori=protocol['angle (deg)'],
                                          contrast=protocol['contrast (norm.)'])
        self.bg = visual.GratingStim(win=self.win,
                                     size=1000, pos=[0,0], sf=0,
                                     color=protocol['bg-color (lum.)'])

    def run(self, parent):
        self.start_screen(parent)
        self.single_static_patterns_presentation([self.bg, self.grating],
                                                 self.protocol['presentation-duration'])
        self.end_screen(parent)
        parent.statusBar.showMessage('stimulation over !')


class off_center_grating_stim(visual_stim):

    def __init__(self, protocol):
        super().__init__(protocol)
        self.grating = visual.GratingStim(win=self.win,
                                          size=1000, pos=[0,0],
                                          sf=protocol['spatial-freq (cycle/deg)'],
                                          ori=protocol['angle (deg)'],
                                          contrast=protocol['contrast (norm.)'])
        self.center = visual.GratingStim(win=self.win,
                                         size=protocol['radius (deg)'], mask='circle', sf=0,
                                         pos=[protocol['x-center (deg)'], protocol['y-center (deg)']],
                                         color=protocol['bg-color (lum.)'])

    def run(self, parent):
        self.start_screen(parent)
        self.single_static_patterns_presentation([self.grating, self.center],
                                                 self.protocol['presentation-duration'])
        self.end_screen(parent)
        parent.statusBar.showMessage('stimulation over !')

        
class surround_grating_stim(visual_stim):

    def __init__(self, protocol):
        super().__init__(protocol)
        self.grating = visual.GratingStim(win=self.win,
                                          size=protocol['radius-end (deg)'], mask='circle',
                                          pos=[protocol['x-center (deg)'], protocol['y-center (deg)']],
                                          sf=protocol['spatial-freq (cycle/deg)'],
                                          ori=protocol['angle (deg)'],
                                          contrast=protocol['contrast (norm.)'])
        self.bg = visual.GratingStim(win=self.win,
                                     size=1000, pos=[0,0], sf=0,
                                     color=protocol['bg-color (lum.)'])
        self.mask = visual.GratingStim(win=self.win,
                                       size=protocol['radius-start (deg)'], mask='circle',
                                       pos=[protocol['x-center (deg)'], protocol['y-center (deg)']],
                                       color=protocol['bg-color (lum.)'])

    def run(self, parent):
        self.start_screen(parent)
        self.single_static_patterns_presentation([self.bg, self.grating, self.mask],
                                                 self.protocol['presentation-duration'])
        self.end_screen(parent)
        parent.statusBar.showMessage('stimulation over !')
        

##############################################################
##  ----   PRESENTING FULL FIELD DRIFTING GRATINGS   --- #####           
##############################################################


class full_field_drifting_grating_stim(visual_stim):

    def __init__(self, protocol):
        super().__init__(protocol)
        self.grating = visual.GratingStim(win=self.win,
                                          size=1000, pos=[0,0],
                                          sf=protocol['spatial-freq (cycle/deg)'],
                                          ori=protocol['angle (deg)'],
                                          contrast=protocol['contrast (norm.)'])
        self.speed = protocol['speed (cycle/s)']
            
    def run(self, parent):
        self.start_screen(parent)
        self.single_dynamic_gratings_presentation([self.grating],
                                                  self.protocol['presentation-duration'])
        self.end_screen(parent)
        parent.statusBar.showMessage('stimulation over !')


class SET_of_full_field_drifting_grating_stim(visual_stim):

    def __init__(self, protocol):
        super().__init__(protocol)
        # spatial freq
        if protocol['N-spatial-freq']>1:
            self.SF = np.linspace(protocol['spatial-freq-1'], protocol['spatial-freq-2'],
                                  protocol['N-spatial-freq'])
        else:
            self.SF = np.array([protocol['spatial-freq-1']])
        # angle
        if protocol['N-angle']>1:
            self.Angle = np.linspace(protocol['angle-1'], protocol['angle-2'],
                                  protocol['N-angle'])
        else:
            self.Angle = np.array([protocol['angle-1']])
        # contrast
        if protocol['N-contrast']>1:
            self.Contrast = np.linspace(protocol['contrast-1'], protocol['contrast-2'],
                                  protocol['N-contrast'])
        else:
            self.Contrast = np.array([protocol['contrast-1']])
        # speed
        if protocol['N-speed']>1:
            self.speeds = np.linspace(protocol['speed-1'], protocol['speed-2'],
                                      protocol['N-speed'])
        else:
            self.speeds = np.array([protocol['speed-1']])
            
        self.PATTERNS = []
        self.experiment ={'spatial-freq':[], 'angle':[], 'contrast':[], 'speeds':[]}
        
        for sf, a, c, sp in itertools.product(self.SF, self.Angle, self.Contrast, self.speeds):
            self.PATTERNS.append([visual.GratingStim(win=self.win,
                                                     size=1000, pos=[0,0],
                                                     sf=sf, ori=a, contrast=c)])
            self.experiment['spatial-freq'].append(sf)
            self.experiment['angle'].append(a)
            self.experiment['contrast'].append(c)
            self.experiment['speeds'].append(sp)
            
        self.experiment['index'] = np.arange(len(self.PATTERNS))

    def run(self, parent):
        self.start_screen(parent)
        for i in self.experiment['index']:
            if stop_signal(parent):
                break
            self.speed = self.experiment['speeds'][i]
            if not parent.stop_flag:
                self.single_dynamic_gratings_presentation(parent,
                                                          self.PATTERNS[i],
                                                          self.protocol['presentation-duration'])
                self.inter_screen(parent)
        self.end_screen(parent)
        parent.statusBar.showMessage('stimulation over !')
        

        
class center_drifting_grating_stim(visual_stim):

    def __init__(self, protocol):
        super().__init__(protocol)
        self.grating = visual.GratingStim(win=self.win,
                                          size=protocol['radius (deg)'], mask='circle',
                                          pos=[protocol['x-center (deg)'], protocol['y-center (deg)']],
                                          sf=protocol['spatial-freq (cycle/deg)'],
                                          ori=protocol['angle (deg)'],
                                          contrast=protocol['contrast (norm.)'])
        self.bg = visual.GratingStim(win=self.win,
                                     size=1000, pos=[0,0], sf=0,
                                     color=protocol['bg-color (lum.)'])
        self.speed = protocol['speed (cycle/s)']
            
    def run(self, parent):
        self.start_screen(parent)
        self.single_dynamic_gratings_presentation([self.bg, self.grating],
                                                  self.protocol['presentation-duration'])
        self.end_screen(parent)

class off_center_drifting_grating_stim(visual_stim):
    
    def __init__(self, protocol):
        super().__init__(protocol)
        self.grating = visual.GratingStim(win=self.win,
                                          size=1000, pos=[0,0],
                                          sf=protocol['spatial-freq (cycle/deg)'],
                                          ori=protocol['angle (deg)'],
                                          contrast=protocol['contrast (norm.)'])
        self.center = visual.GratingStim(win=self.win,
                                         size=protocol['radius (deg)'], mask='circle', sf=0,
                                         pos=[protocol['x-center (deg)'], protocol['y-center (deg)']],
                                         color=protocol['bg-color (lum.)'])
        self.speed = protocol['speed (cycle/s)']

    def run(self, parent):
        self.start_screen(parent)
        self.single_dynamic_gratings_presentation([self.grating, self.center],
                                                  self.protocol['presentation-duration'])
        self.end_screen(parent)
        parent.statusBar.showMessage('stimulation over !')


class surround_drifting_grating_stim(visual_stim):
    def __init__(self, protocol):
        super().__init__(protocol)
        self.grating = visual.GratingStim(win=self.win,
                                          size=protocol['radius-end (deg)'], mask='circle',
                                          pos=[protocol['x-center (deg)'], protocol['y-center (deg)']],
                                          sf=protocol['spatial-freq (cycle/deg)'],
                                          ori=protocol['angle (deg)'],
                                          contrast=protocol['contrast (norm.)'])
        self.bg = visual.GratingStim(win=self.win,
                                     size=1000, pos=[0,0], sf=0,
                                     color=protocol['bg-color (lum.)'])
        self.mask = visual.GratingStim(win=self.win,
                                       size=protocol['radius-start (deg)'], mask='circle',
                                       pos=[protocol['x-center (deg)'], protocol['y-center (deg)']],
                                       color=protocol['bg-color (lum.)'])
        self.speed = protocol['speed (cycle/s)']

    def run(self, parent):
        self.start_screen(parent)
        self.single_dynamic_gratings_presentation([self.bg, self.grating, self.mask],
                                                  self.protocol['presentation-duration'])
        self.end_screen(parent)
        parent.statusBar.showMessage('stimulation over !')
        

if __name__=='__main__':

    import json
    with open('protocol.json', 'r') as fp:
        protocol = json.load(fp)
    
    stim = light_level_single_stim(protocol)
    stim.run()
    stim.close()
