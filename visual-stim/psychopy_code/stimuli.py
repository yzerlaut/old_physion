from psychopy import visual, core, event, clock #import some libraries from PsychoPy
import numpy as np

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
    # if (protocol['Presentation']=='Stimuli-Sequence'):
    #     if (protocol['Stimulus']=='light-level'):
    else:
        print('Protocol not recognized !')
        return None
    

class visual_stim:

    def __init__(self, protocol):
        """
        """
        self.protocol = protocol
        self.win = visual.Window(SCREEN, monitor=MONITOR, units=UNITS) #create a window
        # blank screens
        self.blank_start = visual.GratingStim(win=self.win, size=1000, pos=[0,0], sf=0,
                                color=self.protocol['presentation-prestim-screen'])
        if 'presentation-middlestim-screen' in self.protocol:
            self.blank_middle = visual.GratingStim(win=self.win, size=1000, pos=[0,0], sf=0,
                                color=self.protocol['presentation-middlestim-screen'])
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
    def start_screen(self):
        self.blank_start.draw()
        self.off.draw()
        self.win.flip()
        clock.wait(self.protocol['presentation-prestim-period'])

    # screen at end
    def end_screen(self):
        self.blank_end.draw()
        self.off.draw()
        self.win.flip()
        clock.wait(self.protocol['presentation-poststim-period'])

    # showing a single static pattern
    def single_static_patterns_presentation(self, PATTERNS, duration):
        start = clock.getTime()
        while (clock.getTime()-start)<duration:
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
    def single_dynamic_gratings_presentation(self, PATTERNS, duration):
        start, prev_t = clock.getTime(), clock.getTime()
        while (prev_t-start)<duration:
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
        
    
class light_level_single_stim(visual_stim):

    def __init__(self, protocol):
        super().__init__(protocol)
        self.grating = visual.GratingStim(win=self.win,
                                          size=1000, pos=[0,0], sf=0,
                                          color=protocol['light-level (lum.)'])

    def run(self):
        self.start_screen()
        self.single_static_patterns_presentation([self.grating], self.protocol['presentation-duration'])
        self.end_screen()

        
class full_field_grating_stim(visual_stim):

    def __init__(self, protocol):
        super().__init__(protocol)
        self.grating = visual.GratingStim(win=self.win,
                                          size=1000, pos=[0,0],
                                          sf=protocol['spatial-freq (cycle/deg)'],
                                          ori=protocol['angle (deg)'],
                                          contrast=protocol['contrast (norm.)'])

    def run(self):
        self.start_screen()
        self.single_static_patterns_presentation([self.grating], self.protocol['presentation-duration'])
        self.end_screen()
        

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

    def run(self):
        self.start_screen()
        self.single_static_patterns_presentation([self.bg, self.grating],
                                                 self.protocol['presentation-duration'])
        self.end_screen()


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

    def run(self):
        self.start_screen()
        self.single_static_patterns_presentation([self.grating, self.center],
                                                 self.protocol['presentation-duration'])
        self.end_screen()

        
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

    def run(self):
        self.start_screen()
        self.single_static_patterns_presentation([self.bg, self.grating, self.mask],
                                                 self.protocol['presentation-duration'])
        self.end_screen()
        

class full_field_drifting_grating_stim(visual_stim):

    def __init__(self, protocol):
        super().__init__(protocol)
        self.grating = visual.GratingStim(win=self.win,
                                              size=1000, pos=[0,0],
                                              sf=protocol['spatial-freq (cycle/deg)'],
                                              ori=protocol['angle (deg)'],
                                              contrast=protocol['contrast (norm.)'])
        self.speed = protocol['speed (cycle/s)']
            
    def run(self):
        self.start_screen()
        self.single_dynamic_gratings_presentation([self.grating],
                                                  self.protocol['presentation-duration'])
        self.end_screen()


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
            
    def run(self):
        self.start_screen()
        self.single_dynamic_gratings_presentation([self.bg, self.grating],
                                                  self.protocol['presentation-duration'])
        self.end_screen()

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

    def run(self):
        self.start_screen()
        self.single_dynamic_gratings_presentation([self.grating, self.center],
                                                  self.protocol['presentation-duration'])
        self.end_screen()


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

    def run(self):
        self.start_screen()
        self.single_dynamic_gratings_presentation([self.bg, self.grating, self.mask],
                                                  self.protocol['presentation-duration'])
        self.end_screen()
        

if __name__=='__main__':

    import json
    with open('protocol.json', 'r') as fp:
        protocol = json.load(fp)
    
    stim = light_level_single_stim(protocol)
    stim.run()
    stim.close()
