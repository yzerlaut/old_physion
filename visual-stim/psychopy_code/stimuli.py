from psychopy import visual, core, event #import some libraries from PsychoPy
import numpy as np

SCREEN = [800,600]


class visual_stim:

    def __init__(self, protocol='Single-Stimulus', stimulus='light-level'):
        """

        """
        self.stimulus, self.protocol = stimulus, protocol
        #create a window
        self.window = visual.Window(SCREEN,monitor="testMonitor", units="deg")
        
    def build_protocol(self, **args):

        if self.stimulus=='light-level':
            self.pattern = visual.ImageStim(self.window, np.ones(SCREEN), size=100)
        elif self.stimulus=='full-field-grating':
            pass
    
    # def center_grating(self):
    #     #create some stimuli

    # def full_field_grating(self):
    #     #create some stimuli
    #     self.pattern = visual.GratingStim(win=self.window, mask='circle', size=100, pos=[-4,0], sf=3)
        
    # def drifting_grating()
    # grating = visual.GratingStim(win=mywin, mask='circle', size=3, pos=[-4,0], sf=3)
    # fixation = visual.GratingStim(win=mywin, size=0.2, pos=[0,0], sf=0, rgb=-1)

    def show(self):
        
        #draw the stimuli and update the window
        while True: #this creates a never-ending loop
            # if self.protocol!='Single-Stimulus':
            #     self.pattern.setPhase(0.01, '+')#advance phase by 0.05 of a cycle
            self.pattern.draw()
            # fixation.draw()
            self.window.flip()

            if len(event.getKeys())>0:
                break
            event.clearEvents()

        #cleanup
        self.window.close()
        core.quit()
