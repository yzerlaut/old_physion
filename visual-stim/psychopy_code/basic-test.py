from psychopy import visual, core, event #import some libraries from PsychoPy
import numpy as np

SCREEN = [800,600]
#create a window
mywin = visual.Window(SCREEN,monitor="testMonitor", units="deg")

x, y, t = np.meshgrid(np.arange(SCREEN[0]), np.arange(SCREEN[1]), np.linspace(0,2*np.pi))

z = (np.sin(3*y)+1.)/2.
z = np.sin(.1*y+.1*t)

#create some stimuli
# grating = visual.ImageStim(win=mywin, image=z, size=10)
grating = visual.MovieStim(win=mywin, image=z, size=10)
# grating = visual.GratingStim(win=mywin, mask='circle', size=3, pos=[-4,0], sf=3)
fixation = visual.GratingStim(win=mywin, size=0.2, pos=[0,0], sf=0, rgb=-1)

#draw the stimuli and update the window
while True: #this creates a never-ending loop
    # grating.setPhase(0.01, '+')#advance phase by 0.05 of a cycle
    grating.draw()
    fixation.draw()
    mywin.flip()

    if len(event.getKeys())>0:
        break
    event.clearEvents()

#cleanup
mywin.close()
core.quit()
