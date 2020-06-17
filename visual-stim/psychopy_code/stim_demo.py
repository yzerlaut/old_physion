from psychopy import visual, core, event, clock #import some libraries from PsychoPy
import numpy as np
import sys

SCREEN = [800,600]

if sys.argv[-1]=='light-level':

    mywin = visual.Window(SCREEN,monitor="testMonitor", units="deg") #create a window
    LEVELS = []
    for i, level in enumerate([0, 1, -1, 0]):
        LEVELS.append(visual.GratingStim(win=mywin, size=1000, pos=[0,0], sf=0, color=level))

    for i, duration in enumerate([1, 2, 2, 1]):
        LEVELS[i].draw()
        mywin.flip()
        clock.wait(duration)
    mywin.close()
    core.quit()
        
if sys.argv[-1]=='grating':

    mywin = visual.Window(SCREEN,monitor="testMonitor", units="deg") #create a window
    ORI = []
    for i, theta in enumerate(np.linspace(0, 5*180/6., 6)):
        ORI.append(visual.GratingStim(win=mywin, size=1000, sf=1, ori=theta))
    blank = visual.GratingStim(win=mywin, size=1000, pos=[0,0], sf=0, color=0)
    
    for i in range(len(ORI)):
        ORI[i].draw()
        mywin.flip()
        clock.wait(3)
        blank.draw()
        mywin.flip()
        clock.wait(1)
        
    mywin.close()
    core.quit()


if sys.argv[-1]=='center-grating':

    x, y, theta = np.meshgrid([-5, 0, 5], [-5, 0, 5], [0, 60, 120])
    X, Y, Theta = x.flatten(), y.flatten(), theta.flatten()
    
    mywin = visual.Window(SCREEN,monitor="testMonitor", units="deg") #create a window
    
    CG = []
    for i in np.random.choice(np.arange(len(X)), len(X), replace=False):
        CG.append(visual.GratingStim(win=mywin, mask='circle', size=3,
                                     pos=[X[i],Y[i]], sf=1, ori=Theta[i]))
    blank = visual.GratingStim(win=mywin, size=1000, pos=[0,0], sf=0, color=0)
    
    for i in range(len(CG)):
        CG[i].draw()
        mywin.flip()
        clock.wait(1)
        blank.draw()
        mywin.flip()
        clock.wait(0.5)
        
    mywin.close()
    core.quit()

if sys.argv[-1]=='contour-grating':

    x, y, theta = np.meshgrid([-5, 0, 5], [-5, 0, 5], [0, 60, 120])
    X, Y, Theta = x.flatten(), y.flatten(), theta.flatten()
    
    mywin = visual.Window(SCREEN,monitor="testMonitor", units="deg") #create a window
    
    CG, MASK = [], []
    for i in np.random.choice(np.arange(len(X)), len(X), replace=False):
        CG.append(visual.GratingStim(win=mywin, size=1000, sf=1, ori=Theta[i]))
        MASK.append(visual.GratingStim(win=mywin, mask='circle', size=4,
                                     pos=[X[i],Y[i]], sf=0, color=0))
    blank = visual.GratingStim(win=mywin, size=1000, pos=[0,0], sf=0, color=0)
    
    for i in range(len(CG))[:10]:
        CG[i].draw()
        MASK[i].draw()
        mywin.flip()
        clock.wait(1)
        blank.draw()
        mywin.flip()
        clock.wait(0.5)
        
    mywin.close()
    core.quit()

if sys.argv[-1]=='monitoring':
    
    mywin = visual.Window(SCREEN,monitor="testMonitor", units="deg") #create a window

    # monitoring signal
    on = visual.GratingStim(win=mywin, size=1, pos=[7,-7], sf=0, color=1)
    off = visual.GratingStim(win=mywin, size=1, pos=[7,-7], sf=0, color=-1)

    # on top of orientation protocol
    ORI = []
    for i, theta in enumerate(np.linspace(0, 5*180/6., 6)):
        ORI.append(visual.GratingStim(win=mywin, size=1000, sf=1, ori=theta))
    blank = visual.GratingStim(win=mywin, size=1000, pos=[0,0], sf=0, color=0)

    Ton, Toff = 200, 800 # ms
    Tfull, Tfull_first = int(Ton+Toff), int((Ton+Toff)/2.)
    for i in range(len(ORI)):
        
        #draw the stimuli and update the window
        start = clock.getTime()
        while (clock.getTime()-start)<5:
            ORI[i].draw()
            if (int(1e3*clock.getTime()-1e3*start)<Tfull) and\
               (int(1e3*clock.getTime()-1e3*start)%Tfull_first<Ton):
                on.draw()
            elif int(1e3*clock.getTime()-1e3*start)%Tfull<200:
                on.draw()
            else:
                off.draw()
            mywin.flip()
        blank.draw()
        off.draw()
        mywin.flip()
        clock.wait(2)
    
    #cleanup
    mywin.close()
    core.quit()
    
if sys.argv[-1]=='drifting-grating':
    
    mywin = visual.Window(SCREEN,monitor="testMonitor", units="deg") #create a window
    df = 2 # drifting grating frequency: cycle/s
    
    #create some stimuli
    grating = visual.GratingStim(win=mywin, mask='circle', size=5, pos=[-3,0], sf=1, ori=60)
    fixation = visual.GratingStim(win=mywin, size=0.2, pos=[0,0], sf=0, color=-1)

    #draw the stimuli and update the window
    start = clock.getTime()
    prev = start
    while (clock.getTime()-start)<5:
        grating.setPhase(df*(clock.getTime()-prev), '+') # advance phase
        prev = clock.getTime()
        grating.draw()
        mywin.flip()

    #cleanup
    mywin.close()
    core.quit()

if sys.argv[-1]=='dense-noise':
    
    mywin = visual.Window(SCREEN,monitor="testMonitor", units="deg") #create a window
    df = 2 # drifting grating frequency: cycle/s
    
    #create some stimuli
    grating = visual.GratingStim(win=mywin, mask='circle', size=5, pos=[-3,0], sf=1, ori=60)
    fixation = visual.GratingStim(win=mywin, size=0.2, pos=[0,0], sf=0, color=-1)

    #draw the stimuli and update the window
    start = clock.getTime()
    prev = start
    while (clock.getTime()-start)<5:
        grating.setPhase(df*(clock.getTime()-prev), '+') # advance phase
        prev = clock.getTime()
        grating.draw()
        mywin.flip()

    #cleanup
    mywin.close()
    core.quit()
    
else:
    print("""
    Use as:
              python stim_demo.py 'drifting-grating'
    or:
              python stim_demo.py 'sparse-noise'
    [...]
    """)
