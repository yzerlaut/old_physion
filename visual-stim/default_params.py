import numpy as np

PRESENTATIONS = ['Single-Stimulus', 'Stimuli-Sequence', 'Randomized-Sequence']

BLANK_SCREENS = ['Black Screen' ,'Grey Screen', 'White Screen']
NAME_TO_COLOR = {'Black Screen':-1, 'Grey Screen':0, 'White Screen':1}
SETUP = ['demo-mode', 'Bacci-2P-setup']

PRESENTATION = {
    # duration
    'presentation-duration':5.,
    'presentation-prestim-period':2,
    'presentation-interstim-period':2,
    'presentation-poststim-period':2,
    # blank screen
    'presentation-prestim-screen':-1,
    'presentation-interstim-screen':-1,
    'presentation-poststim-screen':-1,
    # repetition
    'N-repeat':2,
    'shuffling-seed':1,
    'starting-index':0,
}

STIMULI = {
    
    # simple screen at constant light level
    'light-level':{
        'light-level (lum.)':0.5,
        # range
        'light-level-1':-1., 'light-level-2':1., 'N-light-level':3},
    
    # full-field static grating
    'full-field-grating':{
        'angle (deg)':60,
        'spatial-freq (cycle/deg)':0.5,
        'contrast (norm.)':1.,
        # range
        'spatial-freq-1':0.5, 'spatial-freq-2':2., 'N-spatial-freq':2,
        'angle-1':0., 'angle-2':180, 'N-angle':0,
        'contrast-1':0.2, 'contrast-2':1., 'N-contrast':0},
    
    # center grating
    'center-grating':{
        'angle (deg)':60, 'spatial-freq (cycle/deg)':0.5,
        'radius (deg)':5, 'x-center (deg)':0, 'y-center (deg)':0,
        'contrast (norm.)':1.,
        'bg-color (lum.)':0., # not thought to be varied
        # range
        'angle-1':0, 'angle-2':150., 'N-angle':0,
        'spatial-freq-1':0.5, 'spatial-freq-2':2., 'N-spatial-freq':2,
        'radius-1':2, 'radius-2':8., 'N-radius':0,
        'x-center-1':-5., 'x-center-2':5., 'N-x-center':0,
        'y-center-1':-4., 'y-center-2':4., 'N-y-center':0,
        'contrast-1':0.2, 'contrast-2':1., 'N-contrast':0,
        'bg-color-1':-1., 'bg-color-2':1., 'N-bg-color':0},

    # off-center grating
    'off-center-grating':{
        'angle (deg)':60, 'spatial-freq (cycle/deg)':0.5,
        'radius (deg)':5, 'x-center (deg)':0, 'y-center (deg)':0,
        'contrast (norm.)':1.,
        'bg-color (lum.)':0., # not thought to be varied
        # range
        'angle-1':0, 'angle-2':150., 'N-angle':0,
        'spatial-freq-1':0.5, 'spatial-freq-2':2., 'N-spatial-freq':2,
        'radius-1':2, 'radius-2':8., 'N-radius':0,
        'x-center-1':-5., 'x-center-2':5., 'N-x-center':0,
        'y-center-1':-4., 'y-center-2':4., 'N-y-center':0,
        'contrast-1':0.2, 'contrast-2':1., 'N-contrast':0,
        'bg-color-1':-1., 'bg-color-2':1., 'N-bg-color':0},

    # surround grating
    'surround-grating':{
        'angle (deg)':60, 'spatial-freq (cycle/deg)':0.5,
        'radius-start (deg)':4, 'radius-end (deg)':8,
        'x-center (deg)':0, 'y-center (deg)':0,
        'contrast (norm.)':1.,
        'bg-color (lum.)':0., # not thought to be varied
        # range
        'angle-1':0, 'angle-2':150., 'N-angle':0,
        'spatial-freq-1':0.5, 'spatial-freq-2':2., 'N-spatial-freq':2,
        'radius-start-1':2., 'radius-start-2':8., 'N-radius-start':0,
        'radius-end-1':0., 'radius-end-2':20., 'N-radius-end':0,
        'x-center-1':-5., 'x-center-2':5., 'N-x-center':0,
        'y-center-1':-4., 'y-center-2':4., 'N-y-center':0,
        'contrast-1':0.2, 'contrast-2':1., 'N-contrast':0,
        'bg-color-1':-1., 'bg-color-2':1., 'N-bg-color':0},
    
    # full-field drifting grating
    'drifting-full-field-grating':{
        'angle (deg)':60,
        'spatial-freq (cycle/deg)':0.5,
        'speed (cycle/s)':1.,
        'contrast (norm.)':1.,
        # range
        'spatial-freq-1':0.5, 'spatial-freq-2':2., 'N-spatial-freq':2,
        'angle-1':0., 'angle-2':180, 'N-angle':0,
        'speed-1':0.1, 'speed-2':50, 'N-speed':5,
        'contrast-1':0.2, 'contrast-2':1., 'N-contrast':0},
    
    # center drifting grating
    'drifting-center-grating':{
        'angle (deg)':60, 'spatial-freq (cycle/deg)':0.5,
        'radius (deg)':5, 'x-center (deg)':0, 'y-center (deg)':0,
        'speed (cycle/s)':1.,
        'contrast (norm.)':1.,
        'bg-color (lum.)':0., # not thought to be varied
        # range
        'angle-1':0, 'angle-2':150., 'N-angle':0,
        'spatial-freq-1':0.5, 'spatial-freq-2':2., 'N-spatial-freq':2,
        'speed-1':0.1, 'speed-2':50, 'N-speed':5,
        'radius-1':2, 'radius-2':8., 'N-radius':0,
        'x-center-1':-5., 'x-center-2':5., 'N-x-center':0,
        'y-center-1':-4., 'y-center-2':4., 'N-y-center':0,
        'contrast-1':0.2, 'contrast-2':1., 'N-contrast':0,
        'bg-color-1':-1., 'bg-color-2':1., 'N-bg-color':0},
    
    # off-center grating
    'drifting-off-center-grating':{
        'angle (deg)':60, 'spatial-freq (cycle/deg)':0.5,
        'radius (deg)':5, 'x-center (deg)':0, 'y-center (deg)':0,
        'speed (cycle/s)':1.,
        'contrast (norm.)':1.,
        'bg-color (lum.)':0., # not thought to be varied
        # range
        'angle-1':0, 'angle-2':150., 'N-angle':0,
        'spatial-freq-1':0.5, 'spatial-freq-2':2., 'N-spatial-freq':2,
        'radius-1':2, 'radius-2':8., 'N-radius':0,
        'speed-1':0.1, 'speed-2':50, 'N-speed':5,
        'x-center-1':-5., 'x-center-2':5., 'N-x-center':0,
        'y-center-1':-4., 'y-center-2':4., 'N-y-center':0,
        'contrast-1':0.2, 'contrast-2':1., 'N-contrast':0,
        'bg-color-1':-1., 'bg-color-2':1., 'N-bg-color':0},

    # surround drifting grating
    'drifting-surround-grating':{
        'angle (deg)':60, 'spatial-freq (cycle/deg)':0.5,
        'radius-start (deg)':4, 'radius-end (deg)':8,
        'x-center (deg)':0, 'y-center (deg)':0,
        'speed (cycle/s)':1.,
        'contrast (norm.)':1.,
        'bg-color (lum.)':0., # not thought to be varied
        # range
        'angle-1':0, 'angle-2':150., 'N-angle':0,
        'spatial-freq-1':0.5, 'spatial-freq-2':2., 'N-spatial-freq':2,
        'radius-start-1':2., 'radius-start-2':8., 'N-radius-start':0,
        'radius-end-1':0., 'radius-end-2':20., 'N-radius-end':0,
        'speed-1':0.1, 'speed-2':50, 'N-speed':5,
        'x-center-1':-5., 'x-center-2':5., 'N-x-center':0,
        'y-center-1':-4., 'y-center-2':4., 'N-y-center':0,
        'contrast-1':0.2, 'contrast-2':1., 'N-contrast':0,
        'bg-color-1':-1., 'bg-color-2':1., 'N-bg-color':0},
    
    'center-surround-grating':{},
    'Natural-Image':{},
    'Natural-Image+VEM':{},
    'sparse-noise':{},
    'dense-noise':{},
    'full-field-grating+VEM':{},
}
