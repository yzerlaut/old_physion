import numpy as np

STIMULI = {
    
    # simple screen at constant light level
    'light-level':{
        'light-level (lum.)':0.5,
        'light-level-1':-1., 'light-level-2':1., 'N-light-level':3},
    
    # full-field static grating
    'full-field-grating':{
        'angle (Rd)':90,
        'angle-1':0., 'angle-2':2*np.pi*5./6., 'N-angle':6},
    
    # center grating
    'center-grating':{
        'angle (Rd)':90, 'radius (Rd)':5, 'x-center (Rd)':0, 'y-center (Rd)':0,
        'angle-1':-1., 'angle-2':1., 'N-angle':6,
        'radius-1':-1., 'radius-2':1., 'N-radius':6,
        'x-center-1':-1., 'x-center-2':1., 'N-x-center':6,
        'y-center-1':-1., 'y-center-2':1., 'N-y-center':6},

    # 
    'surround-grating':{},
    'inverse-RF-grating':{},
    'drifting-FF-grating':{},
    'center-surround-grating':{},
    'Natural-Image':{},
    'natural-image+VEM':{},
    'full-field-grating+VEM':{},
}
