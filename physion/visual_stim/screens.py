import numpy as np

SCREENS = {
    'Lilliput':{
        'name':'Lilliput',
        'screen_id':1,
        'resolution':[1280, 768],
        'width':16, # in cm
        'distance_from_eye':15, # in cm
        'fullscreen':True,
        'monitoring_square':{'size':8,
                             'color-on':1,
                             'color-off':-1,
                             'time-on':0.2,
                             'time-off':0.8,
                             'location':'bottom-left',
                             'x':-24,
                             'y':-13.5},
        'gamma_correction':{'k':1.03,
                            'gamma':1.77},
    },
    'Dell-P2018H':{
        'name':'Dell-P2018H',
        'screen_id':1,
        'resolution':[1280, 720],
        'width':43.4, # in cm
        'height':23.6, # in cm
        'distance_from_eye':15.0, # in cm
        'fullscreen':True,
        'monitoring_square':{'size':40.,
                             'location':'top-right',
                             'color-on':1,
                             'color-off':-1,
                             'time-on':0.2,
                             'time-off':0.8},
        'gamma_correction':{'k':1.03,
                            'gamma':1.77},
    },
    'Dell-2020':{
        'name':'Dell-2020',
        'screen_id':1,
        'resolution':[1280, 720],
        'width':48.0, # in cm
        'height':27.0, # in cm
        'distance_from_eye':15.0, # in cm
        'fullscreen':True,
        'monitoring_square':{'size':60.,
                             'location':'top-right',
                             'color-on':1,
                             'color-off':-1,
                             'time-on':0.2,
                             'time-off':0.8},
        'gamma_correction':{'k':1.03,
                            'gamma':1.77},
    },
}
