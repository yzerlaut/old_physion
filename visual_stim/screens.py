import numpy as np

SCREENS = {
    'Lilliput':{
        'name':'Lilliput',
        'screen_id':1,
        'resolution':[1280, 768],
        'width':16, # in cm
        'distance_from_eye':15, # in cm
        'monitoring_square':{'size':100,
                             'location':'bottom-left',
                             'on_times':np.concatenate([[0],[0.5],np.arange(1, 100)]), # single stimuli won't last too long
                             'on_duration':0.2},
        'gamma_correction':{'k':1.03,
                            'gamma':1.77},
    },
    'Dell-P2018H':{
        'name':'Dell-P2018H',
        'screen_id':1,
        'resolution':[1280, 720],
        'width':43.4, # in cm
        'height':23.6, # in cm
        'distance_from_eye':15., # in cm
        'monitoring_square':{'size':60,
                             'location':'bottom-left',
                             'on_times':np.concatenate([[0],[0.5],np.arange(1, 100)]), # single stimuli won't last too long
                             'on_duration':0.2},
        'gamma_correction':{'k':1.03,
                            'gamma':1.77},
    },
}
