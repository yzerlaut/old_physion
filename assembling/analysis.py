#####################################
## basic display and analysis of recordings ##
#####################################

import numpy as np
import os, sys, pathlib
from datavyz import ges as ge # cistum visualization program

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.fetching import get_multimodal_dataset, get_list_of_datafiles

def last_datafile(parent):
    return get_list_of_datafiles(parent)[-1]

def quick_data_view(filename, dt=1e-3):

    data = get_multimodal_dataset(filename)

    fig, ax = ge.figure(axes_extents=(4,1))
    
    # for i in ['start', 'end']:
    #     if os.path.isfile(filename.replace('visual-stim.npz', 'NIdaq.%s.npy' % i)):
    #         x = data['NIdaq-%s' % i][1,:]
    #         ax.plot(np.arange(len(x))[::10]*dt+data['time_start'][i], x[::10])

    i=0
    for i in data['index']:
        if ('NIdaq-%s'%i) in  data:
            x = data['NIdaq-%s' % i][1,:]
            ax.plot(np.arange(len(x))[::10]*dt+data['time_start'][i], x[::10])

    ge.show()
    
