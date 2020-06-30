#####################################
## basic display and analysis of recordings ##
#####################################

import numpy as np
import os, sys, pathlib, tempfile
from datavyz import ges as ge # cistum visualization program

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.fetching import get_multimodal_dataset, get_list_of_datafiles

def last_datafile(data_folder):
    return get_list_of_datafiles(data_folder)[-1]

def quick_data_view(filename, dt=1e-3, subsampling=20):

    data = get_multimodal_dataset(filename)
    
    if 'NIdaq' in data:
        fig1, AX = ge.figure(axes = (1,data['NIdaq'].shape[0]), figsize=(3.,.9), left=.2, bottom=0.1, right=0.)
        for i in range(data['NIdaq'].shape[0]):
            x = data['NIdaq'][i,:]
            t = np.arange(len(x))*dt
            AX[i].plot(t[::subsampling], x[::subsampling])
            ge.set_plot(AX[i], xlabel='time (s)', ylabel='channel #%i' % i)
        cond = data['time_stop']<np.max(t)
        AX[0].plot(data['time_start'][cond], np.ones(np.sum(cond))*AX[0].get_ylim()[1], 'r*')
        AX[0].plot(data['time_stop'][cond], np.ones(np.sum(cond))*AX[0].get_ylim()[1], 'b*')

    ge.show()

def analyze_data(filename, dt=1e-3, subsampling=20):

    data = get_multimodal_dataset(filename)

    # print(data['images'][0])
    ge.image(data['images'][0])
    # fig, ax = ge.figure(axes_extents=(4,1))
    
    # if 'NIdaq' in data:
    #     x = data['NIdaq'][0,:]
    #     t = np.arange(len(x))*dt
    #     ax.plot(t[::subsampling], x[::subsampling])
    #     cond = data['time_start']<np.max(t)
    #     ax.plot(data['time_start'][cond], np.ones(np.sum(cond))*x.max(), 'r*')
    #     ax.plot(data['time_stop'][cond], np.ones(np.sum(cond))*x.max(), 'b*')
    ge.show()
    


if __name__=='__main__':

    quick_data_view(last_datafile(tempfile.gettempdir()))
    # analyze_data(last_datafile(tempfile.gettempdir()))
