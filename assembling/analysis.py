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

    fig, AX = ge.figure(axes_extents=[[[1,1], [4,1]] for i in range(len(data['images']))])

    t = np.arange(data['NIdaq'].shape[1])*dt
    norm_NIdaq = (data['NIdaq'].T-data['NIdaq'].min(axis=1).T)/(data['NIdaq'].max(axis=1).T-data['NIdaq'].min(axis=1).T)
    print(norm_NIdaq.shape)
    for i in range(len(data['images'])):
       ge.image(data['images'][i], ax=AX[i][0])
       cond = (t>=data['time_start'][i]) & (t<data['time_start'][i]+5)#data['time_stop'][i])
       for j in range(data['NIdaq'].shape[0]):
           AX[i][1].plot(t[cond]-t[cond][0], norm_NIdaq[cond, j], color=ge.colors[j])
    ge.show()
    


if __name__=='__main__':

    # quick_data_view(last_datafile(tempfile.gettempdir()))
    analyze_data(last_datafile(tempfile.gettempdir()))
