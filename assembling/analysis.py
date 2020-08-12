#####################################
## basic display and analysis of recordings ##
#####################################

import numpy as np
import os, sys, pathlib
from datavyz import ges as ge # cistum visualization program

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.fetching import get_multimodal_dataset, get_list_of_datafiles, last_datafile, transform_into_realigned_episodes

def quick_data_view(filename, dt=1e-3, subsampling=20 ,Nimage=10, realign=False):

    data = get_multimodal_dataset(filename, image_sampling_number=Nimage)
    if realign:
        transform_into_realigned_episodes(data)
    
    Npanel = 0

    t = [0, data['time_stop'][-1]]
    if 'FaceCamera-times' in data:
        Npanel +=1
        t = np.array([0, data['FaceCamera-times'][-1]])
    if 'NIdaq' in data:
        Npanel += data['NIdaq'].shape[0]
        t = np.arange(data['NIdaq'].shape[1])*dt # overrides the obove
        
    fig, AX = ge.figure(axes = (1,Npanel), figsize=(3.,.9), left=.3, bottom=0.1, right=0.1)
    i0=0
    if 'FaceCamera-times' in data:
        Npanel +=1
        AX[i0].plot(data['FaceCamera-times'], np.zeros(len(data['FaceCamera-times'])), '|')
        for i, im in enumerate(data['FaceCamera-imgs']):
            ax = ge.inset(AX[i0], [i/Nimage, 0.2, 1./Nimage, 0.6])
            ge.image(im, ax=ax)
        ge.set_plot(AX[i0], ['bottom'], ylabel='camera', ylim=[0,1], xlim=[t[0], t[-1]], xlabel='time (s)')
        i0+=1
    if 'NIdaq' in data:
        for i in range(data['NIdaq'].shape[0]):
            x = data['NIdaq'][i,:]
            t = np.arange(len(x))*dt
            AX[i0].plot(t[::subsampling], x[::subsampling])
            ge.set_plot(AX[i0], ylabel='chan. #%i (V)' % i, xlim=[t[0], t[-1]])
            cond = data['time_start']<np.max(t)
            AX[i0].plot(data['time_start'][cond], np.ones(np.sum(cond))*.9*AX[i0].get_ylim()[1], 'r*', ms=3)
            if realign:
                AX[i0].plot(data['time_start_realigned'], np.ones(len(data['time_start_realigned']))*.9*AX[i0].get_ylim()[1], 'r|', ms=10)
            i0+=1
            
    return data, fig

    
def analyze_data(filename='', data=None, dt=1e-3, subsampling=10):

    if data is None:
        data = get_multimodal_dataset(filename)
        transform_into_realigned_episodes(data)

    fig, AX = ge.figure(axes_extents=[[[1,1], [4,1]] for i in range(len(np.unique(data['angle'])))])

    for a, angle in enumerate(np.unique(data['angle'])):
        Is =np.argwhere(data['angle']==angle).flatten()
        try:
            ge.image(data['images'][Is[0]], ax=AX[a][0])
        except IndexError:
            pass
        for k, i in enumerate(Is):
            for c in range(data['NIdaq'].shape[0]):
                try:
                    norm_signal = (data['NIdaq_realigned'][i][c,:]-np.min(data['NIdaq_realigned'][i][c,:]))/\
                        (np.max(data['NIdaq_realigned'][i][c,:])-np.min(data['NIdaq_realigned'][i][c,:]))
                    AX[a][1].plot(data['t_realigned'][::subsampling], (c+norm_signal[::subsampling])/data['NIdaq'].shape[0], color=ge.colors[c])
                except IndexError:
                    pass
        ge.set_plot(AX[a][1], ['bottom'], xlim=[data['t_realigned'][0], data['t_realigned'][-1]])

    # t = np.arange(data['NIdaq'].shape[1])*dt
    # norm_NIdaq = (data['NIdaq'].T-data['NIdaq'].min(axis=1).T)/(data['NIdaq'].max(axis=1).T-data['NIdaq'].min(axis=1).T)

    # for i in range(len(data['images'])):
    #    cond = (t>=data['time_start'][i]) & (t<data['time_start'][i]+5)#data['time_stop'][i])
    #    for j in range(data['NIdaq'].shape[0]):
    #        AX[i][1].plot(t[cond]-t[cond][0], norm_NIdaq[cond, j], color=ge.colors[j])

    return data, fig

    


if __name__=='__main__':

    import tempfile
    quick_data_view(last_datafile(tempfile.gettempdir()), realign=True)
    # data, fig = analyze_data(last_datafile(tempfile.gettempdir()))
    ge.show()
