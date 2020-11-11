import sys, time, tempfile, os, pathlib, json, datetime, string
import numpy as np
# from PyQt5 import QtGui, QtWidgets, QtCore
import pyqtgraph as pg
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import day_folder
from assembling.dataset import Dataset, MODALITIES
from analysis import guiparts, plots

import numpy as np

def print_summary(dataset):
    S = ''
    for key in dataset.VisualStim:
        if key not in ['time_start', 'time_stop', 'index']:
            # print('then this is a parameter of the stimulus')
            if len(np.unique(dataset.VisualStim[key]))>1:
                S+='N-%s = %i (from %s to %s)' % (key, len(np.unique(dataset.VisualStim[key])),
                                                      np.min(dataset.VisualStim[key]),np.max(dataset.VisualStim[key]))
            else:
                S+='%s = %.2f' % (key, dataset.VisualStim[key][0])
            S+='\n'
    if 'time_start_realigned' in dataset.metadata:
        S += 'completed N=%i/%i episodes' %(\
                       len(dataset.metadata['time_start_realigned']), len(dataset.metadata['time_start']))
    else:
        print('"time_start_realigned" not available')
        print('--> Need to realign data with respect to Visual-Stimulation !!')
    return S

def build_episodes(dataset):

    EPISODES = {}
    if dataset.Locomotion is not None:
        EPISODES['Locomotion'] = []
    
    for tstart, tstop in zip(dataset.metadata['time_start_realigned'], dataset.metadata['time_stop_realigned']):

        if dataset.Locomotion is not None:
            cond = (dataset.Locomotion.t>=tstart) & (dataset.Locomotion.t<tstop)
            EPISODES['Locomotion'].append(dataset.Locomotion.val[cond])
        if dataset.Electrophy is not None:
            cond = (dataset.Electrophy.t>=tstart) & (dataset.Electrophy.t<tstop)
        if dataset.CaImaging is not None:
            cond = (dataset.CaImaging.t>=tstart) & (dataset.CaImaging.t<tstop)
            # EPISODES['CaImaging_Firing'] = 
            # dataset.CaImaging.Firing
            # print(tstart, tstop)
            print(cond)

if __name__=='__main__':

    folder = os.path.join(os.path.expanduser('~'),\
                          'DATA', '2020_11_04', '15-09-39')

    # stim = np.load(os.path.join(folder, 'visual-stim.npy'), allow_pickle=True).item()
    dataset = Dataset(folder)
    build_episodes(dataset)
