# general modules
import pynwb, os, sys, pathlib, itertools
import numpy as np
import matplotlib.pylab as plt
from PIL import Image

# custom modules
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from dataviz.show_data import *
from datavyz import graph_env
ge = graph_env('screen')


def draw_figure(top_row_bottom=0.75, top_row_space=0.08, top_row_height=0.2):

    AX = {'time_plot_ax':None}
    fig, AX['time_plot_ax'] = ge.figure(figsize=(2,3),
            bottom=0.05, right=0.5)

    width = (1.-4*top_row_space)/4.
    for i, label in enumerate(['setup', 'screen', 'camera', 'imaging']):
        AX['%s_ax'%label] = ge.inset(fig, (top_row_space/2.+i*(width+top_row_space), top_row_bottom, width, top_row_height))
    return fig, AX



if __name__=='__main__':

    import argparse

    parser=argparse.ArgumentParser()
    parser.add_argument("datafile", type=str)
    parser.add_argument('-o', "--ops", default='raw', help='')
    parser.add_argument("--tlim", type=float, nargs='*', default=[10, 100], help='')
    parser.add_argument('-e', "--episode", type=int, default=0)
    parser.add_argument('-nmax', "--Nmax", type=int, default=4)
    parser.add_argument("--Npanels", type=int, default=8)
    parser.add_argument('-roi', "--roiIndex", type=int, default=0)
    parser.add_argument('-pid', "--protocol_id", type=int, default=0)
    parser.add_argument('-q', "--quantity", type=str, default='dFoF')
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")

    args = parser.parse_args()

    data = MultimodalData(args.datafile)

    fig, AX = draw_figure()    

    # setup drawing
    img = Image.open('doc/exp-rig.png')
    AX['setup_ax'].imshow(img)
    ge.set_plot(AX['setup_ax'], [])

    for i, label in enumerate(['screen', 'camera', 'imaging']):
        ge.set_plot(AX['%s_ax'%label], [], title=label)

    
    data.plot_raw_data(args.tlim, 
                       settings={'CaImagingRaster':dict(fig_fraction=4, subsampling=1,
                                               roiIndices='all',
                                               normalization='per-line',
                                               subquantity='dF/F'),
                        'CaImaging':dict(fig_fraction=3, subsampling=1, 
                                         subquantity='dF/F', color='#2ca02c',
                                         roiIndices=np.sort(np.random.choice(np.arange(np.sum(data.iscell)), np.min([args.Nmax, data.iscell.sum()]), replace=False))),
                        'Locomotion':dict(fig_fraction=1, subsampling=1, color='#1f77b4'),
                        'Pupil':dict(fig_fraction=2, subsampling=1, color='#d62728'),
                        'GazeMovement':dict(fig_fraction=1, subsampling=1, color='#ff7f0e'),
                        'Photodiode':dict(fig_fraction=.5, subsampling=1, color='grey'),
                        'VisualStim':dict(fig_fraction=.005, color='black')},
                        Tbar=1, ax=AX['time_plot_ax'])
        
    ge.show()




