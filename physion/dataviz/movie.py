# general modules
import pynwb, os, sys, pathlib, itertools
import numpy as np
import matplotlib.pylab as plt
import matplotlib.animation as animation
from PIL import Image

# custom modules
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from assembling.saving import get_files_with_extension, list_dayfolder, check_datafolder, get_TSeries_folders
from assembling.move_CaImaging_folders import StartTime_to_day_seconds
from assembling.tools import load_FaceCamera_data

from dataviz.show_data import *
from datavyz import graph_env
ge = graph_env('screen')


def draw_figure(args, data,
                top_row_bottom=0.75,
                top_row_space=0.08,
                top_row_height=0.2,
                Ndiscret=100):

    fractions = {'photodiode':0.09, 'photodiode_start':0,
            'running':0.13, 'running_start':0.1,
            'whisking':0.15, 'whisking_start':0.25,
            'gaze':0.1, 'gaze_start':0.35,
            'pupil':0.13, 'pupil_start':0.45,
            'rois':0.14, 'rois_start':0.6,
            'raster':0.25, 'raster_start':0.75}
    times = np.linspace(args.tlim[0], args.tlim[1], Ndiscret)

    AX = {'time_plot_ax':None}
    fig, AX['time_plot_ax'] = ge.figure(figsize=(2,3.5),
            bottom=0.02, right=0.5)

    width = (1.-4*top_row_space)/4.
    for i, label in enumerate(['setup', 'screen', 'camera', 'imaging']):
        AX['%s_ax'%label] = ge.inset(fig, (top_row_space/2.+i*(width+top_row_space), top_row_bottom, width, top_row_height))


    AX['whisking_ax'] = ge.inset(fig, [0.04,0.15,0.11,0.11]) 
    AX['pupil_ax'] = ge.inset(fig, [0.04,0.28,0.11,0.13]) 
    AX['ROI_ax'] = ge.inset(fig, [0.04,0.45,0.11,0.13]) 
    AX['FOV_ax'] = ge.inset(fig, [0.04,0.6,0.11,0.13]) 

    t0 = times[0]

    # setup drawing
    img = Image.open('doc/exp-rig.png')
    AX['setup_ax'].imshow(img)
    ge.set_plot(AX['setup_ax'], [])
    # time = AX['setup_ax'].annotate('t=%.1fs' % times[0], (1,0), xycoords='axes fraction', ha='right')

    # screen inset
    AX['screen_img'] = data.visual_stim.show_frame(0, ax=AX['screen_ax'],
                                                   return_img=True,
                                                   label=None)
    
    # time cursor
    cursor, = AX['time_plot_ax'].plot(np.ones(2)*times[0], np.arange(2), 'k-')#color=ge.grey, lw=3, alpha=.3) 

    #   ----  filling time plot

    # photodiode and visual stim
    data.add_VisualStim(args.tlim, AX['time_plot_ax'], 
                        fig_fraction=2.,
                        with_screen_inset=False,
                        name='')
    data.add_Photodiode(args.tlim, AX['time_plot_ax'], 
                        fig_fraction_start=fractions['photodiode_start'], 
                        fig_fraction=fractions['photodiode'], 
                        name='')
    ge.annotate(AX['time_plot_ax'], 'photodiode', (-0.1, fractions['photodiode_start']), ha='center', va='bottom', color=ge.grey, size='small')


    # locomotion
    data.add_Locomotion(args.tlim, AX['time_plot_ax'], 
                        fig_fraction_start=fractions['running_start'], 
                        fig_fraction=fractions['running'], 
                        name='')
    ge.annotate(AX['time_plot_ax'], 'running-speed', (-0.1, fractions['running_start']), ha='center', va='bottom', color=ge.blue, size='small')

    # whisking 
    data.add_FaceMotion(args.tlim, AX['time_plot_ax'], 
                        fig_fraction_start=fractions['whisking_start'], 
                        fig_fraction=fractions['whisking'], 
                        name='')
    ge.annotate(AX['time_plot_ax'], 'whisking  ', (-0.01, fractions['whisking_start']), ha='right', va='bottom', color=ge.purple, size='small')

    # gaze 
    data.add_GazeMovement(args.tlim, AX['time_plot_ax'], 
                        fig_fraction_start=fractions['gaze_start'], 
                        fig_fraction=fractions['gaze'], 
                        name='')
    ge.annotate(AX['time_plot_ax'], 'gaze \nmov. ', (-0.01, fractions['gaze_start']), ha='right', va='bottom', color=ge.orange, size='small')

    # pupil 
    data.add_Pupil(args.tlim, AX['time_plot_ax'], 
                        fig_fraction_start=fractions['pupil_start'], 
                        fig_fraction=fractions['pupil'], 
                        name='')
    ge.annotate(AX['time_plot_ax'], 'pupil \ndiam. ', (-0.01, fractions['pupil_start']), ha='right', va='bottom', color=ge.red, size='small')

    # rois 
    data.add_CaImaging(args.tlim, AX['time_plot_ax'], 
                       roiIndices=args.ROIs, 
                       fig_fraction_start=fractions['rois_start'], 
                       fig_fraction=fractions['rois'], 
                       name='', with_annotation=False)
    # ge.annotate(AX['time_plot_ax'], 'pupil\ndiam.', (-0.1, fractions['pupil_start']), ha='center', va='bottom', color=ge.red, size='small')

    # raster 
    data.add_CaImagingRaster(args.tlim, AX['time_plot_ax'], 
                             subquantity='dFoF', normalization='per-line',
                             fig_fraction_start=fractions['raster_start'], 
                             fig_fraction=fractions['raster'], 
                             name='')
    # ge.annotate(AX['time_plot_ax'], 'pupil\ndiam.', (-0.1, fractions['pupil_start']), ha='center', va='bottom', color=ge.red, size='small')
    


    ge.set_plot(AX['time_plot_ax'], [], xlim=[times[0], times[-1]], ylim=[-0.01, 1.01])

    for i, label in enumerate(['screen', 'camera', 'imaging']):
        ge.set_plot(AX['%s_ax'%label], [], title=label)

    def update(i=0):
        iEp = data.find_episode_from_time(times[i])
        if iEp==-1:
            AX['screen_img'].set_array(data.visual_stim.x*0+0.5)
        else:
            tEp = data.nwbfile.stimulus['time_start_realigned'].data[iEp]
            data.visual_stim.update_frame(iEp, AX['screen_img'],
                                          time_from_episode_start=times[i]-tEp)
        cursor.set_data(np.ones(2)*times[i], np.arange(2))
        return [cursor, AX['screen_img']]
        
    ani = animation.FuncAnimation(fig, 
                                  update,
                                  np.arange(len(times)),
                                  init_func=update,
                                  interval=100,
                                  blit=True)

    return fig, AX, ani


def load_faceCamera(metadata):
    imgfolder = os.path.join(metadata['filename'], 'FaceCamera-imgs')
    times, FILES, nframes, Lx, Ly = load_FaceCamera_data(imgfolder, t0=metadata['NIdaq_Tstart'], verbose=True)
    
    print(times)

def load_NIdaq(metadata):
    metadata['NIdaq_Tstart'] = np.load(os.path.join(metadata['filename'], 'NIdaq.start.npy'))[0]


if __name__=='__main__':

    import argparse

    parser=argparse.ArgumentParser()
    parser.add_argument("datafile", type=str)
    parser.add_argument('-o', "--ops", default='raw', help='')
    parser.add_argument("--tlim", type=float, nargs='*', default=[10, 100], help='')
    parser.add_argument('-e', "--episode", type=int, default=0)
    parser.add_argument('-nmax', "--Nmax", type=int, default=4)
    parser.add_argument("--Npanels", type=int, default=8)
    parser.add_argument('-rois', "--ROIs", type=int, default=[0,1], nargs='*')
    parser.add_argument('-pid', "--protocol_id", type=int, default=0)
    parser.add_argument('-q', "--quantity", type=str, default='dFoF')
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")

    args = parser.parse_args()

    data = MultimodalData(args.datafile, with_visual_stim=True)

    metadata = dict(data.metadata)

    if os.path.isdir(metadata['filename']):
        load_NIdaq(metadata)
        load_faceCamera(metadata)

    fig, AX, ani = draw_figure(args, data)    

    # data.plot_raw_data(args.tlim, 
                       # settings={'CaImagingRaster':dict(fig_fraction=4, subsampling=1,
                                               # roiIndices='all',
                                               # normalization='per-line',
                                               # subquantity='dF/F'),
                        # 'CaImaging':dict(fig_fraction=3, subsampling=1, 
                                         # subquantity='dF/F', color='#2ca02c',
                                         # roiIndices=np.sort(np.random.choice(np.arange(np.sum(data.iscell)), np.min([args.Nmax, data.iscell.sum()]), replace=False))),
                        # 'Locomotion':dict(fig_fraction=1, subsampling=1, color='#1f77b4'),
                        # 'Pupil':dict(fig_fraction=2, subsampling=1, color='#d62728'),
                        # 'GazeMovement':dict(fig_fraction=1, subsampling=1, color='#ff7f0e'),
                        # 'Photodiode':dict(fig_fraction=.5, subsampling=1, color='grey'),
                        # 'VisualStim':dict(fig_fraction=.005, color='black')},
                        # Tbar=1, ax=AX['time_plot_ax'])
        
    ge.show()




