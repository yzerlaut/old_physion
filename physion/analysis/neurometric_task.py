# general modules
import pynwb, os, sys
import numpy as np
import matplotlib.pylab as plt
from scipy import stats
plt.style.use('ggplot')

# custom modules
sys.path.append('.')
from physion.dataviz import plots
from physion.analysis.read_NWB import read as read_NWB, Data
from physion.analysis.trial_averaging import build_episodes
from physion.visual_stim.psychopy_code.stimuli import build_stim
from physion.analysis.orientation_direction_selectivity import *

# we define a data object fitting this analysis purpose

ORIENTATION_PROTOCOLS = ['spatial-selectivity-static']
DIRECTION_PROTOCOLS = ['spatial-selectivity-drifting']


def spatial_selectivity_plot(angles, responses, ax=None, figsize=(2.5,1.5), color='k'):
    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)
    ax.plot(angles, responses/1e3, color=color, lw=2) # CHECK UNITS
    SI = spatial_selectivity_index(angles, responses)
    ax.set_xticks(angles)
    ax.set_ylabel('resp. integral ($\Delta$F/F.s)', fontsize=9)
    ax.set_xlabel('angle ($^o$)', fontsize=9)
    return SI

def spatial_selectivity_analysis(FullData, roiIndex=0, with_std=True, verbose=True, subprotocol_id=0):
    iprotocol = [i for (i,p) in enumerate(FullData.protocols) if (p in ORIENTATION_PROTOCOLS)][subprotocol_id]
    data = CellResponse(FullData, protocol_id=iprotocol,
                        quantity='CaImaging', subquantity='dF/F',
                        roiIndex = roiIndex, verbose=verbose)
    fig, AX = plt.subplots(1, len(data.varied_parameters['angle']), figsize=(11.4,2.5))
    plt.subplots_adjust(left=0.03, right=.75, top=0.9, bottom=0.05)
    for i, angle in enumerate(data.varied_parameters['angle']):
        data.plot('angle',i, ax=AX[i], with_std=with_std)
        AX[i].set_title('%.0f$^o$' % angle, fontsize=9)
    YLIM = (np.min([ax.get_ylim()[0] for ax in AX]), np.max([ax.get_ylim()[1] for ax in AX]))
    for ax in AX:
        ax.set_ylim(YLIM)
        data.add_stim(ax)
        ax.axis('off')
    add_bar(AX[0], Xbar=2, Ybar=1)
    ax = plt.axes([0.84,0.25,0.13,0.6])
    responsive = responsiveness(data)
    SI = orientation_selectivity_plot(*data.compute_integral_responses('angle'), ax=ax, color=('k' if responsive else 'lightgray'))
    ax.annotate('SI=%.2f ' % SI, (1, 0.97), va='top', ha='right', xycoords='figure fraction',
                weight='bold', fontsize=9, color=('k' if responsive else 'lightgray'))
    ax.annotate(('responsive' if responsive else 'unresponsive'), (0.85, 0.97), ha='left', va='top',
                xycoords='figure fraction', weight='bold', fontsize=9, color=(plt.cm.tab10(2) if responsive else plt.cm.tab10(3)))
    AX[0].annotate(' ROI#%i' % (roiIndex+1), (0, 0.1), xycoords='figure fraction', weight='bold', fontsize=9)
    return fig, SI, responsive

if __name__=='__main__':
    
    filename = os.path.join(os.path.expanduser('~'), 'DATA', 'Wild_Type', '2021_03_11-17-13-03.nwb')
    FullData= Data(filename)
    print('the datafile has %i validated ROIs (over %i from the full suite2p output) ' % (np.sum(FullData.iscell),
                                                                                          len(FullData.iscell)))
    # for i in [2, 6, 9, 10, 13, 15, 16, 17, 21, 38, 41, 136]: # for 2021_03_11-17-13-03.nwb
    # # for i in range(np.sum(FullData.iscell))[:5]:
    #     # fig1, _, _ = orientation_selectivity_analysis(FullData, roiIndex=i)
    #     fig2, _, _ = direction_selectivity_analysis(FullData, roiIndex=i)
    #     # fig1.savefig(os.path.join(os.path.expanduser('~'), 'Desktop', 'data3', 'ROI#%i.svg' % (i+1)))
    #     plt.show()


