# general modules
import os, sys

# custom modules
sys.path.append('.')
from physion.analysis.orientation_direction_selectivity import *
from datavyz import ge


SURROUND_SUPPRESSION_PROTOCOLS = ['surround-suppression-fast']

def size_dependence_plot(sizes, responses, baselines, ax=None, figsize=(2.5,1.5), color='k'):
    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)
    ax.plot([0]+list(sizes), [np.mean(baselines)/1e3]+list(responses/1e3), 'o-', color=color, lw=2) # CHECK UNITS
    ax.set_xticks([0]+list(sizes))
    ax.set_ylabel('resp. integral ($\Delta$F/F.s)', fontsize=9)
    ax.set_xlabel('angle ($^o$)', fontsize=9)

    
def orientation_size_selectivity_analysis(FullData, roiIndex=0, with_std=True, verbose=True, subprotocol_id=0):
    iprotocol = [i for (i,p) in enumerate(FullData.protocols) if (p in SURROUND_SUPPRESSION_PROTOCOLS)][subprotocol_id]
    data = CellResponse(FullData, protocol_id=iprotocol, quantity='CaImaging', subquantity='dF/F', roiIndex = roiIndex, verbose=verbose)
    fig, AX = plt.subplots(len(data.varied_parameters['angle']), len(data.varied_parameters['radius'])+1,
                           figsize=(11.4,2.*len(data.varied_parameters['angle'])))
    plt.subplots_adjust(left=0.03, top=0.95, bottom=0.12, right=.97)
    for i, angle in enumerate(data.varied_parameters['angle']):
        for j, size in enumerate(data.varied_parameters['radius']):
            if 'contrast' in data.varied_parameters:
                for ic, c in enumerate(data.varied_parameters['contrast']):
                    data.plot(['angle', 'radius', 'contrast'], [i,j,ic], ax=AX[i,j], with_std=with_std,
                              color=plt.cm.gray(0.5-ic/len(data.varied_parameters['radius'])/2.))
            else:
                data.plot(['angle', 'radius'], [i,j], ax=AX[i,j], with_std=with_std)
            AX[i,j].set_title('r=%.0f$^o$, $\\theta=$%.0f$^o$' % (size,angle), fontsize=9)
        if 'contrast' in data.varied_parameters:
            for ic, c in enumerate(data.varied_parameters['contrast']):
                size_dependence_plot(*data.compute_integral_responses('radius',
                                                                      keys=['angle', 'contrast'], indexes=[i, ic],
                                                                      with_baseline=True),
                                                                      color=plt.cm.gray(0.5-ic/len(data.varied_parameters['radius'])/2.),
                                                                      ax=AX[i,j+1])
        else:
            size_dependence_plot(*data.compute_integral_responses('radius', keys=['angle'], indexes=[i], with_baseline=True), ax=AX[i,j+1])
    responsive = responsiveness(data)
    YLIM = (np.min([ax.get_ylim()[0] for ax in ge.flat(AX)]), np.max([ax.get_ylim()[1] for ax in ge.flat(AX[:,:-1])]))
    for ax in ge.flat(AX[:,:-1]):
        ax.set_ylim(YLIM)
        data.add_stim(ax)
        ax.axis('off')
    YLIM = (np.min([ax.get_ylim()[0] for ax in ge.flat(AX)]), np.max([ax.get_ylim()[1] for ax in ge.flat(AX[:,-1:])]))
    for ax in ge.flat(AX[:,-1:]):
        ax.set_ylim(YLIM)
    add_bar(AX[0, 0], Xbar=2, Ybar=1)
    AX[0,j+1].annotate(('responsive' if responsive else 'unresponsive'), (0.85, 0.97), ha='left', va='top',
                       xycoords='figure fraction', weight='bold', fontsize=9, color=(plt.cm.tab10(2) if responsive else plt.cm.tab10(3)))
    AX[0,0].annotate(' ROI#%i' % (roiIndex+1), (0, 0.02), xycoords='figure fraction', weight='bold', fontsize=9)
    return fig, responsive



if __name__=='__main__':
    
    filename = os.path.join(os.path.expanduser('~'), 'DATA', 'CaImaging', 'Wild_Type_GCamp6s', '2021_04_15', '2021_04_15-15-48-07.nwb')
    # filename = sys.argv[-1]
    
    FullData= Data(filename)
    print(FullData.protocols)
    print('the datafile has %i validated ROIs (over %i from the full suite2p output) ' % (np.sum(FullData.iscell),
                                                                                          len(FullData.iscell)))
    # # for i in [2, 6, 9, 10, 13, 15, 16, 17, 21, 38, 41, 136]: # for 2021_03_11-17-13-03.nwb
    for i in range(np.sum(FullData.iscell))[:3]:
        fig, responsive = orientation_size_selectivity_analysis(FullData, roiIndex=i)
        if responsive:
            print('cell %i -> responsive !' % (i+1))
        plt.show()








