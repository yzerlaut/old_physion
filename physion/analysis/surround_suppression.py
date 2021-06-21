# general modules
import os, sys

# custom modules
sys.path.append('.')
from physion.analysis.orientation_direction_selectivity import *
from datavyz import ge


SURROUND_SUPPRESSION_PROTOCOLS = ['surround-suppression-fast', 'surround-suppression-static', 'surround-suppression-drifting']

def size_dependence_plot(sizes, responses, baselines, ax=None, figsize=(2.5,1.5), color='k'):
    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)
    ax.plot([0]+list(sizes), [np.mean(baselines)/1e3]+list(responses/1e3), 'o-', color=color, lw=2) # CHECK UNITS
    ax.set_xticks([0]+list(sizes))
    ax.set_xticklabels(['%.1f' % f for f in [0]+list(sizes)])
    ax.set_ylabel('resp. integral ($\Delta$F/F.s)', fontsize=9)
    ax.set_xlabel('angle ($^o$)', fontsize=9)

    
def orientation_size_selectivity_analysis(FullData, roiIndex=0, with_std=True, verbose=True, subprotocol_id=0):
    iprotocol = [i for (i,p) in enumerate(FullData.protocols) if (p in SURROUND_SUPPRESSION_PROTOCOLS)][subprotocol_id]
    data = CellResponse(FullData, protocol_id=iprotocol, quantity='CaImaging', subquantity='dF/F', roiIndex = roiIndex, verbose=verbose)

    if 'x-center' not in data.varied_parameters:
        data.varied_parameters['x-center'] = [data.metadata['x-center-1']]
    if 'y-center' not in data.varied_parameters:
        data.varied_parameters['y-center'] = [data.metadata['y-center-1']]
    if 'contrast' not in data.varied_parameters:
        data.varied_parameters['contrast'] = [data.metadata['contrast-1']]

    Ny = len(data.varied_parameters['angle'])*len(data.varied_parameters['x-center'])*len(data.varied_parameters['y-center'])
    fig, AX = plt.subplots(Ny, len(data.varied_parameters['radius'])+1,
                           figsize=(11.4,2.*Ny))
    plt.subplots_adjust(left=0.03, top=1-0.1/Ny, bottom=0.2/Ny, right=.97)

    for i, angle in enumerate(data.varied_parameters['angle']):
        for j, size in enumerate(data.varied_parameters['radius']):
            for ix, x in enumerate(data.varied_parameters['x-center']):
                for iy, y in enumerate(data.varied_parameters['y-center']):
                    iax = i*len(data.varied_parameters['x-center'])*len(data.varied_parameters['y-center'])+\
                        ix*len(data.varied_parameters['y-center'])+iy
                    for ic, c in enumerate(data.varied_parameters['contrast']):
                        data.plot(['angle', 'radius', 'contrast', 'x-center', 'y-center'], [i,j,ic,ix,iy],
                                  ax=AX[iax,j], with_std=with_std,
                                  color=plt.cm.gray(0.5-ic/len(data.varied_parameters['radius'])/2.))
                        AX[iax,j].set_title('r=%.0f$^o$, $\\theta=$%.0f$^o$, x=%.1f$^o$, y=%.1f$^o$' % (size,angle,x,y), fontsize=7)
    for ic, c in enumerate(data.varied_parameters['contrast']):
        AX[0,0].annotate('contrast=%.1f' % c + ic*'\n', (0.5,0), xycoords='figure fraction',
                         color=plt.cm.gray(0.5-ic/len(data.varied_parameters['radius'])/2.))

    for i, angle in enumerate(data.varied_parameters['angle']):
        for ix, x in enumerate(data.varied_parameters['x-center']):
            for iy, y in enumerate(data.varied_parameters['y-center']):
                iax = i*len(data.varied_parameters['x-center'])*len(data.varied_parameters['y-center'])+\
                    ix*len(data.varied_parameters['y-center'])+iy
                for ic, c in enumerate(data.varied_parameters['contrast']):
                    size_dependence_plot(*data.compute_integral_responses('radius',
                                                                          keys=['angle', 'contrast', 'x-center', 'y-center'],
                                                                          indexes=[i, ic, ix, iy],
                                                                          with_baseline=True),
                                         color=plt.cm.gray(0.5-ic/len(data.varied_parameters['radius'])/2.),
                                         ax=AX[iax,len(data.varied_parameters['radius'])])
                    
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
    
    filename = os.path.join(os.path.expanduser('~'), 'DATA', 'CaImaging', 'Wild_Type_GCamp6s', '2021_04_28', '2021_04_28-15-05-26.nwb')
    filename = os.path.join(os.path.expanduser('~'), 'DATA', 'CaImaging', 'Wild_Type_GCamp6s', '2021_04_28', '2021_04_28-15-05-26.nwb')
    # filename = sys.argv[-1]
    
    FullData= Data(filename)
    print(FullData.protocols)
    print('the datafile has %i validated ROIs (over %i from the full suite2p output) ' % (np.sum(FullData.iscell),
                                                                                          len(FullData.iscell)))
    # # for i in [2, 6, 9, 10, 13, 15, 16, 17, 21, 38, 41, 136]: # for 2021_03_11-17-13-03.nwb
    for i in range(np.sum(FullData.iscell))[:1]:
        fig, responsive = orientation_size_selectivity_analysis(FullData, roiIndex=i)
        if responsive:
            print('cell %i -> responsive !' % (i+1))
        plt.show()








