# general modules
import pynwb, os, sys
import numpy as np
import matplotlib.pylab as plt
from scipy.ndimage import gaussian_filter    
# custom modules
sys.path.append('.')
from physion.analysis.read_NWB import Data
from physion.Ca_imaging.tools import compute_CaImaging_trace
from physion.visual_stim.psychopy_code.stimuli import build_stim

class CellResponse:

    def __init__(self, data,
                 protocol_id=0,
                 quantity='CaImaging', 
                 subquantity='dF/F', 
                 roiIndex = 0,
                 verbose=False):
        """ build the episodes corresponding to a specific protocol and a ROIR"""
        for key in data.__dict__.keys():
            setattr(self, key, getattr(data, key))
        # if verbose:
        #     print('Building episodes for "%s"' % self.protocols[protocol_id])
        # data.CaImaging_key, data.roiIndices = subquantity, [roiIndex]
        # self.EPISODES = build_episodes(data, protocol_id=protocol_id, quantity=quantity)
        # self.metadata = data.metadata
        # pass

        # self.varied_parameters = data.varied_parameters
        self.dF = compute_CaImaging_trace(self, subquantity, [roiIndex]).sum(axis=0) # validROI inside
        if 'protocol_id' in self.nwbfile.stimulus:
            self.Pcond = (self.nwbfile.stimulus['protocol_id'].data[:]==protocol_id)
        else:
            self.Pcond = np.ones(self.nwbfile.stimulus['time_start_realigned'].num_samples, dtype=bool)
    
    def reverse_correlation(self, roiIndex, subquantity='dF/F',
                            metrics='mean'):

        dF = compute_CaImaging_trace(self, subquantity, [roiIndex]).sum(axis=0) # validROI inside
        self.t = self.Fluorescence.timestamps[:]
        
        full_img, cum_weight = np.zeros(self.visual_stim.screen['resolution'], dtype=float).T, 0

        for i in np.arange(self.nwbfile.stimulus['time_start_realigned'].num_samples)[self.Pcond]:
            tstart = self.nwbfile.stimulus['time_start_realigned'].data[i]
            tstop = self.nwbfile.stimulus['time_stop_realigned'].data[i]
            cond = (self.t>tstart) & (self.t<tstop)
            weight = np.inf
            if metrics == 'mean':
                weight = np.mean(dF[cond])
            elif np.sum(cond)>0 and (metrics=='max'):
                weight = np.max(dF[cond])
            if np.isfinite(weight):
                full_img += weight*2*(self.visual_stim.get_image(i)-.5)
                cum_weight += weight
            else:
                print('For episode #%i in t=(%.1f, %.1f), pb in estimating the weight !' % (i, tstart, tstop) )
                
        return full_img/cum_weight


def RF_analysis(FullData, roiIndex=0, verbose=True, subprotocol_id=0):
    iprotocol = [i for (i,p) in enumerate(FullData.protocols) if ('noise' in p)][subprotocol_id]
    data = CellResponse(FullData, protocol_id=iprotocol,
                        quantity='CaImaging', subquantity='dF/F',
                        roiIndex = roiIndex, verbose=verbose)
    fig, ax = plt.subplots(1, figsize=(11.4,3))
    plt.subplots_adjust(left=0.3, right=.7, top=0.85, bottom=0.15)
    ax.axis('off')
    img = data.reverse_correlation(i, subquantity='Deconvolved', metrics='max')
    img = gaussian_filter(img, (10,10))
    ax.imshow(img, cmap=plt.cm.PiYG,
               vmin=-np.max(np.abs(img)), vmax=np.max(np.abs(img)))
    
    # for i, angle in enumerate(data.varied_parameters['angle']):
    #     data.plot('angle',i, ax=AX[i], with_std=True)
    #     AX[i].set_title('%.0f$^o$' % angle, fontsize=9)
    # # put all on the same axis range
    # YLIM = (np.min([ax.get_ylim()[0] for ax in AX]), np.max([ax.get_ylim()[1] for ax in AX]))
    # for ax in AX:
    #     ax.set_ylim(YLIM)
    #     data.add_stim(ax)
    #     ax.axis('off')
    # # add scale bar
    # add_bar(AX[0], Xbar=2, Ybar=1)
    # # Orientation selectivity plot based on the integral of the trial-averaged response
    # ax = plt.axes([0.88, 0.2, 0.11, 0.6], projection='polar')
    # responsive = responsiveness(data)
    # SI = direction_selectivity_plot(*data.compute_integral_responses('angle'), ax=ax, color=('k' if responsive else 'lightgray'))
    # ax.annotate('SI=%.2f ' % SI, (1, 0.97), va='top', ha='right', xycoords='figure fraction',
    #             weight='bold', fontsize=9, color=('k' if responsive else 'lightgray'))
    # ax.annotate(('responsive' if responsive else 'unresponsive'), (0.85, 0.97), ha='left', va='top',
    #             xycoords='figure fraction', weight='bold', fontsize=9, color=(plt.cm.tab10(2) if responsive else plt.cm.tab10(3)))
    ax.annotate(' ROI#%i' % (roiIndex+1), (0, 0.1), xycoords='figure fraction', weight='bold', fontsize=9)
    SI, responsive = 0, False
    return fig, SI, responsive
    
if __name__=='__main__':
    
    filename = os.path.join(os.path.expanduser('~'), 'DATA', 'Wild_Type', '2021_03_11-17-32-34.nwb')
    FullData = Data(filename, with_visual_stim=True)
    print(FullData.protocols)
    
    for i in range(np.sum(FullData.iscell))[:1]:
        print('ROI#', i+1)
        fig1, _, _ = RF_analysis(FullData, roiIndex=i)
    
    # ax.axis('off')
    # fig.savefig(os.path.join(os.path.expanduser('~'), 'Desktop', 'RF', 'ROI#%i.png' % (i+1)))
    plt.show()
    plt.close()
    





