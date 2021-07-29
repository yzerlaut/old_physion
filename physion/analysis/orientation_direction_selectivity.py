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

# we define a data object fitting this analysis purpose

ORIENTATION_PROTOCOLS = ['Pakan-et-al-static']
DIRECTION_PROTOCOLS = ['Pakan-et-al-drifting']

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
        if verbose:
            print('Building episodes for "%s"' % self.protocols[protocol_id])
        data.CaImaging_key, data.roiIndices = subquantity, [roiIndex]
        self.EPISODES = build_episodes(data, protocol_id=protocol_id, quantity=quantity, verbose=verbose)
        self.varied_parameters = data.varied_parameters
        self.metadata = data.metadata


    def find_cond(self, key, index):
        if (type(key) in [list, np.ndarray]) and (type(index) in [list, np.ndarray]) :
            cond = (self.EPISODES[key[0]]==self.varied_parameters[key[0]][index[0]])
            for n in range(1, len(key)):
                cond = cond & (self.EPISODES[key[n]]==self.varied_parameters[key[n]][index[n]])
        else:
            cond = (self.EPISODES[key]==self.varied_parameters[key][index])
        return cond
    
    def compute_repeated_trials(self, key, index):
        cond = self.find_cond(key, index)
        return np.mean(self.EPISODES['resp'][cond,:], axis=0), np.std(self.EPISODES['resp'][cond,:], axis=0), np.sum(cond)

    
    def responsiveness_wrt_prestim(self, key, index,
                                   threshold=1e-3, higher_only=True):
        """
        anova test between pre and post means to decide if responsive
        """
        cond = self.find_cond(key, index)
        pre_cond = (self.EPISODES['t']<=0)
        stim_cond = (self.EPISODES['t']>0) &\
            (self.EPISODES['t']<self.EPISODES['time_duration'][0])
        pre=[np.mean(self.EPISODES['resp'][ic,:][pre_cond]) for ic, c in enumerate(cond) if c]
        post=[np.mean(self.EPISODES['resp'][ic,:][stim_cond]) for ic, c in enumerate(cond) if c]
        f, pval = stats.f_oneway(pre, post)
        if higher_only and (np.mean(post)>np.mean(pre)) and (pval<threshold):
            return True
        elif pval<threshold and not higher_only:
            return True
        else:
            return False
    
    def compute_integral_responses(self, key, keys=None, indexes=None, with_baseline=False):
        integrals, baselines = [], []
        for i, value in enumerate(self.varied_parameters[key]):
            if keys is not None:
                mean_response, _, _ = self.compute_repeated_trials([key]+list(keys), [i]+list(indexes))
            else:
                mean_response, _, _ = self.compute_repeated_trials(key, i)
            stim_cond = (self.EPISODES['t']>0) & (self.EPISODES['t']<=self.EPISODES['time_duration'][0])
            if with_baseline:
                pre_cond = (self.EPISODES['t']<=0) & (self.EPISODES['t']>-self.EPISODES['time_duration'][0])
                baselines.append(np.trapz(mean_response[pre_cond]))        
            integrals.append(np.trapz(mean_response[stim_cond]))        
        if with_baseline:
            return self.varied_parameters[key], np.array(integrals), np.array(baselines)
        else:
            return self.varied_parameters[key], np.array(integrals)
            
    
    def plot(self, key, index, ax=None, with_std=True, with_N=True, with_bars={}, color='k'):
        if ax is None:
            _, ax = plt.subplots(1)
            ax.axis('off')
        my, sy, ny = self.compute_repeated_trials(key, index)
        self.responsiveness_wrt_prestim(key, index)
        ax.plot(self.EPISODES['t'], my, color=color, lw=1)
        if with_std:
            ax.fill_between(self.EPISODES['t'], my-sy, my+sy, alpha=0.1, color='k', lw=0)
        if with_N:
            ax.annotate('n=%i' % ny, (1,1), xycoords='axes fraction', ha='right', va='top', fontsize=7)
            
    def add_stim(self, ax):
        ylim = ax.get_ylim()
        ax.fill_between([0, self.EPISODES['time_duration'][0]], 
                        ylim[0]*np.ones(2), ylim[1]*np.ones(2), lw=0, color='grey', alpha=0.1)
    
    
    def show_stim(self, key, figsize=(15,2), with_arrow=False):
        fig, AX = plt.subplots(1, len(self.varied_parameters[key]), figsize=figsize)
        self.metadata['load_from_protocol_data'], self.metadata['no-window'] = True, True
        visual_stim = build_stim(self.metadata, no_psychopy=True)
        for i, value in enumerate(self.varied_parameters[key]):
            AX[i].set_title('%.0f$^o$' % value, fontsize=9)
            icond = np.argwhere(self.nwbfile.stimulus[key].data[:]==value)[0][0]
            if with_arrow:
                visual_stim.show_frame(icond, ax=AX[i],
                                       arrow={'direction':value, 'center':(0,0),
                                              'length':25,
                                              'width_factor':0.1, 'color':'red'},
                                    label={'degree':10, 'shift_factor':0.02, 'lw':1, 'fontsize':11})
            else:
                visual_stim.show_frame(icond, ax=AX[i],
                                    label={'degree':10, 'shift_factor':0.02, 'lw':1, 'fontsize':11})
        return fig, AX
    

def add_bar(ax, Ybar=1, Ybar_unit='$\Delta$F/F', Xbar=1, Xbar_unit='s', fontsize=10):
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    dx, dy = .02*(xlim[1]-xlim[0]), .02*(ylim[1]-ylim[0])
    ax.plot([xlim[0]-dx, xlim[0]-dx+Xbar], [ylim[1]-dy, ylim[1]-dy], 'k', lw=1)
    ax.plot([xlim[0]-dx, xlim[0]-dx], [ylim[1]-dy, ylim[1]-dy-Ybar], 'k', lw=1)
    ax.annotate(str(Ybar)+Ybar_unit, (xlim[0], ylim[1]), xycoords='data', ha='right', va='top', rotation=90, fontsize=fontsize)
    ax.annotate(str(Xbar)+Xbar_unit, (xlim[0], ylim[1]), xycoords='data', fontsize=fontsize)

    
def responsiveness(data,
                   # pre_std_threshold=3.,
                   threshold=1e-2):
    """
    mean response going above fluctuations ?
    """
    responsive = False
    for i, angle in enumerate(data.varied_parameters['angle']):
        resp = data.responsiveness_wrt_prestim('angle', i,
                                               threshold=threshold, higher_only=True)
        if resp:
            responsive=True
        # my, sy, ny = data.compute_repeated_trials('angle', i)
        # stim_cond = (data.EPISODES['t']>0) & (data.EPISODES['t']<data.EPISODES['time_duration'][0])
        # pre_cond = (data.EPISODES['t']<=0)
        # if np.max(my[stim_cond])>pre_std_threshold*np.mean(sy[pre_cond]):
        #     responsive=True
    return responsive
        
def orientation_selectivity_index(angles, resp):
    """
    computes 
    """
    imax = np.argmax(resp)
    iop = np.argmin(((angles[imax]+90)%(180)-angles)**2)
    if (resp[imax]>0):
        return min([1,max([0,(resp[imax]-resp[iop])/(resp[imax]+resp[iop])])])
    else:
        return 0

def orientation_selectivity_plot(angles, responses, ax=None, figsize=(2.5,1.5), color='k'):
    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)
    ax.plot(angles, responses/1e3, color=color, lw=2) # CHECK UNITS
    SI = orientation_selectivity_index(angles, responses)
    ax.set_xticks(angles)
    ax.set_ylabel('resp. integral ($\Delta$F/F.s)', fontsize=9)
    ax.set_xlabel('angle ($^o$)', fontsize=9)
    return SI

def OS_ROI_analysis(FullData, roiIndex=0, with_std=True, verbose=True, subprotocol_id=0):
    """
    orientation selectivity ROI analysis
    """

    iprotocol = [i for (i,p) in enumerate(FullData.protocols) if (p in ORIENTATION_PROTOCOLS)][subprotocol_id]
    
    fig, AX = FullData.plot_trial_average(protocol_id=iprotocol,
                                          quantity='CaImaging', subquantity='dF/F',
                                          roiIndex = roiIndex,
                                          ybar=1., ybarlabel='1dF/F',
                                          with_stat_test=True,
                                          verbose=verbose)
    
    # fig, AX = plt.subplots(1, len(data.varied_parameters['angle']), figsize=(11.4,2.5))
    # plt.subplots_adjust(left=0.03, right=.75, top=0.9, bottom=0.05)
    # for i, angle in enumerate(data.varied_parameters['angle']):
    #     data.plot('angle',i, ax=AX[i], with_std=with_std)
    #     AX[i].set_title('%.0f$^o$' % angle, fontsize=9)
    # YLIM = (np.min([ax.get_ylim()[0] for ax in AX]), np.max([ax.get_ylim()[1] for ax in AX]))
    # for ax in AX:
    #     ax.set_ylim(YLIM)
    #     data.add_stim(ax)
    #     ax.axis('off')
    # add_bar(AX[0], Xbar=2, Ybar=1)
    # ax = plt.axes([0.84,0.25,0.13,0.6])
    responsive = responsiveness(data)
    SI = orientation_selectivity_plot(*data.compute_integral_responses('angle'), ax=ax, color=('k' if responsive else 'lightgray'))
    # ax.annotate('SI=%.2f ' % SI, (1, 0.97), va='top', ha='right', xycoords='figure fraction',
    #             weight='bold', fontsize=9, color=('k' if responsive else 'lightgray'))
    # ax.annotate(('responsive' if responsive else 'unresponsive'), (0.85, 0.97), ha='left', va='top',
    #             xycoords='figure fraction', weight='bold', fontsize=9, color=(plt.cm.tab10(2) if responsive else plt.cm.tab10(3)))
    AX[0].annotate(' ROI#%i' % (roiIndex+1), (0, 0.1), xycoords='figure fraction', weight='bold', fontsize=9)
    return fig, SI, responsive


def direction_selectivity_plot(angles, responses, ax=None, figsize=(1.5,1.5), color='k'):
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.axes([0,0,1,1], projection='polar')
    ax.set_theta_direction(-1)
    Angles = angles*np.pi/180.
    ax.plot(np.concatenate([Angles, [Angles[0]]]), np.concatenate([responses, [responses[0]]]), color=color, lw=2)
    ax.fill_between(np.concatenate([Angles, [Angles[0]]]), np.zeros(len(Angles)+1),
                    np.concatenate([responses, [responses[0]]]), color='k', lw=0, alpha=0.3)
    ax.set_rticks([])
    return orientation_selectivity_index(Angles, responses)


def DS_ROI_analysis(FullData, roiIndex=0, verbose=True, subprotocol_id=0):
    """
    direction selectivity ROI analysis
    """
    iprotocol = [i for (i,p) in enumerate(FullData.protocols) if (p in DIRECTION_PROTOCOLS)][subprotocol_id]
    data = CellResponse(FullData, protocol_id=iprotocol,
                        quantity='CaImaging', subquantity='dF/F',
                        roiIndex = roiIndex, verbose=verbose)
    fig, AX = plt.subplots(1, len(data.varied_parameters['angle']), figsize=(11.4,2.5))
    plt.subplots_adjust(left=0.02, right=.85, top=0.85)
    for i, angle in enumerate(data.varied_parameters['angle']):
        data.plot('angle',i, ax=AX[i], with_std=True)
        AX[i].set_title('%.0f$^o$' % angle, fontsize=9)
    # put all on the same axis range
    YLIM = (np.min([ax.get_ylim()[0] for ax in AX]), np.max([ax.get_ylim()[1] for ax in AX]))
    for ax in AX:
        ax.set_ylim(YLIM)
        data.add_stim(ax)
        ax.axis('off')
    # add scale bar
    add_bar(AX[0], Xbar=2, Ybar=1)
    # Orientation selectivity plot based on the integral of the trial-averaged response
    ax = plt.axes([0.88, 0.2, 0.11, 0.6], projection='polar')
    responsive = responsiveness(data)
    SI = direction_selectivity_plot(*data.compute_integral_responses('angle'), ax=ax, color=('k' if responsive else 'lightgray'))
    ax.annotate('SI=%.2f ' % SI, (1, 0.97), va='top', ha='right', xycoords='figure fraction',
                weight='bold', fontsize=9, color=('k' if responsive else 'lightgray'))
    ax.annotate(('responsive' if responsive else 'unresponsive'), (0.85, 0.97), ha='left', va='top',
                xycoords='figure fraction', weight='bold', fontsize=9, color=(plt.cm.tab10(2) if responsive else plt.cm.tab10(3)))
    AX[0].annotate(' ROI#%i' % (roiIndex+1), (0, 0.1), xycoords='figure fraction', weight='bold', fontsize=9)
    return fig, SI, responsive

def summary_fig(Nresp, Ntot, quantity,
                label='Orient. Select. Index',
                labels=['responsive', 'unresponsive']):
    fig, AX = plt.subplots(1, 4, figsize=(11.4, 2.5))
    fig.subplots_adjust(left=0.1, right=0.8, bottom=0.2)
    AX[1].pie([100*Nresp/Ntot, 100*(1-Nresp/Ntot)], explode=(0, 0.1),
              colors=[plt.cm.tab10(2), plt.cm.tab10(3)],
              labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
    AX[1].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    AX[3].hist(quantity)
    AX[3].set_xlabel(label, fontsize=9)
    AX[3].set_ylabel('count', fontsize=9)
    for ax in [AX[0], AX[2]]:
        ax.axis('off')
    return fig, AX
    

if __name__=='__main__':
    
    filename = os.path.join(os.path.expanduser('~'), 'DATA', 'Wild_Type', '2021_03_11-17-13-03.nwb')
    
    filename = sys.argv[-1]
    FullData= Data(filename)
    print('the datafile has %i validated ROIs (over %i from the full suite2p output) ' % (np.sum(FullData.iscell),
                                                                                          len(FullData.iscell)))
    # for i in [2, 6, 9, 10, 13, 15, 16, 17, 21, 38, 41, 136]: # for 2021_03_11-17-13-03.nwb
    # # for i in range(np.sum(FullData.iscell))[:5]:
    #     # fig1, _, _ = orientation_selectivity_analysis(FullData, roiIndex=i)
    # for i in [2, 6, 9, 10, 13, 15, 16, 17, 21, 38, 41, 136]: # for 2021_03_11-17-13-03.nwb
    for i in range(np.sum(FullData.iscell)):
        fig2, SI, responsive = direction_selectivity_analysis(FullData, roiIndex=i)
        if responsive:
            fig2.savefig(os.path.join(os.path.expanduser('~'), 'Desktop', 'ODS', 'ROI#%i.png' % (i+1)), dpi=300)
        # plt.show()


