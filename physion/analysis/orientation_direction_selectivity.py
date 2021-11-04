import sys, os, pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from datavyz import graph_env_manuscript as ge

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from dataviz.show_data import MultimodalData
from analysis.tools import summary_pdf_folder
from analysis.process_NWB import EpisodeResponse
from analysis import stat_tools
from analysis.summary_pdf import summary_fig


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


def OS_ROI_analysis(FullData,
                    roiIndex=0,
                    iprotocol = 0,
                    subprotocol_id=0,
                    verbose=False,
                    response_significance_threshold=0.01,
                    with_responsive_angles = False,
                    stat_test_props=dict(interval_pre=[-2,0], interval_post=[1,3],
                                         test='wilcoxon', positive=True)):
    """
    orientation selectivity ROI analysis
    """

    EPISODES = EpisodeResponse(FullData,
                               protocol_id=iprotocol,
                               quantity='CaImaging', subquantity='dF/F',
                               roiIndex = roiIndex)

    fig, AX = FullData.plot_trial_average(EPISODES=EPISODES,
                                          quantity='CaImaging', subquantity='dF/F',
                                          roiIndex = roiIndex,
                                          column_key='angle',
                                          ybar=1., ybarlabel='1dF/F',
                                          xbar=1., xbarlabel='1s',
                                          fig_preset='raw-traces-preset+right-space',
                                          with_annotation=True,
                                          with_stat_test=True, stat_test_props=stat_test_props,
                                          verbose=verbose)
    
    ax = ge.inset(fig, (0.88,0.4,0.1,0.4))
    angles, y, sy, responsive_angles = [], [], [], []
    responsive = False
    
    for i, angle in enumerate(EPISODES.varied_parameters['angle']):

        stats = EPISODES.stat_test_for_evoked_responses(episode_cond=EPISODES.find_episode_cond('angle', i),
                                                        **stat_test_props)

        angles.append(angle)
        y.append(np.mean(stats.y)) # means "post"
        sy.append(np.std(stats.y)) # std "post"
        
        if stats.significant(threshold=response_significance_threshold):
            responsive = True
            responsive_angles.append(angle)
            
    ge.plot(angles, np.array(y), sy=np.array(sy), ax=ax,
            axes_args=dict(ylabel='<post dF/F>         ', xlabel='angle ($^{o}$)',
                           xticks=angles, size='small'), m='o', ms=2, lw=1)

    SI = orientation_selectivity_index(angles, y)
    ge.annotate(fig, 'SI=%.2f ' % SI, (1, 0.97), va='top', ha='right', xycoords='figure fraction',
                weight='bold', fontsize=8, color=('k' if responsive else 'lightgray'))
    ge.annotate(fig, ('responsive' if responsive else 'unresponsive'), (0.78, 0.98), ha='left', va='top',
                xycoords='figure fraction', weight='bold', fontsize=8, color=(plt.cm.tab10(2) if responsive else plt.cm.tab10(3)))
    
    if with_responsive_angles:
        return fig, SI, responsive, responsive_angles
    else:
        return fig, SI, responsive




def direction_selectivity_plot(angles, responses, ax=None, figsize=(1.5,1.5), color='k'):
    """
    polar plot of direction selectivity
    """
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


def DS_ROI_analysis(FullData,
                    roiIndex=0,
                    iprotocol=0,
                    verbose=False,
                    response_significance_threshold=0.01,
                    with_responsive_angles = False,
                    fig=None, AX=None,
                    CaImaging_options = dict(quantity='CaImaging', subquantity='dF/F'),
                    stat_test_props=dict(interval_pre=[-2,0], interval_post=[1,3],
                                         test='wilcoxon'),
                    inset_coords=(0.92,0.4,0.07,0.4)):
    """
    direction selectivity ROI analysis
    """

    EPISODES = EpisodeResponse(FullData,
                               protocol_id=iprotocol,
                               roiIndex = roiIndex, verbose=verbose, **CaImaging_options)
    
    fig, AX = FullData.plot_trial_average(EPISODES=EPISODES,
                                          protocol_id=iprotocol,
                                          roiIndex = roiIndex,
                                          column_key='angle',
                                          ybar=1., ybarlabel='1dF/F',
                                          xbar=1., xbarlabel='1s',
                                          fig=fig, AX=AX, no_set=False,
                                          fig_preset='raw-traces-preset+right-space',
                                          with_annotation=True,
                                          with_stat_test=True, stat_test_props=stat_test_props,
                                          verbose=verbose, **CaImaging_options)
    
    ax = ge.inset(fig, inset_coords)
    angles, y, sy, responsive_angles = [], [], [], []
    responsive = False
    for i, angle in enumerate(EPISODES.varied_parameters['angle']):
        means_pre = EPISODES.compute_stats_over_repeated_trials('angle', i,
                                                                interval_cond=EPISODES.compute_interval_cond(stat_test_props['interval_pre']),
                                                                quantity='mean')
        means_post = EPISODES.compute_stats_over_repeated_trials('angle', i,
                                                                 interval_cond=EPISODES.compute_interval_cond(stat_test_props['interval_post']),
                                                                 quantity='mean')

        stats = EPISODES.stat_test_for_evoked_responses(episode_cond=EPISODES.find_episode_cond('angle', i),
                                                        **stat_test_props)

        angles.append(angle)
        y.append(np.mean(means_post))
        sy.append(np.std(means_post))
        
        if stats.significant(response_significance_threshold):
            responsive = True
            responsive_angles.append(angle)
            
    ge.plot(angles, np.array(y), sy=np.array(sy), ax=ax,
            axes_args=dict(ylabel='<post dF/F>         ', xlabel='angle ($^{o}$)',
                           xticks=np.array(angles)[::int(len(angles)/4)], size='small'), m='o', ms=2, lw=1)

    SI = orientation_selectivity_index(angles, y)
    ge.annotate(ax, 'SI=%.2f\n ' % SI, (1, 1), ha='right',
                weight='bold', fontsize=8, color=('k' if responsive else 'lightgray'))
    ge.annotate(ax, ('responsive' if responsive else 'unresponsive'), (0., 1), ha='left',
                weight='bold', fontsize=8, color=(plt.cm.tab10(2) if responsive else plt.cm.tab10(3)))
    
    if with_responsive_angles:
        return fig, SI, responsive, responsive_angles
    else:
        return fig, SI, responsive


def summary_fig2(Nresp, Ntot, quantity,
                label='Orient. Select. Index',
                labels=['responsive', 'unresponsive']):
    fig, AX = ge.figure(axes=(4, 1))
    AX[1].pie([100*Nresp/Ntot, 100*(1-Nresp/Ntot)], explode=(0, 0.1),
              colors=[plt.cm.tab10(2), plt.cm.tab10(3)],
              labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
    AX[1].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ge.hist(quantity, bins=np.linspace(0,1,20), ax=AX[3], axes_args=dict(xlabel=label, ylabel='count', xlim=[0,1]))
    for ax in [AX[0], AX[2]]:
        ax.axis('off')
    return fig, AX
    

def OS_analysis_pdf(datafile, iprotocol=0, Nmax=1000000):

    data = MultimodalData(datafile)
    
    pdf_filename = os.path.join(summary_pdf_folder(datafile), '%s-orientation_selectivity.pdf' % data.protocols[iprotocol])

    Nresp, SIs = 0, []
    with PdfPages(pdf_filename) as pdf:

        for roi in np.arange(data.iscell.sum())[:Nmax]:
            
            print('   - orientation-selectivity analysis for ROI #%i / %i' % (roi+1, data.iscell.sum()))
            fig, SI, responsive = OS_ROI_analysis(data, roiIndex=roi, iprotocol=iprotocol)
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()

            if responsive:
                Nresp += 1
                SIs.append(SI)

        summary_fig2(Nresp, data.iscell.sum(), SIs, label='Orient. Select. Index')
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()

    print('[ok] orientation selectivity analysis saved as: "%s" ' % pdf_filename)
    
def DS_analysis_pdf(datafile, iprotocol=0, Nmax=1000000):

    data = MultimodalData(datafile)

    pdf_filename = os.path.join(summary_pdf_folder(datafile), '%s-direction_selectivity.pdf' % data.protocols[iprotocol])
    
    Nresp, SIs = 0, []
    with PdfPages(pdf_filename) as pdf:

        for roi in np.arange(data.iscell.sum())[:Nmax]:
            
            print('   - direction-selectivity analysis for ROI #%i / %i' % (roi+1, data.iscell.sum()))
            fig, SI, responsive = DS_ROI_analysis(data, roiIndex=roi, iprotocol=iprotocol)
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()

            if responsive:
                Nresp += 1
                SIs.append(SI)

        summary_fig2(Nresp, data.iscell.sum(), SIs, label='Direct. Select. Index')
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()

    print('[ok] direction selectivity analysis saved as: "%s" ' % pdf_filename)

def C_ROI_analysis(FullData,
                 roiIndex=0,
                 iprotocol = 0,
                 subprotocol_id=0,
                 verbose=False,
                 response_significance_threshold=0.05,
                 with_responsive_angles = False,
                 stat_test_props=dict(interval_pre=[-2,0], interval_post=[2,4],
                                      test='ttest', positive=True)):
    """
    orientation selectivity ROI analysis
    """

    EPISODES = EpisodeResponse(FullData,
                               protocol_id=iprotocol,
                               quantity='CaImaging', subquantity='dF/F',
                               prestim_duration=-stat_test_props['interval_pre'][0],
                               roiIndex = roiIndex)

    fig, AX = FullData.plot_trial_average(EPISODES=EPISODES,
                                          protocol_id=iprotocol,
                                          quantity='CaImaging', subquantity='dF/F',
                                          roiIndex = roiIndex,
                                          column_key='angle',
                                          row_key='contrast',
                                          color_keys=['spatial-freq', 'speed'],
                                          ybar=1., ybarlabel='1dF/F',
                                          xbar=1., xbarlabel='1s',
                                          fig_preset='raw-traces-preset',
                                          with_annotation=True,
                                          with_stat_test=True, stat_test_props=stat_test_props,
                                          with_screen_inset=True,
                                          verbose=verbose)

    cell_resp = EPISODES.compute_summary_data(stat_test_props,
                                              response_significance_threshold=response_significance_threshold)
    
    return fig, cell_resp


def C_analysis_pdf(datafile, iprotocol=0, Nmax=1000000):

    data = MultimodalData(datafile)

    if not os.path.isdir(summary_pdf_folder(datafile)):
        os.mkdir(summary_pdf_folder(datafile))
    
    pdf_filename = os.path.join(summary_pdf_folder(datafile), '%s-gratings-analysis.pdf' % data.protocols[iprotocol])
    
    CELL_RESPS = []
    with PdfPages(pdf_filename) as pdf:

        for roi in np.arange(data.iscell.sum())[:Nmax]:

            fig, cell_resp = C_ROI_analysis(data, roiIndex=roi, iprotocol=iprotocol)
            CELL_RESPS.append(cell_resp)
            
            pdf.savefig(fig)  # saves the current figure into a pdf page
            plt.close(fig)

        fig = summary_fig(CELL_RESPS)
        
        pdf.savefig(fig)
        plt.close(fig)

    print('[ok] moving dot analysis saved as: "%s" ' % pdf_filename)
    

if __name__=='__main__':
    
    import argparse

    parser=argparse.ArgumentParser()
    parser.add_argument("datafile", type=str)
    parser.add_argument("analysis", type=str, help='should be either "orientation"/"direction"')
    parser.add_argument("--iprotocol", type=int, default=0, help='index for the protocol in case of multiprotocol in datafile')
    parser.add_argument('-nmax', "--Nmax", type=int, default=1000000)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    if '.nwb' in args.datafile:
        if args.analysis=='orientation':
            OS_analysis_pdf(args.datafile, iprotocol=args.iprotocol, Nmax=args.Nmax)
        elif args.analysis=='direction':
            DS_analysis_pdf(args.datafile, iprotocol=args.iprotocol, Nmax=args.Nmax)
        elif args.analysis=='gratings':
            C_analysis_pdf(args.datafile, iprotocol=args.iprotocol, Nmax=args.Nmax)
        else:
            print('need to choose either "direction"/"orientation" as an analysis type')
    else:
        print('/!\ Need to provide a NWB datafile as argument ')








