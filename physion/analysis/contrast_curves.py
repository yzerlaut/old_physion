import sys, os, pathlib
import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from datavyz import graph_env_manuscript as ge

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from dataviz.show_data import MultimodalData, format_key_value
from analysis.tools import summary_pdf_folder
from analysis.process_NWB import EpisodeResponse
from analysis import stat_tools


def ROI_analysis(FullData,
                 roiIndex=0,
                 iprotocol=0,
                 verbose=False,
                 response_significance_threshold=0.05,
                 with_responsive_angles = False,
                 stat_test_props=dict(interval_pre=[-2,0], interval_post=[1,3],
                                      test='ttest', positive=True),
                 CaImaging_options = dict(quantity='CaImaging', subquantity='dF/F'),
                 log_spaced=False,
                 Npanels=4):
    """
    direction selectivity ROI analysis
    """

    EPISODES = EpisodeResponse(FullData,
                               protocol_id=iprotocol,
                               prestim_duration=-stat_test_props['interval_pre'][0],
                               roiIndex = roiIndex,
                               **CaImaging_options)

    fig, AX = FullData.plot_trial_average(EPISODES=EPISODES,
                                          protocol_id=iprotocol,
                                          roiIndex = roiIndex,
                                          column_key='contrast', row_key='angle',
                                          ybar=1., ybarlabel='1dF/F',
                                          xbar=1., xbarlabel='1s',
                                          fig_preset='raw-traces-preset+right-space',
                                          with_annotation=True,
                                          with_std=True,
                                          with_stat_test=True, stat_test_props=stat_test_props,
                                          verbose=verbose,
                                          **CaImaging_options)

    AXI, ylims, CURVES = [], [10, -10], []
    max_response_curve, imax_response_curve, max_angle = np.zeros(len(EPISODES.varied_parameters['contrast'])+1), -1, -1
    for ia, angle in enumerate(EPISODES.varied_parameters['angle']):

        resp, contrasts, significants = [0], [0], [False]
        for ic, contrast in enumerate(EPISODES.varied_parameters['contrast']):

            # stat test "pre" vs "post"
            stats = EPISODES.stat_test_for_evoked_responses(episode_cond=EPISODES.find_episode_cond(['angle', 'contrast'], [ia, ic]),
                                                            **stat_test_props)
            
            resp.append(np.mean(stats.y-stats.x)) # "post"-"pre"
            contrasts.append(contrast)
            significants.append(stats.significant(threshold=response_significance_threshold))

        AXI.append(AX[ia][-1].inset_axes([1.8, .2, .7, .6]))
        ge.plot(contrasts, resp, ax=AXI[-1], no_set=True, ms=3, m='o')
        ylims = [np.min([np.min(resp), ylims[0]]), np.max([np.max(resp), ylims[1]])]

        if (np.sum(significants)>0) and np.max(resp)>np.max(max_response_curve):
            imax_response_curve = ia
            max_response_curve = np.array(resp)
            max_angle = angle
            
    for ia, axi in enumerate(AXI):
        ge.set_plot(axi, xlabel='contrast', ylabel='$\delta$ dF/F',
                    xscale=('log' if log_spaced else 'linear'),
                    ylim=[ylims[0]-.05*(ylims[1]-ylims[0]),ylims[1]+.05*(ylims[1]-ylims[0])])
        if ia==imax_response_curve:
            axi.fill_between(contrasts, ylims[0]*np.ones(len(contrasts)), ylims[1]*np.ones(len(contrasts)),
                             color='k', alpha=0.1, lw=0)

    cr = EPISODES.compute_summary_data(stat_test_props,
                                       response_significance_threshold=response_significance_threshold)
            
    return fig, contrasts, max_response_curve, max_angle, cr

def mergedROI_analysis(FullData,
                       iprotocol=0,
                       stat_test_props=dict(interval_pre=[-2,0], interval_post=[1,3], test='ttest', positive=True),
                       CaImaging_options = dict(quantity='CaImaging', subquantity='dF/F'),
                       verbose=True,
                       nmax=1000000):
    
    figS, AXs = ge.figure(axes=(5,1), figsize=(1,1.2), bottom=.7, wspace=0.6, right=14, reshape_axes=False) # summary fig
    inset = ge.inset(figS, (0.87,0.4,0.13,0.33))

    roiIndices = np.arange(FullData.iscell.sum())[:nmax]
    EPISODES = EpisodeResponse(FullData,
                               protocol_id=iprotocol,
                               prestim_duration=-stat_test_props['interval_pre'][0],
                               roiIndices = roiIndices,
                               **CaImaging_options)

    
    FullData.plot_trial_average(EPISODES=EPISODES,
                                protocol_id=iprotocol,
                                roiIndex = roiIndices,
                                column_key='contrast',
                                ybar=0.2, ybarlabel='0.2dF/F',
                                xbar=1., xbarlabel='1s',
                                fig=figS, AX=AXs, no_set=False,
                                with_annotation=True,
                                with_std=True,
                                with_stat_test=True, stat_test_props=stat_test_props,
                                verbose=verbose,
                                **CaImaging_options)

    pop_data = EPISODES.compute_summary_data(stat_test_props,
                                             exclude_keys=['repeat', 'angle'],
                                             response_significance_threshold=0.01)

    ge.scatter(pop_data['contrast'], pop_data['value'], lw=1, ax=inset, no_set=True)
    ge.title(inset, 'n=%i rois' % len(roiIndices), size='small')

    ge.set_plot(inset, xlabel='contrast', xscale='log', ylabel='evoked dF/F     ')

    ge.annotate(figS, 'grand average', (0.5,0), ha='center')
    return figS, AXs

def Ephys_analysis(FullData,
                   iprotocol=0,
                   verbose=False,
                   response_significance_threshold=0.05,
                   with_responsive_angles = False,
                   stat_test_props=dict(interval_pre=[-.2,0], interval_post=[0.1,0.3],
                                        test='ttest', positive=False),
                   Npanels=4):
    """
    response plots
    """

    EPISODES = EpisodeResponse(FullData,
                               protocol_id=iprotocol,
                               quantity='Electrophysiological-Signal',
                               prestim_duration=-stat_test_props['interval_pre'][0],
                               baseline_substraction=True)

    fig, AX = FullData.plot_trial_average(EPISODES=EPISODES,
                                          protocol_id=iprotocol,
                                          column_key='contrast', row_key='angle',
                                          fig_preset='raw-traces-preset+right-space',
                                          with_annotation=True,
                                          ybar=.1, ybarlabel='100uV',
                                          xbar=.1, xbarlabel='100ms',
                                          with_std=True,
                                          with_stat_test=True, stat_test_props=stat_test_props,
                                          verbose=verbose)

    fig2, AX2 = FullData.plot_trial_average(EPISODES=EPISODES,
                                          protocol_id=iprotocol,
                                          column_key='contrast',
                                          fig_preset='raw-traces-preset+right-space',
                                          with_annotation=True,
                                          ybar=.1, ybarlabel='100uV',
                                          xbar=.1, xbarlabel='100ms',
                                          with_std=False,
                                          with_stat_test=True, stat_test_props=stat_test_props,
                                          verbose=verbose)
    
    # AXI, ylims, CURVES = [], [10, -10], []
    # max_response_curve, imax_response_curve = np.zeros(len(EPISODES.varied_parameters['contrast'])+1), -1
    # for ia, angle in enumerate(EPISODES.varied_parameters['angle']):

    #     resp, contrasts, significants = [0], [0], [False]
    #     for ic, contrast in enumerate(EPISODES.varied_parameters['contrast']):

    #         # stat test "pre" vs "post"
    #         stats = EPISODES.stat_test_for_evoked_responses(episode_cond=EPISODES.find_episode_cond(['angle', 'contrast'], [ia, ic]),
    #                                                         **stat_test_props)
            
    #         resp.append(np.mean(stats.y-stats.x)) # "post"-"pre"
    #         contrasts.append(contrast)
    #         significants.append(stats.significant(threshold=response_significance_threshold))

    #     AXI.append(AX[ia][-1].inset_axes([1.8, .2, .7, .6]))
    #     ge.plot(contrasts, resp, ax=AXI[-1], no_set=True, ms=3, m='o')
    #     ylims = [np.min([np.min(resp), ylims[0]]), np.max([np.max(resp), ylims[1]])]

    #     if (np.sum(significants)>0) and np.max(resp)>np.max(max_response_curve):
    #         imax_response_curve = ia
    #         max_response_curve = np.array(resp)

    # for ia ,axi in enumerate(AXI):
    #     ge.set_plot(axi, xlabel='contrast', ylabel='$\delta$ dF/F',
    #                 ylim=[ylims[0]-.05*(ylims[1]-ylims[0]),ylims[1]+.05*(ylims[1]-ylims[0])])
    #     if ia==imax_response_curve:
    #         axi.fill_between(contrasts, ylims[0]*np.ones(len(contrasts)), ylims[1]*np.ones(len(contrasts)), color='k', alpha=0.1)

    # return fig, contrasts, max_response_curve
    return fig, fig2

def summary_fig(contrasts, CURVES, results,
                log_spaced=False):

    fig, AX = ge.figure(axes=(5,1), figsize=(1., 1.))

    AX[0].axis('off')

    if len(CURVES)>1:
        ge.plot(contrasts, np.mean(np.array(CURVES), axis=0),
                sy=np.std(np.array(CURVES), axis=0), ax=AX[1], no_set=True)
    else:
        AX[1].axis('off')
    ge.set_plot(AX[1], xlabel='contrast', ylabel='$\delta$ dF/F',
                xscale=('log' if log_spaced else 'linear'))
    
    AX[2].axis('off')
        
    ge.annotate(AX[2], '\nn=%i resp. cells (/%i)\n(p<%.2f)' % (len(CURVES), results['Ntot'], results['pthresh']), (0.5,.0), ha='center', va='top', size='small')

    frac_resp = len(CURVES)/results['Ntot']
    data = np.array([100*frac_resp, 100*(1-frac_resp)])
    ge.pie(data,
           COLORS=[plt.cm.tab10(2), plt.cm.tab10(3)],
           pie_labels = ['  %.1f%%' % (100*d/data.sum()) for d in data],
           ext_labels = ['', ''],
           ax=AX[2])

    for ax, key in zip(AX[3:], ['angle', 'contrast']):
        ge.hist(results[key], bins=results['%s-bins' % key], ax=ax)
        ge.set_plot(ax, xlabel='pref. %s' % key, ylabel='count')
    
    return fig

def analysis_pdf(datafile, iprotocol=0, Nmax=1000000, response_significance_threshold=0.05):

    data = MultimodalData(datafile)

    pdf_filename = os.path.join(summary_pdf_folder(datafile), '%s-contrast_curves.pdf' % data.protocols[iprotocol])
    
    if data.metadata['CaImaging']:
        
        results = {'Ntot':data.iscell.sum(), 'angle':[], 'contrast':[], 'pthresh':response_significance_threshold}
    
        with PdfPages(pdf_filename) as pdf:

            CURVES = []
            for roi in np.arange(data.iscell.sum())[:Nmax]:
                print('   - contrast-curves analysis for ROI #%i / %i' % (roi+1, data.iscell.sum()))
                fig, contrasts, max_response_curve, max_angle, cr = ROI_analysis(data, roiIndex=roi, iprotocol=iprotocol,
                                                                                 log_spaced=('log-spaced' in pdf_filename),
                                                                                 response_significance_threshold=response_significance_threshold)
                pdf.savefig(fig)
                plt.close(fig)
                if np.max(max_response_curve)>0:
                    CURVES.append(max_response_curve)
                    results['angle'].append(max_angle)
                    results['contrast'].append(contrasts[np.argmax(max_response_curve)])
                    for key in ['angle-bins', 'contrast-bins']:
                        results[key] = cr[key]

            #
            fig, AX = mergedROI_analysis(data,
                                         iprotocol=iprotocol,
                                         nmax=Nmax)
            pdf.savefig(fig)
            plt.close(fig)
            
            #
            fig = summary_fig(contrasts, CURVES, results,
                              log_spaced=('log-spaced' in pdf_filename))
            pdf.savefig(fig)
            plt.close(fig)

    elif data.metadata['Electrophy']:
        with PdfPages(pdf_filename) as pdf:
            fig, fig2 = Ephys_analysis(data, iprotocol=iprotocol)
            pdf.savefig(fig)
            plt.close(fig)
            pdf.savefig(fig2)
            plt.close(fig2)
            

    print('[ok] contrast-curves analysis saved as: "%s" ' % pdf_filename)


if __name__=='__main__':
    
    import argparse

    parser=argparse.ArgumentParser()
    parser.add_argument("datafile", type=str)
    parser.add_argument("--iprotocol", type=int, default=0, help='index for the protocol in case of multiprotocol in datafile')
    parser.add_argument('-rst', "--response_significance_threshold", type=float, default=0.01)
    parser.add_argument('-nmax', "--Nmax", type=int, default=1000000)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    if '.nwb' in args.datafile:
        analysis_pdf(args.datafile,
                     iprotocol=args.iprotocol,
                     Nmax=args.Nmax,
                     response_significance_threshold=args.response_significance_threshold)
    else:
        print('/!\ Need to provide a NWB datafile as argument ')
        


    
