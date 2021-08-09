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
                 response_significance_threshold=0.01,
                 radius_threshold_for_center=20.,
                 with_responsive_angles = False,
                 stat_test_props=dict(interval_pre=[-2,0], interval_post=[1,3],
                                      test='wilcoxon'),
                 Npanels=4):
    """
    direction selectivity ROI analysis
    """

    EPISODES = EpisodeResponse(FullData,
                               protocol_id=iprotocol,
                               quantity='CaImaging', subquantity='dF/F',
                               roiIndex = roiIndex)


    ROW_KEYS, ROW_VALUES, ROW_INDICES, Nfigs = [], [], [], 1
    for key in ['x-center', 'y-center', 'contrast']:
        if key in EPISODES.varied_parameters:
            ROW_KEYS.append(key)
            ROW_VALUES.append(EPISODES.varied_parameters[key])
            ROW_INDICES.append(np.arange(len(EPISODES.varied_parameters[key])))
            Nfigs *= len(EPISODES.varied_parameters[key])
            
    fig, AX = FullData.plot_trial_average(EPISODES=EPISODES,
                                          protocol_id=iprotocol,
                                          quantity='CaImaging', subquantity='dF/F',
                                          roiIndex = roiIndex,
                                          column_key='radius', row_keys=ROW_KEYS, color_key='angle',
                                          ybar=1., ybarlabel='1dF/F',
                                          xbar=1., xbarlabel='1s',
                                          fig_preset='raw-traces-preset',
                                          with_annotation=True,
                                          with_std=False,
                                          with_stat_test=True, stat_test_props=stat_test_props,
                                          verbose=verbose)

    # now computing the size-response curve for all conditions
    if 'angle' in EPISODES.varied_parameters:
        ROW_KEYS.append('angle')
        ROW_VALUES.append(EPISODES.varied_parameters['angle'])
        ROW_INDICES.append(np.arange(len(EPISODES.varied_parameters['angle'])))
        Nfigs *= len(EPISODES.varied_parameters['angle'])


    fig2, AX = ge.figure(axes=(Npanels, int(Nfigs/Npanels)), hspace=1.5)

    iax, ylims = 0, [10, -10]
    max_response_level, max_response_curve, imax_response_curve = 0, None, -1
    for indices in itertools.product(*ROW_INDICES):

        title = ''
        for key, index in zip(ROW_KEYS, indices):
            title+=format_key_value(key, EPISODES.varied_parameters[key][index])+', ' # should have a unique value
        ge.title(ge.flat(AX)[iax], title, size='small')
        
        resp, radii, significants = [0], [0], [False]
        for ia, radius in enumerate(EPISODES.varied_parameters['radius']):
            means_pre = EPISODES.compute_stats_over_repeated_trials(ROW_KEYS+['radius'], list(indices)+[ia],
                                                                    interval_cond=EPISODES.compute_interval_cond(stat_test_props['interval_pre']),
                                                                    quantity='mean')
            means_post = EPISODES.compute_stats_over_repeated_trials(ROW_KEYS+['radius'], list(indices)+[ia],
                                                                    interval_cond=EPISODES.compute_interval_cond(stat_test_props['interval_post']),
                                                                    quantity='mean')
            resp.append(np.mean(means_post)-np.mean(means_pre))
            radii.append(radius)

            stats = EPISODES.stat_test_for_evoked_responses(episode_cond=EPISODES.find_episode_cond(ROW_KEYS+['radius'], list(indices)+[ia]),
                                                            **stat_test_props)
            ge.annotate(ge.flat(AX)[iax], '    '+stats.pval_annot()[0], (radius, resp[-1]), size='x-small', rotation=90, ha='center', xycoords='data')

            significants.append(stats.significant(threshold=response_significance_threshold))

        # check if max resp
        center_cond = np.array(significants) & (np.array(radii)<radius_threshold_for_center)
        if np.sum(center_cond) and (np.max(np.array(resp)[center_cond])>max_response_level):
            max_response_level = np.max(np.array(resp)[center_cond])
            max_response_curve = np.array(resp)
            imax_response_curve = iax
            
        ylims = [np.min([np.min(resp), ylims[0]]), np.max([np.max(resp), ylims[1]])]
        ge.flat(AX)[iax].plot(radii, resp, 'ko-', ms=4)
        iax += 1
        
    for iax, ax in enumerate(ge.flat(AX)):
        ge.set_plot(ax, ylim=[ylims[0]-.05*(ylims[1]-ylims[0]), ylims[1]+.05*(ylims[1]-ylims[0])],
                    ylabel=('$\delta$ dF/F' if (iax%Npanels==0) else ''),
                    xlabel=('size ($^{o}$)' if (int(iax/Npanels)==(int(Nfigs/Npanels)-1)) else ''))
        ax.fill_between([0, radius_threshold_for_center], ylims[0]*np.ones(2), ylims[1]*np.ones(2), color='k', alpha=0.05, lw=0)
        
        if iax==imax_response_curve:
            ax.fill_between(radii, ylims[0]*np.ones(len(radii)), ylims[1]*np.ones(len(radii)), color='k', alpha=0.1, lw=0)
            
    
    return fig, fig2, radii, max_response_curve, imax_response_curve


def analysis_pdf(datafile, iprotocol=0, Nmax=1000000):

    data = MultimodalData(datafile)

    pdf_filename = os.path.join(summary_pdf_folder(datafile), '%s-surround_suppression.pdf' % data.protocols[iprotocol])
    
    curves = []
    with PdfPages(pdf_filename) as pdf:

        for roi in np.arange(data.iscell.sum())[:Nmax]:
            
            print('   - surround-suppression analysis for ROI #%i / %i' % (roi+1, data.iscell.sum()))
            fig, fig2, radii, max_response_curve, imax_response_curve = ROI_analysis(data, roiIndex=roi, iprotocol=iprotocol)
            pdf.savefig(fig)
            pdf.savefig(fig2)
            plt.close(fig)
            plt.close(fig2)

            if max_response_curve is not None:
                curves.append(max_response_curve)

        fig = summary_fig(radii, curves, data.iscell.sum())
        pdf.savefig(fig)  # saves the current figure into a pdf page
        plt.close(fig)

    print('[ok] direction selectivity analysis saved as: "%s" ' % pdf_filename)

def summary_fig(radii, curves, Ntot):

    fig, ax = ge.figure(right=8, figsize=(1.5, 1.5))
    ge.title(ax, 'n=%i ROIS' % len(curves))

    axi = ge.inset(fig, [.7,.55,.3,.4])
    axi.pie([100*len(curves)/Ntot, 100*(1-len(curves)/Ntot)], explode=(0, 0.1),
            colors=[plt.cm.tab10(2), plt.cm.tab10(3)],
            labels=['responsive', ''], autopct='%1.1f%%', shadow=True, startangle=90)
    axi.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
    curves = np.array(curves)
    ge.plot(radii, np.mean(curves, axis=0), sy=np.std(curves, axis=0), no_set=True, ms=4, ax=ax)
    ge.set_plot(ax, xlabel='size ($^{o}$)', ylabel='$\delta$ dF/F')
    return fig

if __name__=='__main__':
    
    import argparse

    parser=argparse.ArgumentParser()
    parser.add_argument("datafile", type=str)
    parser.add_argument("--iprotocol", type=int, default=0, help='index for the protocol in case of multiprotocol in datafile')
    parser.add_argument('-nmax', "--Nmax", type=int, default=1000000)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    
    if '.nwb' in args.datafile:
        analysis_pdf(args.datafile, iprotocol=args.iprotocol, Nmax=args.Nmax)
    else:
        print('/!\ Need to provide a NWB datafile as argument ')








