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
from analysis.spatial_selectivity import summary_fig as SS_summary_fig

def ROI_analysis(FullData,
                 roiIndex=0,
                 iprotocol_center=0,
                 iprotocol_surround=1,
                 verbose=False,
                 response_significance_threshold=0.01,
                 radius_threshold_for_center=20.,
                 with_responsive_angles = False,
                 stat_test_props=dict(interval_pre=[-2,0], interval_post=[1,3],
                                      test='wilcoxon', positive=True),
                 Npanels=4):
    """
    ROI analysis for secondary RF
    """

    EPISODES_CENTER = EpisodeResponse(FullData,
                                      protocol_id=iprotocol_center,
                                      quantity='CaImaging', subquantity='dF/F',
                                      prestim_duration=2.,
                                      roiIndex = roiIndex)

    EPISODES_SURROUND = EpisodeResponse(FullData,
                                        protocol_id=iprotocol_surround,
                                        quantity='CaImaging', subquantity='dF/F',
                                        prestim_duration=2.,
                                        roiIndex = roiIndex)
    

    FIGS = []
    
    for iangle, angle in enumerate(EPISODES_CENTER.varied_parameters['angle']):
    
        fig, AX = FullData.plot_trial_average(EPISODES=EPISODES_CENTER,
                                              condition=EPISODES_CENTER.find_episode_cond('angle', iangle),
                                              protocol_id=iprotocol_center,
                                              quantity='CaImaging', subquantity='dF/F',
                                              roiIndex = roiIndex,
                                              column_key='x-center', row_key='y-center',
                                              ybar=1., ybarlabel='1dF/F',
                                              xbar=1., xbarlabel='1s',
                                              fig_preset='raw-traces-preset',
                                              with_std=False,
                                              with_annotation=True,
                                              verbose=verbose, color=ge.blue)
        ge.annotate(fig, 'center ', (0.5, 0), color=ge.blue, ha='right')

        FullData.plot_trial_average(EPISODES=EPISODES_SURROUND,
                                    condition=EPISODES_SURROUND.find_episode_cond('angle', iangle),
                                    protocol_id=iprotocol_surround,
                                    quantity='CaImaging', subquantity='dF/F',
                                    roiIndex = roiIndex,
                                    column_key='x-center', row_key='y-center',
                                    with_std=False, with_stim=False,
                                    verbose=verbose, AX=AX, color=ge.red)
        ge.annotate(fig, ' surround', (0.5, 0), color=ge.red)
        ge.annotate(fig, format_key_value('angle', angle)+'  ', (1., 0), ha='right')

        # just for the axis extent
        _, AX2 = FullData.plot_trial_average(EPISODES=EPISODES_SURROUND,
                                    condition=EPISODES_SURROUND.find_episode_cond('angle', iangle),
                                    protocol_id=iprotocol_surround,
                                    quantity='CaImaging', subquantity='dF/F',
                                    roiIndex = roiIndex,
                                    column_key='x-center', row_key='y-center')
        
        ylims = [0,1]
        for ax, ax2 in zip(ge.flat(AX), ge.flat(AX2)):
            ylims = [np.min([ylims[0], ax.get_ylim()[0], ax2.get_ylim()[0]]),
                     np.max([ylims[1], ax.get_ylim()[1], ax2.get_ylim()[1]])]
        for ax in ge.flat(AX):
            ax.set_ylim(ylims)

        FIGS.append(fig)

    # fig2, AX = ge.figure(axes=(Npanels, int(Nfigs/Npanels)), hspace=1.5)

    # full_resp = {'value':[], 'significant':[], 'radius':[]}
    # for key, bins in zip(ROW_KEYS, ROW_BINS):
    #     full_resp[key] = []
    #     full_resp[key+'-bins'] = bins
    
    # iax, ylims = 0, [10, -10]
    # max_response_level, max_response_curve, imax_response_curve = 0, None, -1
    # for indices in itertools.product(*ROW_INDICES):

    #     title = ''
    #     for key, index in zip(ROW_KEYS, indices):
    #         title+=format_key_value(key, EPISODES.varied_parameters[key][index])+', ' # should have a unique value
    #     ge.title(ge.flat(AX)[iax], title, size='small')
        
    #     resp, radii, significants = [0], [0], [False]
    #     for ia, radius in enumerate(EPISODES.varied_parameters['radius']):
            
    #         # stat test "pre" vs "post"
    #         stats = EPISODES.stat_test_for_evoked_responses(episode_cond=EPISODES.find_episode_cond(ROW_KEYS+['radius'], list(indices)+[ia]),
    #                                                         **stat_test_props)
            
    #         resp.append(np.mean(stats.y-stats.x)) # "post"-"pre"
    #         radii.append(radius)
            
    #         ge.annotate(ge.flat(AX)[iax], '    '+stats.pval_annot()[0], (radius, resp[-1]),
    #                     size='x-small', rotation=90, ha='center', xycoords='data')

    #         significants.append(stats.significant(threshold=response_significance_threshold))

    #         # store all in full resp
    #         for key, index in zip(ROW_KEYS, indices):
    #             full_resp[key].append(EPISODES.varied_parameters[key][index])
    #         full_resp['radius'].append(radius)
    #         full_resp['value'].append(np.mean(stats.y-stats.x))
    #         full_resp['significant'].append(stats.significant(threshold=response_significance_threshold))

    #     # check if max resp
    #     center_cond = np.array(significants) & (np.array(radii)<radius_threshold_for_center)
    #     if np.sum(center_cond) and (np.max(np.array(resp)[center_cond])>max_response_level):
    #         max_response_level = np.max(np.array(resp)[center_cond])
    #         max_response_curve = np.array(resp)
    #         imax_response_curve = iax
            
    #     ylims = [np.min([np.min(resp), ylims[0]]), np.max([np.max(resp), ylims[1]])]
    #     ge.flat(AX)[iax].plot(radii, resp, 'ko-', ms=4)
    #     iax += 1
        
    # for iax, ax in enumerate(ge.flat(AX)):
    #     ge.set_plot(ax, ylim=[ylims[0]-.05*(ylims[1]-ylims[0]), ylims[1]+.05*(ylims[1]-ylims[0])],
    #                 ylabel=('$\delta$ dF/F' if (iax%Npanels==0) else ''),
    #                 xlabel=('size ($^{o}$)' if (int(iax/Npanels)==(int(Nfigs/Npanels)-1)) else ''))
    #     ax.fill_between([0, radius_threshold_for_center], ylims[0]*np.ones(2), ylims[1]*np.ones(2), color='k', alpha=0.05, lw=0)
        
    #     if iax==imax_response_curve:
    #         ax.fill_between(radii, ylims[0]*np.ones(len(radii)), ylims[1]*np.ones(len(radii)), color='k', alpha=0.1, lw=0)
            
    # x = np.unique(EPISODES.varied_parameters['radius'])
    # full_resp['radius-bins'] = np.concatenate([[x[0]-.5*(x[1]-x[0])], .5*(x[1:]+x[:-1]), [x[-1]+.5*(x[-1]-x[-2])]])

    # for key in full_resp:
    #     full_resp[key] = np.array(full_resp[key])
    
    # return fig, fig2, radii, max_response_curve, imax_response_curve, full_resp
    # return fig, None, None, None, None, None
    return FIGS


def analysis_pdf(datafile, iprotocol=0, Nmax=1000000):

    data = MultimodalData(datafile)

    pdf_filename = os.path.join(summary_pdf_folder(datafile), '%s-secondary_RF.pdf' % data.protocols[iprotocol])

    # find center and surround
    icenter = np.argwhere([('center' in p) for p in data.protocols])[0][0]
    isurround = np.argwhere([('surround' in p) for p in data.protocols])[0][0]
    
    curves, results = [], {'Ntot':data.iscell.sum()}
    with PdfPages(pdf_filename) as pdf:

        for roi in np.arange(data.iscell.sum())[:Nmax]:

            print('   - secondary-RF analysis for ROI #%i / %i' % (roi+1, data.iscell.sum()))
            # fig, fig2, radii, max_response_curve, imax_response_curve, full_resp = ROI_analysis(data, roiIndex=roi,
            #                                                                                     iprotocol_center=icenter,
            #                                                                                     iprotocol_surround=isurround)
            FIGS = ROI_analysis(data, roiIndex=roi,
                                iprotocol_center=icenter,
                                iprotocol_surround=isurround)
            for fig in FIGS:
                pdf.savefig(fig)
                plt.close(fig)
            
        #     pdf.savefig(fig2)
        #     plt.close(fig2)

        #     if max_response_curve is not None:
        #         curves.append(max_response_curve)

        #     # initialize if not done
        #     for key in full_resp:
        #         if ('-bins' in key) and (key not in results):
        #             results[key] = full_resp[key]
        #         elif (key not in results):
        #             results[key] = []
                    
        #     significant_cond = (full_resp['significant']==True)
        #     if np.sum(significant_cond)>0:
        #         imax = np.argmax(full_resp['value'][significant_cond])
        #         for key in full_resp:
        #             if ('-bins' not in key):
        #                 results[key].append(full_resp[key][significant_cond][imax])

        # if len(results['value'])>0:
            
        #     fig = SS_summary_fig(results)
        #     pdf.savefig(fig)
        #     plt.close(fig)

        #     fig = summary_fig(radii, curves, data.iscell.sum())
        #     pdf.savefig(fig)
        #     plt.close(fig)

    print('[ok] secondary RF analysis saved as: "%s" ' % pdf_filename)

def summary_fig(radii, curves, Ntot):

    fig, ax = ge.figure(right=8, figsize=(1.5, 1.5))
    ge.title(ax, 'n=%i ROIS' % len(curves))

    axi = ge.inset(fig, [.73,.35,.3,.4])
    axi.pie([100*len(curves)/Ntot, 100*(1-len(curves)/Ntot)], explode=(0, 0.1),
            colors=[plt.cm.tab10(2), plt.cm.tab10(3)],
            labels=['responsive at low radius\n\n\n', ''],
            autopct='%1.1f%%', shadow=True, startangle=90)
    axi.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
    curves = np.array(curves)
    ge.scatter(x=radii, y=np.mean(curves, axis=0), sy=np.std(curves, axis=0), no_set=True, ms=5, ax=ax, lw=2)
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
        analysis_pdf(args.datafile, Nmax=args.Nmax)
    else:
        print('/!\ Need to provide a NWB datafile as argument ')








