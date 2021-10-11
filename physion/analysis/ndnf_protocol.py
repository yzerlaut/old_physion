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
                 response_significance_threshold=0.01,
                 with_responsive_angles = False,
                 stat_test_props=dict(interval_pre=[-2,0], interval_post=[1,3],
                                      test='wilcoxon', positive=True),
                 options = dict(quantity='CaImaging', subquantity='d(F-0.7*Fneu)',
                                dt_sampling=1, prestim_duration=2, 
                                baseline_substraction=True)):
    """
    direction selectivity ROI analysis
    """
    pattern = [[2,1],[2,1],[2,1],[2,1], # patches
               [1,1], # -- space --
               [3,1], # looming
               [1,1], # -- space --
               [2,1],[2,1],[2,1],[2,1], # patches
               [1,1], # -- space --
               [2,1],[2,1],[2,1],[2,1]]

    fig, AX = ge.figure(axes_extents=[pattern], figsize=(.5,1.), wspace=0.4, top=2., reshape_axes=False)
    for iax in [4,6,11]:
        AX[0][iax].axis('off')

    ge.annotate(AX[0][1], 'static-patches\n\n', (1,1), ha='center')
    ge.annotate(AX[0][5], 'looming-stim\n\n', (.5,1), ha='center')
    ge.annotate(AX[0][8], 'drifting-gratings\n\n', (1,1), ha='center')
    ge.annotate(AX[0][13], 'moving-dots\n\n', (1,1), ha='center')

    plot_options = dict(with_std=True,
                        with_stat_test=True, stat_test_props=dict(interval_pre=[-2,0], interval_post=[2,4],
                                                                  test='ttest', positive=True),
                        with_annotation=True,
                        fig=fig)

    # static patches
    # -------------------
    EPISODES = EpisodeResponse(FullData,
                               protocol_id=0, roiIndex=roiIndex, **options)

    FullData.plot_trial_average(EPISODES=EPISODES, 
                            protocol_id=0,
                            column_key='angle', AX=[AX[0][:4]], **plot_options)

    # looming stim
    # -------------------
    EPISODES = EpisodeResponse(FullData,
                               protocol_id=1, roiIndex=roiIndex, **options)
    FullData.plot_trial_average(EPISODES=EPISODES,
                            protocol_id=1,
                            AX=[[AX[0][5]]], **plot_options)


    # drifting gratings
    # -------------------
    EPISODES = EpisodeResponse(FullData,
                               protocol_id=2, roiIndex=roiIndex, **options)
    FullData.plot_trial_average(EPISODES=EPISODES, 
                            protocol_id=2,
                            column_key='angle', AX=[AX[0][7:12]], **plot_options)

    # moving dots
    # -------------------
    EPISODES = EpisodeResponse(FullData,
                               protocol_id=3, roiIndex=roiIndex, **options)
    FullData.plot_trial_average(EPISODES=EPISODES, 
                            protocol_id=3,
                            column_key='direction', AX=[AX[0][12:16]], **plot_options)

    ylim = [np.min([ax.get_ylim()[0] for ax in AX[0]]), np.max([ax.get_ylim()[1] for ax in AX[0]])]
    for ax in AX[0]:
        ax.set_ylim(ylim)
        ax.axis('off')

    for iax in [0,5,7,12]:
        AX[0][iax].axis('off')
        ge.draw_bar_scales(AX[0][iax], Xbar=1, Xbar_label='1s', Ybar=1,  Ybar_label='1dF/F',
                           loc='top-left')

    return fig


def analysis_pdf(datafile, Nmax=1000000):

    data = MultimodalData(datafile)

    pdf_filename = os.path.join(summary_pdf_folder(datafile), 'ndnf_protocol.pdf')
    
    curves, results = [], {'Ntot':data.iscell.sum()}
    with PdfPages(pdf_filename) as pdf:

        for roi in np.arange(data.iscell.sum())[:Nmax]:
            
            print('   - surround-suppression analysis for ROI #%i / %i' % (roi+1, data.iscell.sum()))
            fig = ROI_analysis(data, roiIndex=roi)
            pdf.savefig(fig)
            plt.close(fig)

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

    print('[ok] ndnf protocol analysis saved as: "%s" ' % pdf_filename)

# def summary_fig(radii, curves, Ntot):

#     fig, ax = ge.figure(right=8, figsize=(1.5, 1.5))
#     ge.title(ax, 'n=%i ROIS' % len(curves))

#     axi = ge.inset(fig, [.73,.35,.3,.4])
#     axi.pie([100*len(curves)/Ntot, 100*(1-len(curves)/Ntot)], explode=(0, 0.1),
#             colors=[plt.cm.tab10(2), plt.cm.tab10(3)],
#             labels=['responsive at low radius\n\n\n', ''],
#             autopct='%1.1f%%', shadow=True, startangle=90)
#     axi.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
#     curves = np.array(curves)
#     ge.scatter(x=radii, y=np.mean(curves, axis=0), sy=np.std(curves, axis=0), no_set=True, ms=5, ax=ax, lw=2)
#     ge.set_plot(ax, xlabel='size ($^{o}$)', ylabel='$\delta$ dF/F')
#     return fig

if __name__=='__main__':
    
    import argparse

    parser=argparse.ArgumentParser()
    parser.add_argument("datafile", type=str)
    parser.add_argument('-nmax', "--Nmax", type=int, default=1000000)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    
    if '.nwb' in args.datafile:
        analysis_pdf(args.datafile, Nmax=args.Nmax)
    else:
        print('/!\ Need to provide a NWB datafile as argument ')








