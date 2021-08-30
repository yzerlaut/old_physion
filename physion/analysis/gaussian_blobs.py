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


def Ephys_analysis(FullData,
                   iprotocol=0,
                   verbose=False,
                   response_significance_threshold=0.05,
                   radius_threshold_for_center=20.,
                   with_responsive_angles = False,
                   stat_test_props=dict(interval_pre=[-1,0], interval_post=[0.1,0.3],
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
                                          row_key='repeat',
                                          column_key='center-time',
                                          color_key='radius',
                                          fig_preset='raw-traces-preset',
                                          with_annotation=True,
                                          ybar=.2, ybarlabel='200uV',
                                          xbar=1, xbarlabel='1s',
                                          with_std=True,
                                          # with_stat_test=True, stat_test_props=stat_test_props,
                                          verbose=verbose)

    fig2, AX2 = FullData.plot_trial_average(EPISODES=EPISODES,
                                          protocol_id=iprotocol,
                                          column_key='center-time',
                                          fig_preset='raw-traces-preset',
                                          with_annotation=True,
                                          xbar=1, xbarlabel='1s',
                                          ybar=.1, ybarlabel='100uV',
                                          with_std=False,
                                          # with_stat_test=True, stat_test_props=stat_test_props,
                                          verbose=verbose)
    return fig, fig2


def Ophys_analysis(FullData,
                   iprotocol=0,
                   roiIndices=[0,1],
                   verbose=False):
    """
    response plots
    """

    fig, AX = FullData.single_trial_rasters(protocol_id=iprotocol,
                                            # quantity='Photodiode-Signal',
                                            quantity='CaImaging', subquantity='dF/F',
                                            Nmax=3, dt_sampling=10)

    fig = None
    return fig


def analysis_pdf(datafile,
                 iprotocol=-1,
                 Nmax=2):

    data = MultimodalData(datafile)

    if iprotocol<0:
        iprotocol = np.argwhere([('gaussian-blobs' in p) for p in data.protocols])[0][0]
        print('gaussian-blob analysis for protocol #', iprotocol)
    
    pdf_filename = os.path.join(summary_pdf_folder(datafile), '%s-contrast_curves.pdf' % data.protocols[iprotocol])
    
    
    if data.metadata['CaImaging']:

        fig = Ophys_analysis(data,
                             roiIndices=np.arange(np.sum(data.iscell))[:Nmax])
        ge.show()
        # results = {'Ntot':data.iscell.sum()}
    
        # with PdfPages(pdf_filename) as pdf:

        #     CURVES = []
        #     for roi in np.arange(data.iscell.sum())[:Nmax]:
        #         print('   - gaussian-blob analysis for ROI #%i / %i' % (roi+1, data.iscell.sum()))
        #         fig, contrasts, max_response_curve = ROI_analysis(data, roiIndex=roi, iprotocol=iprotocol)
        #         pdf.savefig(fig)
        #         plt.close(fig)
        #         if np.max(max_response_curve)>0:
        #             CURVES.append(max_response_curve)
        #     #
        #     fig = summary_fig(contrasts, CURVES, data.iscell.sum())
        #     pdf.savefig(fig)
        #     plt.close(fig)

    elif data.metadata['Electrophy']:
        with PdfPages(pdf_filename) as pdf:
            fig, fig2 = Ephys_analysis(data, iprotocol=iprotocol)
            pdf.savefig(fig)
            plt.close(fig)
            pdf.savefig(fig2)
            plt.close(fig2)
            

    print('[ok] gaussian-blobs analysis saved as: "%s" ' % pdf_filename)

if __name__=='__main__':
    
    import argparse

    parser=argparse.ArgumentParser()
    parser.add_argument("datafile", type=str)
    parser.add_argument("--iprotocol", type=int, default=-1, help='index for the protocol in case of multiprotocol in datafile')
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    if '.nwb' in args.datafile:
        analysis_pdf(args.datafile, iprotocol=args.iprotocol)
    else:
        print('/!\ Need to provide a NWB datafile as argument ')
        
