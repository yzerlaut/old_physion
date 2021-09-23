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

def ROI_analysis(FullData,
                 roiIndex=0,
                 iprotocol = 0,
                 subprotocol_id=0,
                 verbose=False,
                 response_significance_threshold=0.01,
                 with_responsive_angles = False,
                 stat_test_props=dict(interval_pre=[-2,0], interval_post=[2,4],
                                      test='ttest', positive=True)):
    """
    looming stim analysis
    """

    EPISODES = EpisodeResponse(FullData,
                               protocol_id=iprotocol,
                               quantity='CaImaging', subquantity='dF/F',
                               prestim_duration=3.,
                               roiIndex = roiIndex)

    fig, AX = FullData.plot_trial_average(EPISODES=EPISODES,
                                          protocol_id=iprotocol,
                                          quantity='CaImaging', subquantity='dF/F',
                                          roiIndex = roiIndex,
                                          column_keys=['x-center', 'y-center'],
                                          row_keys=['color'],
                                          # color_key='dotcolor',
                                          ybar=1., ybarlabel='1dF/F',
                                          xbar=1., xbarlabel='1s',
                                          fig_preset='raw-traces-preset',
                                          with_annotation=True,
                                          with_stat_test=True, stat_test_props=stat_test_props,
                                          with_screen_inset=True,
                                          verbose=verbose)

    cell_resp = EPISODES.compute_summary_data(stat_test_props,
                                              response_significance_threshold=0.01)
    
    return fig, cell_resp


def analysis_pdf(datafile, iprotocol=0, Nmax=1000000):

    data = MultimodalData(datafile)

    if not os.path.isdir(summary_pdf_folder(datafile)):
        os.mkdir(summary_pdf_folder(datafile))
    
    pdf_filename = os.path.join(summary_pdf_folder(datafile), '%s-moving-dots_selectivity.pdf' % data.protocols[iprotocol])
    
    CELL_RESPS = []
    with PdfPages(pdf_filename) as pdf:

        for roi in np.arange(data.iscell.sum())[:Nmax]:

            fig, cell_resp = ROI_analysis(data, roiIndex=roi, iprotocol=iprotocol)
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
    parser.add_argument('-ip', "--iprotocol", type=int, default=0, help='index for the protocol in case of multiprotocol in datafile')
    parser.add_argument('-nmax', "--Nmax", type=int, default=1000000)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    if '.nwb' in args.datafile:
        analysis_pdf(args.datafile, iprotocol=args.iprotocol, Nmax=args.Nmax)
    else:
        print('/!\ Need to provide a NWB datafile as argument ')








