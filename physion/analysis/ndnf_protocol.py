import sys, os, pathlib
import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from datavyz import graph_env_manuscript as ge

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from dataviz.show_data import EpisodeResponse
from analysis.tools import summary_pdf_folder
from analysis import stat_tools


class ROI_fig:

    def __init__(self, figsize=(.5,1.), wspace=0.4, top=2.):

        pattern = [[2,1],[2,1],[2,1],[2,1], # patches
                   [1,1], # -- space --
                   [3,1], # looming
                   [1,1], # -- space --
                   [2,1],[2,1],[2,1],[2,1], # patches
                   [1,1], # -- space --
                   [2,1],[2,1],[2,1],[2,1]]

        self.fig, self.AX = ge.figure(axes_extents=[pattern], figsize=(.5,1.),
                                      wspace=wspace, top=top, reshape_axes=False)

        for iax in [4,6,11]:
            self.AX[0][iax].axis('off')

        ge.annotate(self.AX[0][1], 'static-patches\n\n', (1,1), ha='center')
        ge.annotate(self.AX[0][5], 'looming-stim\n\n', (.5,1), ha='center')
        ge.annotate(self.AX[0][8], 'drifting-gratings\n\n', (1,1), ha='center')
        ge.annotate(self.AX[0][13], 'moving-dots\n\n', (1,1), ha='center')

    def set_common_ylim(self):
        
        ylim = [np.min([ax.get_ylim()[0] for ax in self.AX[0]]), np.max([ax.get_ylim()[1] for ax in self.AX[0]])]
        
        for ax in self.AX[0]:
            ax.set_ylim(ylim)
            ax.axis('off')

        for iax in [0,5,7,12]:
            self.AX[0][iax].axis('off')
            ge.draw_bar_scales(self.AX[0][iax], Xbar=1, Xbar_label='1s', Ybar=1,  Ybar_label='1dF/F',
                               loc='top-left')

    def close(self):
        plt.close(self.fig)
    
        
class NDNF_protocol:

    def __init__(self, datafile,
                 quantity='dFoF'):

        self.datafile = datafile
        
        options={'quantities':[quantity]}
        
        # static patches
        # -------------------
        self.static_patches_episodes = EpisodeResponse(datafile,
                                                  protocol_id=0, **options)
        # looming stim
        # -------------------
        self.looming_stim_episodes = EpisodeResponse(datafile,
                                                protocol_id=1, **options)

        # drifting gratings
        # -------------------
        self.drifting_gratings_episodes = EpisodeResponse(datafile,
                                                     protocol_id=2, **options)
        # moving dots
        # -------------------
        self.moving_dots_episodes = EpisodeResponse(datafile,
                                                    protocol_id=3, **options)

        # getting nROIs 
        self.nROIs = self.moving_dots_episodes.data.valid_roiIndices.sum()


    def roi_analysis_on_fig(self, roiIndex, ROI_fig, roiIndices=None):
        
            plot_options = dict(with_std=True,
                                with_stat_test=True, stat_test_props=dict(interval_pre=[-1.5,0],
                                                                          interval_post=[0.5,2],
                                                                          test='ttest',
                                                                          positive=True),
                                with_annotation=True,
                                fig=ROI_fig.fig)

            # static patches
            # -------------------
            self.static_patches_episodes.plot_trial_average(roiIndex=roiIndex, roiIndices=roiIndices,
                                                       column_key='angle', AX=[ROI_fig.AX[0][:4]], **plot_options)

            # looming stim
            # -------------------
            self.looming_stim_episodes.plot_trial_average(roiIndex=roiIndex, roiIndices=roiIndices,
                                                     AX=[[ROI_fig.AX[0][5]]], **plot_options)


            # drifting gratings
            # -------------------
            self.drifting_gratings_episodes.plot_trial_average(roiIndex=roiIndex, roiIndices=roiIndices,
                                                          column_key='angle', AX=[ROI_fig.AX[0][7:12]], **plot_options)

            # moving dots
            # -------------------
            self.moving_dots_episodes.plot_trial_average(roiIndex=roiIndex, roiIndices=roiIndices,
                                                    column_key='direction', AX=[ROI_fig.AX[0][12:16]], **plot_options)

            ROI_fig.set_common_ylim()

            
    def analysis_pdf(self,
                     Nmax=1000000,
                     verbose=False):

    
        # building pdf
        pdf_filename = os.path.join(summary_pdf_folder(self.datafile), 'ndnf_protocol.pdf')

        with PdfPages(pdf_filename) as pdf:

            # roi average
            fig = ROI_fig()
            self.roi_analysis_on_fig(None, fig, roiIndices='sum')
            pdf.savefig(fig.fig)
            fig.close()

            # roi by roi
            for roi in np.arange(self.nROIs)[:Nmax]:

                if verbose:
                    print('   - ndnf-protocol analysis for ROI #%i / %i' % (roi+1, self.nROIs))
                fig = ROI_fig()
                self.roi_analysis_on_fig(roi, fig)
                pdf.savefig(fig.fig)
                fig.close()


        if verbose:
            print('[ok] ndnf protocol analysis saved as: "%s" ' % pdf_filename)


if __name__=='__main__':
    
    import argparse

    parser=argparse.ArgumentParser()
    parser.add_argument("datafile", type=str)
    parser.add_argument('-nmax', "--Nmax", type=int, default=1000000)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    
    if '.nwb' in args.datafile:
        protocol = NDNF_protocol(args.datafile)
        protocol.analysis_pdf(Nmax=args.Nmax, verbose=True)
    else:
        print('/!\ Need to provide a NWB datafile as argument ')








