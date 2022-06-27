import sys, os, pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from analysis.protocol_scripts.motion_contour_interaction import *


def run_analysis_and_save_figs(datafile,
                               suffix='',
                               folder='./',
                               Ybar=0.1):


    pdf_filename = os.path.join(summary_pdf_folder(datafile), 'MCI-neurometric-task.pdf')

    data = MCI_data(datafile)
   
    contour_key= 'contrast'
    contour_values = data.episode_static_patch.varied_parameters['contrast']

    keys = [k for k in data.episode_moving_dots.varied_parameters.keys() if k!='repeat']
    if len(keys)==0:
        motion_key, motion_keys = '', ['']
    elif len(keys)==1:
        motion_key, motion_keys = keys[0], data.episode_moving_dots.varied_parameters[keys[0]]
    else:
        print('\n\n /!\ MORE THAN ONE MOTION KEY /!\ \n    --> needs special analysis   \n\n ')

    if 'patch-delay' in data.episode_mixed.varied_parameters.keys():
        mixed_only_key='patch-delay'
    else:
        mixed_only_key=''


    with PdfPages(pdf_filename) as pdf:

        # static patches
        fig, AX = data.episode_static_patch.plot_trial_average(roiIndices='mean', 
                                                               color_key='contrast',
                                                               color=[ge.viridis(c/(len(contrasts)-1) for c in range(len(contrasts)],
                                                               with_annotation=True,
                                                               with_std=False, ybar=Ybar, ybarlabel='%.1fdF/F'%Ybar, 
                                                               xbar=1, xbarlabel='1s')
        fig.suptitle('static patches\n\n', fontsize=9)
        pdf.savefig(fig);plt.close(fig)
        
        # moving dots
        fig, AX = data.episode_moving_dots.plot_trial_average(roiIndices='mean', 
                                                              with_annotation=True,
                                                              with_std=False, ybar=Ybar, ybarlabel='%.1fdF/F'%Ybar, 
                                                              xbar=1, xbarlabel='1s')
        fig.suptitle('moving dots\n\n', fontsize=9)
        pdf.savefig(fig);plt.close(fig)

        # mixed stimuli
        fig, AX = data.episode_mixed.plot_trial_average(roiIndices='mean', 
                                                        color_key='patch-contrast',
                                                        color=[ge.viridis(c/(len(contrasts)-1) for c in range(len(contrasts)],
                                                        column_key=mixed_only_key,
                                                        with_annotation=True,
                                                        with_std=False, ybar=Ybar, ybarlabel='%.1fdF/F'%Ybar, 
                                                        xbar=1, xbarlabel='1s')
        fig.suptitle('mixed static-patch + moving-dots\n\n', fontsize=9)    
        pdf.savefig(fig);plt.close(fig)


        ## Focusing on cells responding to contour features

        rois_of_interest_contour = find_responsive_cells(data.episode_static_patch,
                                                         param_key=contour_key,
                                                         interval_pre=[-1,0],
                                                         interval_post=[0.5,1.5])

        rois_of_interest_motion = find_responsive_cells(data.episode_moving_dots,
                                                        param_key=motion_key,
                                                        interval_pre=[-2,0],
                                                        interval_post=[1,3])

        rois_of_interest_contour_only = exclude_motion_sensitive_cells(rois_of_interest_contour,
                                                                       rois_of_interest_motion)


        fig, ax = make_proportion_fig(data,
                                      rois_of_interest_contour,
                                      rois_of_interest_motion,
                                      rois_of_interest_contour_only)
        pdf.savefig(fig);plt.close(fig)


        #for key in [key for key in rois_of_interest_contour_only if 'resp' not in key]:
        for key in ['contrast_1.0']:
            # only the max contrast

            print('- %s patch --> %i significantly modulated cells (%.1f%%) ' % (key, len(rois_of_interest_contour_only[key]),
                                                             100*len(rois_of_interest_contour_only[key])/data.nROIs))

            if len(rois_of_interest_contour_only[key])>0:

                fig, AX = data.episode_static_patch.plot_trial_average(roiIndices=rois_of_interest_contour_only[key], 
                                                                       column_key=contour_key,
                                                                       with_annotation=True,
                                                                       with_std=False,
                                                                       ybar=Ybar, ybarlabel='%.1fdF/F'%Ybar, 
                                                                       xbar=1, xbarlabel='1s')
                fig.suptitle('static patches --> cells resp. to %s\n\n\n' % key, fontsize=8)
                pdf.savefig(fig);plt.close(fig)

                fig, AX = data.episode_moving_dots.plot_trial_average(roiIndices=rois_of_interest_contour_only[key], 
                                                                      with_annotation=True,
                                                                      with_std=False,
                                                                      ybar=Ybar, ybarlabel='%.1fdF/F'%Ybar, 
                                                                      xbar=1, xbarlabel='1s')
                fig.suptitle('moving-dots --> cells resp. to %s\n\n\n' % key, fontsize=8)
                pdf.savefig(fig);plt.close(fig)

                fig, AX = data.episode_mixed.plot_trial_average(roiIndices=rois_of_interest_contour_only[key], 
                                                                column_key=('patch-%s'%contour_key if (contour_key!='') else ''),
                                                                row_key=mixed_only_key,
                                                                with_annotation=True,
                                                                with_std=False,
                                                                ybar=Ybar, ybarlabel='%.1fdF/F'%Ybar, 
                                                                xbar=1, xbarlabel='1s')
                fig.suptitle('mixed-stim --> cells resp. to %s\n\n\n' % key, fontsize=8)
                pdf.savefig(fig);plt.close(fig)

        print('[ok] motion-contour-interaction analysis saved as: "%s" ' % pdf_filename)


if __name__=='__main__':
    import argparse

    parser=argparse.ArgumentParser()
    parser.add_argument("datafile", type=str)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument('-s', "--suffix", default='', type=str)

    args = parser.parse_args()

    run_analysis_and_save_figs(args.datafile, 
                               suffix=('-'+args.suffix if args.suffix!='' else ''))


