# need "physion" installed

import sys, os, pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from datavyz import graph_env_screen as ge

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from analysis.read_NWB import Data
from dataviz.show_data import MultimodalData, EpisodeResponse
from analysis.tools import summary_pdf_folder


def build_linear_pred(patch_resp, mvDot_resp, 
                      episode_static_patch, episode_moving_dots,
                      delay=0):

    i_mVdot_center = np.argwhere(episode_moving_dots.t>delay)[0][0]
    patch_evoked_t = episode_static_patch.t>0
    
    resp = 0*episode_moving_dots.t + mvDot_resp # mvDot_resp by default

    # and we add the patch-evoked response (substracting its baseline)
    imax = min([len(patch_resp[patch_evoked_t]), len(mvDot_resp)-i_mVdot_center])-1
    resp[i_mVdot_center:i_mVdot_center+imax] += patch_resp[patch_evoked_t][:imax]-patch_resp[patch_evoked_t][0]
    
    return resp

def interaction_fig(episode_static_patch, episode_moving_dots, episode_mixed,
                    roiIndices=[0],
                    moving_dot_direction_index = 0, moving_dot_label='moving-dots',
                    contour_param_key = 'angle', contour_param_index = 0, 
                    fixed_delay=None, suffix='',
                    Ybar=0.1):

    if 'patch-delay' in episode_mixed.varied_parameters:
        delays = episode_mixed.varied_parameters['patch-delay']
    else:
        delays = [fixed_delay] # will throw an error if fixed_delay is not specified

    fig, AX = ge.figure(axes=(2+len(delays), 1), figsize=(1.,1), wspace=0.5)

    # static patch only
    response = episode_static_patch.get_response('dFoF', roiIndices=roiIndices)
    if contour_param_key!='':
        resp_static_patch = response[episode_static_patch.find_episode_cond(contour_param_key, contour_param_index), :].mean(axis=0)
    else:
        resp_static_patch = response.mean(axis=0)
    resp_static_patch -= resp_static_patch[episode_static_patch.t<0].mean()
    cond = episode_static_patch.t>-1.5
    AX[0].plot(episode_static_patch.t[cond], resp_static_patch[cond],  color='k', label='true resp.')
    if contour_param_key!='':
        ge.set_plot(AX[0], [], title='static-patch (%s=%i$^{o}$)' % (contour_param_key,
                                    episode_static_patch.varied_parameters[contour_param_key][contour_param_index]))
    else:
        ge.set_plot(AX[0], [], title='static-patch')

    # moving dots only
    response = episode_moving_dots.get_response('dFoF', roiIndices=roiIndices)
    resp_moving_dots = response[episode_moving_dots.find_episode_cond('direction', moving_dot_direction_index), :].mean(axis=0)
    resp_moving_dots -= resp_moving_dots[episode_moving_dots.t<0].mean()
    AX[1].plot(episode_moving_dots.t, resp_moving_dots,  color='k')
    ge.set_plot(AX[1], [], title='%s (%i$^{o}$)' % (moving_dot_label, episode_moving_dots.varied_parameters['direction'][ moving_dot_direction_index]))

    # mixed stimuli
    response = episode_mixed.get_response('dFoF', roiIndices=roiIndices)

    for ax, delay_index in zip(AX[2:], range(len(delays))):

        if contour_param_key!='':
            if ('patch-delay' in episode_mixed.varied_parameters):
                resp_mixed = response[episode_mixed.find_episode_cond(['direction',
                                                                       'patch-%s' % contour_param_key,
                                                                       'patch-delay'],
                                  [moving_dot_direction_index, contour_param_index, delay_index]), :].mean(axis=0)
                resp_mixed -= resp_mixed[episode_mixed.t<0].mean()
            else:
                resp_mixed = response[episode_mixed.find_episode_cond(['direction',
                                                                       'patch-%s' % contour_param_key],
                                            [moving_dot_direction_index, contour_param_index]), :].mean(axis=0)
                resp_mixed -= resp_mixed[episode_mixed.t<0].mean()
        else:
            if ('patch-delay' in episode_mixed.varied_parameters):
                resp_mixed = response[episode_mixed.find_episode_cond(['direction', 'patch-delay'],
                                                                      [moving_dot_direction_index, delay_index]), :].mean(axis=0)
                resp_mixed -= resp_mixed[episode_mixed.t<0].mean()
            else:
                resp_mixed = response[episode_mixed.find_episode_cond(['direction'],
                                                                      [moving_dot_direction_index]), :].mean(axis=0)
                resp_mixed -= resp_mixed[episode_mixed.t<0].mean()
        ax.plot(episode_moving_dots.t, build_linear_pred(resp_static_patch, resp_moving_dots, 
                                                         episode_static_patch, episode_moving_dots, 
                                                         delay=delays[delay_index]),
                 color=ge.red, label='linear pred.')
        ax.plot(episode_moving_dots.t, resp_mixed,  color='k', label='true resp.')
        if delay_index==0:
            ge.legend(ax, frameon=False, size='xxx-small', loc='upper right')

    ge.set_common_ylims(AX)
    ge.set_common_xlims(AX)

    patch_duration = episode_mixed.data.metadata['Protocol-%i-presentation-duration' % (episode_mixed.data.get_protocol_id('static-patch'+suffix)+1)]
    dot_duration = episode_mixed.data.metadata['Protocol-%i-presentation-duration' % (episode_mixed.data.get_protocol_id('moving-dots'+suffix)+1)]
    center = dot_duration/2.
    center = 0
    
    AX[0].fill_between([0,patch_duration], AX[0].get_ylim()[0]*np.ones(2), AX[0].get_ylim()[1]*np.ones(2), color=ge.blue, alpha=.3, lw=0)
    AX[1].fill_between([0,dot_duration], AX[1].get_ylim()[0]*np.ones(2), AX[1].get_ylim()[1]*np.ones(2), color=ge.orange, alpha=.1, lw=0)
    for ax, delay_index in zip(AX[2:], range(4)):
        delay = delays[delay_index]
        ax.fill_between([delay+center, delay+center+patch_duration],
                        ax.get_ylim()[0]*np.ones(2), ax.get_ylim()[1]*np.ones(2), color=ge.blue, alpha=.3, lw=0)
        ax.fill_between([0,dot_duration],
                        ax.get_ylim()[0]*np.ones(2), ax.get_ylim()[1]*np.ones(2), color=ge.orange, alpha=.1, lw=0)
        ge.set_plot(ax, [], title='mixed:, delay=%.1fs' % (delay+center))

    for ax in AX:
        ge.draw_bar_scales(ax, Xbar=1, Xbar_label='1s', Ybar=Ybar, Ybar_label=str(Ybar)+'dF/F')
    return fig, AX

def find_contour_responsive_cells(episode_static_patch, contour_param_values,
                                  response_significance_threshold=0.01):
    rois_of_interest = {}
    for v in contour_param_values:
        rois_of_interest['%ideg' % v] = []

    for roi in range(episode_static_patch.data.nROIs):
        summary_data = episode_static_patch.compute_summary_data(dict(interval_pre=[-1,0],
                                                                      interval_post=[0.5,1.5],
                                                                      test='wilcoxon', positive=True),
                                                                  response_args={'quantity':'dFoF', 'roiIndex':roi},
                                                                  response_significance_threshold=response_significance_threshold)
        for v, significant in zip(contour_param_values, summary_data['significant']):
            if significant:
                rois_of_interest['%ideg' % v].append(roi)
    return rois_of_interest

def find_motion_responsive_cells(episode_moving_dots, mvDot_param_values,
                                 response_significance_threshold=0.01):
    rois_of_interest = {}
    for v in mvDot_param_values:
        rois_of_interest['%ideg' % v] = []

    for roi in range(episode_static_patch.data.nROIs):
        summary_data = episode_static_patch.compute_summary_data(dict(interval_pre=[-1,0],
                                                                      interval_post=[0.5,1.5],
                                                                      test='wilcoxon', positive=True),
                                                                  response_args={'quantity':'dFoF', 'roiIndex':roi},
                                                                  response_significance_threshold=response_significance_threshold)
        for v, significant in zip(contour_param_values, summary_data['significant']):
            if significant:
                rois_of_interest['%ideg' % v].append(roi)
    return rois_of_interest

    
def run_analysis_and_save_figs(datafile,
                               suffix='',
                               folder='./',
                               Ybar=0.1):


    pdf_filename = os.path.join(summary_pdf_folder(datafile), 'motion-contour-interaction.pdf')

    data = Data(datafile, metadata_only=True)
    
    # computing episodes
    episode_static_patch = EpisodeResponse(datafile,
                                           protocol_id=data.get_protocol_id('static-patch'+suffix),
                                           quantities=['dFoF'],
                                           prestim_duration=3)
    episode_moving_dots = EpisodeResponse(datafile,
                                          protocol_id=data.get_protocol_id('moving-dots'+suffix),
                                          quantities=['dFoF'],
                                          prestim_duration=3)

    episode_mixed = EpisodeResponse(datafile,
                                    protocol_id=data.get_protocol_id('mixed-moving-dots-static-patch'+suffix),
                                    quantities=['dFoF'],
                                    prestim_duration=3)
    
    episode_random_dots, episode_mixed_random_dots = None, None
    if 'random-line-dots'+suffix in data.protocols:
        episode_random_dots = EpisodeResponse(datafile,
                                              protocol_id=data.get_protocol_id('random-line-dots'+suffix),
                                              quantities=['dFoF'],
                                              prestim_duration=3)
        
    if 'random-mixed-moving-dots-static-patch'+suffix in data.protocols:
        episode_mixed_random_dots = EpisodeResponse(datafile,
                                                    protocol_id=data.get_protocol_id('random-mixed-moving-dots-static-patch'+suffix),
                                                    quantities=['dFoF'],
                                                    prestim_duration=3)

        
    if 'angle' in episode_static_patch.varied_parameters:
        contour_param_key = 'angle'
        contour_param_values = episode_static_patch.varied_parameters['angle']
    elif 'radius' in episode_static_patch.varied_parameters:
        contour_param_key = 'radius'
        contour_param_values = episode_static_patch.varied_parameters['radius']
    else:
        contour_param_key = ''
        contour_param_values = [0]

    with PdfPages(pdf_filename) as pdf:
        # static patches
        fig, AX = episode_static_patch.plot_trial_average(roiIndices='sum', 
                                                          column_key=contour_param_key,
                                                          with_annotation=True,
                                                          with_std=False, ybar=Ybar, ybarlabel='%.1fdF/F'%Ybar, 
                                                          xbar=1, xbarlabel='1s')
        fig.suptitle('static patches\n\n', fontsize=9)
        pdf.savefig(fig);plt.close(fig)
        
        # moving dots
        fig, AX = episode_moving_dots.plot_trial_average(roiIndices='mean', 
                                                         column_key='direction',
                                                         with_annotation=True,
                                                         with_std=False, ybar=Ybar, ybarlabel='%.1fdF/F'%Ybar, 
                                                         xbar=1, xbarlabel='1s')
        fig.suptitle('moving dots\n\n', fontsize=9)
        pdf.savefig(fig);plt.close(fig)

        # mixed stimuli
        fig, AX = episode_mixed.plot_trial_average(roiIndices='mean', 
                                                   column_key='patch-delay',
                                                   row_key='direction',
                                                   color_key=('patch-%s'%contour_param_key if (contour_param_key!='') else ''),
                                                   with_annotation=True,
                                                   with_std=False, ybar=Ybar, ybarlabel='%.1fdF/F'%Ybar, 
                                                   xbar=1, xbarlabel='1s')
        fig.suptitle('mixed static-patch + moving-dots\n\n', fontsize=9)    
        pdf.savefig(fig);plt.close(fig)

        if episode_random_dots is not None:
            fig, AX = episode_random_dots.plot_trial_average(roiIndices='mean', 
                                                             column_key='direction',
                                                             with_annotation=True,
                                                             with_std=False, ybar=Ybar, ybarlabel='%.1fdF/F'%Ybar, 
                                                             xbar=1, xbarlabel='1s')
            fig.suptitle('random line moving-dots\n\n', fontsize=9)    
            pdf.savefig(fig);plt.close(fig)

        if episode_mixed_random_dots is not None:
            fig, AX = episode_mixed_random_dots.plot_trial_average(roiIndices='mean', 
                                                                   column_key='direction',
                                                                   color_key=('patch-%s'%contour_param_key if (contour_param_key!='') else ''),
                                                                   with_annotation=True,
                                                                   with_std=False, ybar=Ybar, ybarlabel='%.1fdF/F'%Ybar, 
                                                                   xbar=1, xbarlabel='1s')
            fig.suptitle('mixed static-patch + random line moving-dots\n\n', fontsize=9)    
            pdf.savefig(fig);plt.close(fig)
            
        ## Focusing on cells responding to contour features

        rois_of_interest = find_contour_responsive_cells(episode_static_patch, contour_param_values)
        for v in contour_param_values:
            print('- %i deg patch --> %i significantly modulated cells (%.1f%%) ' % (v, len(rois_of_interest['%ideg' % v]),
                                                         100*len(rois_of_interest['%ideg' % v])/episode_static_patch.data.nROIs))


        for i, v in enumerate(contour_param_values):

            if len(rois_of_interest['%ideg' % v])>0:
                fig, AX = episode_static_patch.plot_trial_average(roiIndices=rois_of_interest['%ideg' % v], 
                                                                  column_key=contour_param_key,
                                                                  with_annotation=True,
                                                                  with_std=False, ybar=Ybar, ybarlabel='%.1fdF/F'%Ybar, 
                                                                  xbar=1, xbarlabel='1s')
                fig.suptitle('static patches --> cells resp. to %i$^o$\n\n\n' % v, fontsize=8)
                pdf.savefig(fig);plt.close(fig)

                fig, AX = episode_moving_dots.plot_trial_average(roiIndices=rois_of_interest['%ideg' % v], 
                                                                  column_key='direction',
                                                                  with_annotation=True,
                                                                  with_std=False, ybar=Ybar, ybarlabel='%.1fdF/F'%Ybar, 
                                                                  xbar=1, xbarlabel='1s')
                fig.suptitle('moving-dots --> cells resp. to %i$^o$\n\n\n' % v, fontsize=8)
                pdf.savefig(fig);plt.close(fig)
                fig, AX = episode_mixed.plot_trial_average(roiIndices=rois_of_interest['%ideg' % v], 
                                                       column_key='patch-delay',
                                                       row_key='direction',
                                                       color_key=('patch-%s'%contour_param_key if (contour_param_key!='') else ''),
                                                       with_annotation=True,
                                                       with_std=False, ybar=Ybar, ybarlabel='%.1fdF/F'%Ybar, 
                                                       xbar=1, xbarlabel='1s')
                fig.suptitle('mixed-stim --> cells resp. to %i$^o$\n\n\n' % v, fontsize=8)
                pdf.savefig(fig);plt.close(fig)


                for j, d in enumerate(episode_mixed.varied_parameters['direction']):
                    fig, AX = interaction_fig(episode_static_patch, episode_moving_dots, episode_mixed,
                                              roiIndices=rois_of_interest['%ideg' % v],
                                              moving_dot_direction_index = j,
                                              contour_param_key=contour_param_key,
                                              suffix=suffix,
                                              contour_param_index = i)
                    pdf.savefig(fig);plt.close(fig)
                    if (episode_random_dots is not None) and (episode_mixed_random_dots is not None):
                        patch_delay= episode_mixed.data.metadata['Protocol-%i-patch-delay-1' % (episode_mixed.data.get_protocol_id('random-mixed-moving-dots-static-patch'+suffix)+1)]
                        fig, AX = interaction_fig(episode_static_patch, episode_random_dots, episode_mixed_random_dots,
                                                  roiIndices=rois_of_interest['%ideg' % v],
                                                  moving_dot_direction_index = j, moving_dot_label='random-line-dots',
                                                  contour_param_key=contour_param_key,
                                                  fixed_delay = patch_delay,
                                                  suffix=suffix,
                                                  contour_param_index = i)
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
