import sys, os, pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from dataviz.datavyz.datavyz import graph_env_manuscript as ge
from analysis.read_NWB import Data
# from analysis.process_NWB import EpisodeResponse
from dataviz.show_data import MultimodalData, EpisodeResponse
from analysis.tools import summary_pdf_folder

#############################################################################
#############################################################################
#############################################################################
#############################################################################

def find_responsive_cells(episode,
                          param_key='',
                          interval_pre=[-1,0],
                          interval_post=[0.5,1.5],
                          minimum_relative_shift=0.1,
                          response_significance_threshold=0.01):
    """
    Two conditions:
        1) statistically significant positive shift over trials
        2) > 10% increase of trial average response with respect to baseline (pre)
    """
    rois_of_interest = {}
    param_values = episode.varied_parameters[param_key] if param_key!='' else [0]

    for v in param_values:
        rois_of_interest['%s_%i' % (param_key, v)] = []
        rois_of_interest['%s_%i_resp' % (param_key, v)] = []

    for roi in range(episode.data.nROIs):
        summary_data = episode.compute_summary_data(dict(interval_pre=interval_pre,
                                                         interval_post=interval_post,
                                                         test='anova', positive=True),
                                                         response_args={'quantity':'dFoF', 'roiIndex':roi},
                                                         response_significance_threshold=response_significance_threshold)
        for p, significant, v, rv in zip(param_values, 
                                         summary_data['significant'],
                                         summary_data['value'],
                                         summary_data['relative_value']):
            if significant and rv>minimum_relative_shift:
                rois_of_interest['%s_%i' % (param_key, p)].append(roi)
                rois_of_interest['%s_%i_resp' % (param_key, p)].append(v)
    return rois_of_interest


def exclude_motion_sensitive_cells(rois_of_interest_contour, rois_of_interest_motion,
                                   activity_factor = 1.):
    """
    if the activity_factor * motion response is not smaller than the contour response, excluded
    """
    rois_of_interest_contour_only = {}

    contour_keys = [key for key in rois_of_interest_contour if 'resp' not in key]
    motion_keys = [k for k in rois_of_interest_motion if 'resp' not in k]
    
    for key in contour_keys:
        #print(len(rois_of_interest_contour[key]))
        rois_of_interest_contour_only[key] = []
        for k in motion_keys:
            #print(key, k)
            for r, roi in enumerate(rois_of_interest_contour[key]):
                rn = np.argwhere(np.array(rois_of_interest_motion[k])==roi).flatten()
                #print(roi, rn)
                if len(rn)>0:
                    #print('motion sensitive')
                    #print(rois_of_interest_motion[k+'_resp'][rn[0]], rois_of_interest_contour[key+'_resp'][r])
                    if (rois_of_interest_contour[key+'_resp'][r]>(activity_factor*rois_of_interest_motion[k+'_resp'][rn[0]])) and\
                            (roi not in rois_of_interest_contour_only[key]):
                        rois_of_interest_contour_only[key].append(roi)
                        #print('   -> but stronger contour')
                elif roi not in rois_of_interest_contour_only[key]:
                    #print('not motion sensitive')
                    rois_of_interest_contour_only[key].append(roi)
        print('')
    return rois_of_interest_contour_only


def interaction_panel(responses, 
                      ax=None, 
                      title='',
                      linear_key='linear', mixed_key='mixed',
                      with_episodes_highlights=True,
                      tmin=-3):
    if ax is None:
        fig, ax = ge.figure()
    else:
        fig = None
    
    # mixed
    cond = responses['t_motion']>tmin
    if 'linear' in responses:
        ax.plot(responses['t_motion'][cond], responses[linear_key][cond],  color='r')
    ax.plot(responses['t_motion'][cond], responses[mixed_key][cond],  color='k')            
    ge.set_plot(ax, [], title=title)
                
    ax.fill_between(responses['delay']+np.arange(2)*responses['patch-duration'],
                        ax.get_ylim()[0]*np.ones(2), ax.get_ylim()[1]*np.ones(2), color=ge.blue, alpha=.3, lw=0)
    ax.fill_between([0,responses['mvDot-duration']],
                        ax.get_ylim()[0]*np.ones(2), ax.get_ylim()[1]*np.ones(2), color=ge.orange, alpha=.1, lw=0)
        
    return fig, ax
        
def interaction_fig(responses,
                    static_patch_label='...',
                    moving_dots_label='...',
                    mixed_label='...',
                    random=False,
                    Ybar=0.2, Ybar_label='0.2dF/F', 
                    tmin=-3):
    
    fig, AX = ge.figure(axes=(3, 1), figsize=(1.,1), wspace=0.5, right=8, top=2)
    
    ax = ge.inset(fig, [.9,.4,.07,.5])
    ax.bar([0], [responses['linear-integral']], color=ge.red)
    ax.bar([1], [responses['mixed-integral']], color='k')
    ge.set_plot(ax, ['left'], ylabel='integ. (resp.s)')

    # static-patch
    cond = responses['t_contour']>tmin/2.
    if 'contour_std' in responses and responses['contour_std'] is not None:
        ge.plot(responses['t_contour'][cond], responses['contour'][cond], sy=responses['contour_std'][cond] , color='k',ax=AX[0])
    else:
        AX[0].plot(responses['t_contour'][cond], responses['contour'][cond],  color='k')
    ge.set_plot(AX[0], [], title = static_patch_label)

    # mv Dots
    cond = responses['t_motion']>tmin
    if 'motion_std' in responses and responses['motion_std'] is not None:
        ge.plot(responses['t_motion'][cond], responses['motion'][cond], sy=responses['motion_std'][cond] , color='k',ax=AX[1])
    else:
        AX[1].plot(responses['t_motion'][cond], responses['motion' if not random else 'random'][cond],  color='k')
    ge.set_plot(AX[1], [], title = moving_dots_label)

    # mixed
    interaction_panel(responses, 
                      ax=AX[2], title=mixed_label,
                      mixed_key='mixed' if not random else 'mixed-random',
                      linear_key='linear' if not random else 'linear-random',
                      with_episodes_highlights=False, # only after that we have set common lims
                      tmin=tmin)
        
    ge.set_common_ylims(AX)
    ge.set_common_xlims(AX)
        
    AX[0].fill_between([0,responses['patch-duration']], AX[0].get_ylim()[0]*np.ones(2), AX[0].get_ylim()[1]*np.ones(2), color=ge.blue, alpha=.3, lw=0)
    AX[1].fill_between([0,responses['mvDot-duration']], AX[1].get_ylim()[0]*np.ones(2), AX[1].get_ylim()[1]*np.ones(2), color=ge.orange, alpha=.1, lw=0)
        
    AX[2].plot(responses['delay']+np.arange(2)*responses['integral_window'], AX[2].get_ylim()[1]*np.ones(2), 'k-')
    AX[2].fill_between(responses['delay']+np.arange(2)*responses['patch-duration'], 
                           AX[2].get_ylim()[0]*np.ones(2), AX[2].get_ylim()[1]*np.ones(2), color=ge.blue, alpha=.3, lw=0)
    AX[2].fill_between([0,responses['mvDot-duration']], AX[2].get_ylim()[0]*np.ones(2), AX[2].get_ylim()[1]*np.ones(2), color=ge.orange, alpha=.1, lw=0)

    for ax in AX:
        ge.draw_bar_scales(ax, Xbar=1, Xbar_label='1s', Ybar=Ybar, Ybar_label=Ybar_label)
    ge.annotate(fig, ' n=%iROIs' % responses['nROIs'], (0.02,0.02))
    return fig, AX, ax

#############################################################################
#############################################################################
#############################################################################
#############################################################################


class MCI_data:
    
    
    def __init__(self, filename, 
                 quantities=['dFoF'],
                 prestim_duration=3):
    
        data = Data(filename, metadata_only=True, verbose=False)

        # computing episodes       
        self.episode_static_patch = EpisodeResponse(filename,
                                                    protocol_name='static-patch',
                                                    quantities=quantities,            
                                                    prestim_duration=prestim_duration, verbose=False)             

        self.episode_moving_dots = EpisodeResponse(filename,
                                                   protocol_name='moving-dots',
                                                   quantities=quantities,            
                                                   prestim_duration=prestim_duration, verbose=False)             

        self.episode_mixed = EpisodeResponse(filename,
                                             protocol_name='mixed-moving-dots-static-patch',
                                             quantities=quantities,            
                                             prestim_duration=prestim_duration, verbose=False)         

        if hasattr(self.episode_mixed.data, 'nROIs'):
            self.nROIs = self.episode_mixed.data.nROIs
        
        self.episode_random_dots, self.episode_mixed_random_dots = None, None
        if 'random-dots' in data.protocols:
            self.episode_random_dots = EpisodeResponse(filename,
                                                      protocol_name='random-dots',
                                                      quantities=quantities,            
                                                      prestim_duration=prestim_duration, verbose=False)             
        else:
            self.episode_random_dots = None
        if 'mixed-random-dots-static-patch' in data.protocols:
            self.episode_mixed_random_dots = EpisodeResponse(filename,
                                                             protocol_name='mixed-random-dots-static-patch',
                                                             quantities=quantities,            
                                                             prestim_duration=prestim_duration, verbose=False)  
        else:
            self.episode_mixed_random_dots = None
            

    def build_linear_pred(self, 
                          patch_resp, mvDot_resp,
                          delay=0,
                          baseline_window=[-0.1,0]):
        """
        the linear prediction is build by adding the patch evoke resp to the motion trace
            we remove the baseline for mthe patch resp so that it has a zero baseline
            N.B. put the "patch_baseline_end" a bit before t=0 in case some evoked response would be there because of the linear interpolation
        """

        i_patch_start = np.argwhere(self.episode_moving_dots.t>(delay+baseline_window[1]))[0][0]

        patch_evoked_t = self.episode_static_patch.t>baseline_window[1] # everything after the baseline window

        baseline_cond = (self.episode_static_patch.t>=baseline_window[0]) &\
                    (self.episode_static_patch.t<=baseline_window[1])
        baseline = np.mean(patch_resp[baseline_cond])

        resp = 0*self.episode_moving_dots.t + mvDot_resp # mvDot_resp by default

        # and we add the patch-evoked response (substracting its baseline)
        imax = min([len(patch_resp[patch_evoked_t]), len(mvDot_resp)-i_patch_start])-1
        resp[i_patch_start:i_patch_start+imax] += patch_resp[patch_evoked_t][:imax]-baseline

        return resp

    def build_contour_pred(self,
                           mixed_resp, mvDot_resp,
                           delay=0,
                           baseline_window=[-0.1, 0]):
        
        resp = 0*self.episode_static_patch.t
        t_mixed_cond = self.episode_mixed.t>0
        
        i_patch_start_mixed = np.argwhere(self.episode_mixed.t>delay)[0][0]
        i_patch_start_cntr = np.argwhere(self.episode_static_patch.t>0)[0][0]
        
        # substracted baseline on moving dots (more repeats)
        baseline_cond = (self.episode_moving_dots.t>=baseline_window[0]) &\
                (self.episode_moving_dots.t<=baseline_window[1])
        baseline = np.mean(mvDot_resp[baseline_cond])
        
        resp[i_patch_start_cntr:len(self.episode_static_patch.t)] = mixed_resp[i_patch_start_mixed:i_patch_start_mixed+len(self.episode_static_patch.t)-i_patch_start_cntr]-\
                                        mvDot_resp[i_patch_start_mixed:i_patch_start_mixed+len(self.episode_static_patch.t)-i_patch_start_cntr]+baseline
        resp[:i_patch_start_cntr] = baseline
        
        return resp

    def get_responses(self, 
                      static_patch_cond,
                      moving_dots_cond,
                      mixed_cond,
                      quantity='dFoF',
                      norm='', #norm='Zscore-time-variations-after-trial-averaging-per-roi',
                      integral_window=2., force_delay=None,
                      patch_baseline_window=[-0.1,0],
                      roiIndices=[0]):
        
        if norm=='Zscore-time-variations-after-trial-averaging-per-roi':
            # from mixed resp
            mixed_resp = self.episode_mixed.get_response(quantity, roiIndices=roiIndices, average_over_rois=False)[mixed_cond,:,:].mean(axis=0) # trial-average
            scaling_factor = 1./mixed_resp.std(axis=1).reshape(mixed_resp.shape[0],1)
        elif norm=='MinMax-time-variations-after-trial-averaging-per-roi':
            # from static patch
            patch_resp = self.episode_static_patch.get_response(quantity, roiIndices=roiIndices, average_over_rois=False)[static_patch_cond,:,:].mean(axis=0) # trial-average
            scaling_factor = 1./(patch_resp.max(axis=1).reshape(patch_resp.shape[0],1)-patch_resp.min(axis=1).reshape(patch_resp.shape[0],1))
        else:
            scaling_factor = 1.
            
        responses = {'nROIs':len(roiIndices),
                     'integral_window':integral_window}
        
        # static patch 
        resp = self.episode_static_patch.get_response(quantity, roiIndices=roiIndices, average_over_rois=False)[static_patch_cond,:,:].mean(axis=0) # trial-average
        responses['t_contour'] = self.episode_static_patch.t
        responses['contour'] = np.mean(scaling_factor*(resp-resp[:,self.episode_static_patch.t<0].mean(axis=1).reshape(resp.shape[0],1)), axis=0)
        
        # moving dots
        resp = self.episode_moving_dots.get_response(quantity, roiIndices=roiIndices, average_over_rois=False)[moving_dots_cond,:,:].mean(axis=0) # trial-average
        responses['motion'] = np.mean(scaling_factor*(resp-resp[:,self.episode_moving_dots.t<0].mean(axis=1).reshape(resp.shape[0],1)), axis=0)
        responses['t_motion'] = self.episode_moving_dots.t
                
        # mixed stim
        resp = self.episode_mixed.get_response(quantity, roiIndices=roiIndices, average_over_rois=False)[mixed_cond,:,:].mean(axis=0)
        responses['mixed'] = np.mean(scaling_factor*(resp-resp[:,self.episode_mixed.t<0].mean(axis=1).reshape(resp.shape[0],1)), axis=0)
        
        responses['patch-duration'] = self.episode_mixed.data.metadata['Protocol-%i-presentation-duration' % (self.episode_static_patch.protocol_id+1)]
        responses['mvDot-duration'] = self.episode_mixed.data.metadata['Protocol-%i-presentation-duration' % (self.episode_mixed.protocol_id+1)]

        # speeds
        if hasattr(self.episode_mixed, 'speed'):
            speeds = getattr(self.episode_mixed, 'speed')[mixed_cond]
            responses['mvDot-speed'] = speeds[0]
        else:
            responses['mvDot-speed'] = self.episode_mixed.data.metadata['Protocol-%i-speed' % (self.episode_mixed.protocol_id+1)]
            
        # delays
        delays = getattr(self.episode_mixed, 'patch-delay')[mixed_cond]

        if len(np.unique(delays))==1:
            responses['delay'] = delays[0] # storing delay for later
        else:
            raise Exception('\n  /!\  no unique delay: %s, \n --> unpossible to build the linear predictions !' % np.unique(delays))

        # linear pred.
        responses['linear'] = self.build_linear_pred(responses['contour'], responses['motion'], 
                                                     delay=responses['delay'],
                                                     patch_baseline_window=patch_baseline_window)
        integral_cond = (responses['t_motion']>responses['delay']) & (responses['t_motion']<responses['delay']+integral_window)
        responses['linear-integral'] = np.trapz(responses['linear'][integral_cond]-responses['linear'][responses['t_motion']<0].mean(),
                                                responses['t_motion'][integral_cond])
        responses['mixed-integral'] = np.trapz(responses['mixed'][integral_cond]-responses['mixed'][responses['t_motion']<0].mean(),
                                               responses['t_motion'][integral_cond])
        
        return responses
    
    def add_random_responses(self, responses,
                             random_cond,
                             random_mixed_cond,
                             quantity='dFoF',
                             roiIndices=[0]):
        scaling_factor = 1
        
        # moving dots
        resp = self.episode_random_dots.get_response(quantity, roiIndices=roiIndices)[random_cond].mean(axis=0)
        responses['random'] = np.mean(scaling_factor*(resp-resp[:,self.episode_random_dots.t<0].mean(axis=1).reshape(resp.shape[0],1)), axis=0)
        responses['random_std'] = np.std(scaling_factor*(resp-resp[:,self.episode_random_dots.t<0].mean(axis=1).reshape(resp.shape[0],1)), axis=0)


def make_proportion_fig(data,
                        rois_of_interest_contour,
                        rois_of_interest_motion,
                        rois_of_interest_contour_only):

    fig, ax = ge.figure(figsize=(1.2,1), top=1, bottom=2)
    n=0
    ticks=[]
    for key in list(rois_of_interest_contour.keys())[::2]:
        ax.bar([n], [100.*len(rois_of_interest_contour[key])/data.nROIs], color=ge.blue)
        ticks.append(key)
        n+=1

    for key in list(rois_of_interest_motion.keys())[::2]:
        ax.bar([n], [100.*len(rois_of_interest_motion[key])/data.nROIs], color=ge.orange)
        ticks.append(key)
        n+=1
    for key in rois_of_interest_contour_only:
        ax.bar([n], [100.*len(rois_of_interest_contour_only[key])/data.nROIs], color=ge.blue)
        ticks.append(key)
        n+=1
    ge.annotate(ax, 'contour', (0,.95), size='small', color=ge.blue)
    ge.annotate(ax, 'motion', (0.5,.95), size='small', color=ge.orange, ha='center')
    ge.annotate(ax, 'contour\nonly', (1,.95), size='small', color=ge.blue, ha='right', va='top')
    ge.set_plot(ax, xticks=np.arange(n), xticks_labels=ticks, xticks_rotation=90, ylabel='responsive ROI (%)     ')
    return fig, ax



def run_analysis_and_save_figs(datafile,
                               suffix='',
                               folder='./',
                               Ybar=0.1):


    pdf_filename = os.path.join(summary_pdf_folder(datafile), 'motion-contour-interaction.pdf')

    data = MCI_data(datafile)
   
    keys = [k for k in data.episode_static_patch.varied_parameters.keys() if k!='repeat']

    if len(keys)==0:
        contour_key, contour_keys = '', ['']
    elif len(keys)==1:
        contour_key, contour_keys = keys[0], data.episode_static_patch.varied_parameters[keys[0]]
    else:
        print('\n\n /!\ MORE THAN ONE CONTOUR KEY /!\ \n    --> needs special analysis   \n\n ')

    keys = [k for k in data.episode_moving_dots.varied_parameters.keys() if k!='repeat']
    print(keys)
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
                                                               column_key=contour_key,
                                                               with_annotation=True,
                                                               with_std=False, ybar=Ybar, ybarlabel='%.1fdF/F'%Ybar, 
                                                               xbar=1, xbarlabel='1s')
        fig.suptitle('static patches\n\n', fontsize=9)
        pdf.savefig(fig);plt.close(fig)
        
        # moving dots
        fig, AX = data.episode_moving_dots.plot_trial_average(roiIndices='mean', 
                                                              column_key=motion_key,
                                                              with_annotation=True,
                                                              with_std=False, ybar=Ybar, ybarlabel='%.1fdF/F'%Ybar, 
                                                              xbar=1, xbarlabel='1s')
        fig.suptitle('moving dots\n\n', fontsize=9)
        pdf.savefig(fig);plt.close(fig)

        # mixed stimuli
        fig, AX = data.episode_mixed.plot_trial_average(roiIndices='mean', 
                                                        column_key=mixed_only_key,
                                                        row_key=motion_key,
                                                        color_key=('patch-%s'%contour_key if (contour_key!='') else ''),
                                                        with_annotation=True,
                                                        with_std=False, ybar=Ybar, ybarlabel='%.1fdF/F'%Ybar, 
                                                        xbar=1, xbarlabel='1s')
        fig.suptitle('mixed static-patch + moving-dots\n\n', fontsize=9)    
        pdf.savefig(fig);plt.close(fig)


        if data.episode_random_dots is not None:
            fig, AX = data.episode_random_dots.plot_trial_average(roiIndices='mean', 
                                                                  with_annotation=True,
                                                                  with_std=False,
                                                                  ybar=Ybar, ybarlabel='%.1fdF/F'%Ybar, 
                                                                  xbar=1, xbarlabel='1s')
            fig.suptitle('random dots\n\n', fontsize=9)    
            pdf.savefig(fig);plt.close(fig)


        if data.episode_mixed_random_dots is not None:
            fig, AX = data.episode_mixed_random_dots.plot_trial_average(roiIndices='mean', 
                                                                        color_key=('patch-%s'%contour_key if (contour_key!='') else ''),
                                                                   with_annotation=True,
                                                                   with_std=False,
                                                                   ybar=Ybar, ybarlabel='%.1fdF/F'%Ybar, 
                                                                   xbar=1, xbarlabel='1s')
            fig.suptitle('mixed static patch + random dots\n\n', fontsize=9)    
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


        for key in [key for key in rois_of_interest_contour_only if 'resp' not in key]:

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
                                                                      column_key=motion_key,
                                                                      with_annotation=True,
                                                                      with_std=False,
                                                                      ybar=Ybar, ybarlabel='%.1fdF/F'%Ybar, 
                                                                      xbar=1, xbarlabel='1s')
                fig.suptitle('moving-dots --> cells resp. to %s\n\n\n' % key, fontsize=8)
                pdf.savefig(fig);plt.close(fig)

                fig, AX = data.episode_mixed.plot_trial_average(roiIndices=rois_of_interest_contour_only[key], 
                                                                column_key=mixed_only_key,
                                                                row_key=motion_key,
                                                                color_key=('patch-%s'%contour_key if (contour_key!='') else ''),
                                                                with_annotation=True,
                                                                with_std=False,
                                                                ybar=Ybar, ybarlabel='%.1fdF/F'%Ybar, 
                                                                xbar=1, xbarlabel='1s')
                fig.suptitle('mixed-stim --> cells resp. to %s\n\n\n' % key, fontsize=8)
                pdf.savefig(fig);plt.close(fig)


                for contour_index, cnt_value in enumerate(contour_keys):
                    for motion_index, mot_value in enumerate(motion_keys):
                        for mixed_index, mix_value in enumerate(data.episode_mixed.varied_parameters[mixed_only_key]):
                            mixed_keys = [mixed_only_key, motion_key, 
                                    'patch-%s'%contour_key if contour_key!='' else ''] 
                            mixed_indices= [mixed_index, motion_index, contour_index]
                            print(mixed_keys)
                            fig, _, _ = interaction_fig(data.get_responses(data.episode_static_patch.find_episode_cond(contour_key, contour_index),
                                                                           data.episode_moving_dots.find_episode_cond(motion_key, motion_index),
                                                                           data.episode_mixed.find_episode_cond(mixed_keys, mixed_indices),
                                                                           roiIndices=rois_of_interest_contour_only[key]),
                                                         static_patch_label='patch\n (%s=%s)' % (contour_key[:3], cnt_value),
                                                         moving_dots_label='mv-dots\n (%s=%s)' % (motion_key[:3], mot_value),
                                                         mixed_label='mixed\n (%s=%s)' % (mixed_only_key.replace('patch-','')[:3], mix_value), 
                                                         Ybar=Ybar)
                            fig.suptitle('interaction --> cells resp. to %s\n\n\n' % key, fontsize=8)
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
