import sys, os, pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from physion.dataviz.datavyz.datavyz import graph_env_manuscript as ge
from physion.analysis import read_NWB
from physion.analysis import process_NWB 
from physion.dataviz.show_data import EpisodeResponse
from physion.analysis.tools import summary_pdf_folder
# from physion.analysis.protocol_scripts.orientation_direction_selectivity import shift_orientation_according_to_pref


def plot_resp_dependency(Episodes,
                         stim_keys=['Image-ID', 'VSE-seed'],
                         stim_indices=[0,1],
                         responsive_rois=None,
                         running_threshold=0.1,
                         selection_seed=0, N_selected=7):
    """
    """
    # different episode conditions :
    all_eps = Episodes.find_episode_cond(stim_keys, stim_indices)
    running = all_eps & (Episodes.running_speed.mean(axis=1)>running_threshold)
    still = all_eps & (Episodes.running_speed.mean(axis=1)<=running_threshold)

    if responsive_rois is not None:
        np.random.seed(selection_seed)
        if len(responsive_rois)>=N_selected:
            selected_rois=np.random.choice(responsive_rois, N_selected, replace=False)
        else:
            selected_rois = np.concatenate([responsive_rois,
                                            np.random.choice(np.arange(Episodes.dFoF.shape[1]),
                                               N_selected-len(responsive_rois), replace=False)])
    else:
        selected_rois = np.random.choice(np.arange(Episodes.dFoF.shape[1]), N_selected, replace=False)
    selected_rois = np.array(selected_rois, dtype=int)

    fig, AX = ge.figure(axes_extents=[[[1,3] for i in range(4)],
                                     [[1, N_selected] for i in range(4)]],
                        figsize=(1.3,.25), left=1, right=4, top=5, wspace=0.4)


    # RASTER SHOWING RAW VALUES OF dFoF but over a clipped range defined:
    min_dFoF = np.min(Episodes.dFoF[:,selected_rois,:].mean(axis=0).min(axis=-1))
    max_dFoF = 0.8*np.max(Episodes.dFoF[:,selected_rois,:].mean(axis=0).max(axis=-1))

    _, cb = ge.bar_legend(AX[0][0],
                  colorbar_inset=dict(rect=[-0.5,.1,.03,.8], facecolor=None),
                  colormap=ge.binary,
                  bar_legend_args={},
                  #label='n. $\Delta$F/F', bounds=None, ticks = None, ticks_labels=None, no_ticks=False,
                  #label='$\Delta$F/F', bounds=[0,2], ticks = [0,1,2], ticks_labels=['0','1','>2'], no_ticks=False,
                  bounds=[min_dFoF, max_dFoF], ticks=[min_dFoF,max_dFoF],
                  ticks_labels=['<%.1f' % min_dFoF, '>%.1f' % max_dFoF], no_ticks=False,
                  orientation='vertical')
    ge.annotate(cb, '$\Delta$F/F', (1.5, 0.5), rotation=90, va='center')


    ge.bar_legend(AX[1][0],
                  colorbar_inset=dict(rect=[-0.5,.2,.02,.6], facecolor=None),
                  colormap=ge.jet,
                  bar_legend_args={},
                  label='single trials', bounds=None, ticks = None, ticks_labels=None, no_ticks=False,
                  orientation='vertical')

    ##### ---- INSETS ---- #####

    stim_inset = ge.inset(AX[0][3], [0.7, 0.4, 0.9, 1])
    Episodes.visual_stim.plot_stim_picture(np.flatnonzero(all_eps)[0],
                                              ax=stim_inset, vse=True)

    if responsive_rois is not None:
        resp_inset = ge.inset(AX[0][3], [0.95,-0.4,0.5,0.8])
        frac_resp = 100.*len(responsive_rois)/Episodes.dFoF.shape[1]
        ge.pie([frac_resp, 100-frac_resp], COLORS=[ge.green, ge.grey],
               ax=resp_inset)
        ge.annotate(resp_inset, '%.1f%% resp.' % frac_resp, 
                    (1.1,0.9), rotation=90, ha='right',va='top',
                    color=ge.green)


    if hasattr(Episodes, 'pupil_diameter'):
        behav_inset = ge.inset(AX[0][3], [0.2,0.3,0.45,.7])
        Episodes.behavior_variability(episode_condition=all_eps,
                                      threshold2=running_threshold, ax=behav_inset)


    scale_ROIS = np.ones(len(selected_rois))

    for cond, axP, axT, label, color in zip([all_eps, still, running], AX[0], AX[1],
                    ['all eps', 'still', 'running'], ['k', ge.blue, ge.orange]):

        ge.title(axP, '%s (n=%i)' % (label, np.sum(cond)), color=color)

        if np.sum(cond)>0:

            axP.imshow(np.clip(Episodes.dFoF[cond,:,:].mean(axis=0), min_dFoF, max_dFoF),
                       cmap=ge.binary,
                       aspect='auto', interpolation='none',
                       vmin=min_dFoF, vmax=max_dFoF,
                       origin='lower',
                       extent = (Episodes.t[0], Episodes.t[-1],
                                 0, Episodes.dFoF.shape[1]))

            min_dFoF_range = 1.2

            for ir, r in enumerate(selected_rois):

                roi_resp = Episodes.dFoF[cond, r, :]
                scale = max([min_dFoF_range, np.max(roi_resp-roi_resp.mean())]) # 2 dFoF is the min scale range
                # plotting eps with that scale
                for iep in range(np.sum(cond)):
                    axT.plot(Episodes.t, ir+(roi_resp[iep,:]-roi_resp.mean())/scale,
                             color=ge.jet(iep/np.sum(cond)), lw=.5)
                # plotting scale
                axT.plot([Episodes.t[-1], Episodes.t[-1]], [.25+ir, .25+ir+1./scale], 'k-', lw=1.5)

                if 'all' in label:
                    ge.annotate(axT, 'roi#%i ' % (r+1), (Episodes.t[0], ir), xycoords='data',
                                ha='right', size='small')
                    scale_ROIS[ir] = scale

                if label=='running':
                    ge.plot(Episodes.t,
                            ir+Episodes.dFoF[cond, r, :].mean(axis=0)/scale_ROIS[ir],
                            sy=Episodes.dFoF[cond, r, :].std(axis=0)/scale_ROIS[ir],
                            ax=AX[1][3], color=color, no_set=True)

                if label=='still':
                    ge.plot(Episodes.t,
                            ir+Episodes.dFoF[cond, r, :].mean(axis=0)/scale_ROIS[ir],
                            sy=Episodes.dFoF[cond, r, :].std(axis=0)/scale_ROIS[ir],
                            ax=AX[1][3], color=color, no_set=True)

                    AX[1][3].plot([Episodes.t[-1], Episodes.t[-1]], [.25+ir, .25+ir+1./scale], 'k-', lw=1.5)


        ge.annotate(axT, '1$\Delta$F/F', (Episodes.t[-1], 0), xycoords='data',
                    rotation=90)
        ge.set_plot(axT, [], xlim=[Episodes.t[0], Episodes.t[-1]])
        ge.draw_bar_scales(axT, Xbar=1, Xbar_label='1s', Ybar=1e-12)

    ge.set_plot(AX[1][3], [], xlim=[Episodes.t[0], Episodes.t[-1]])
    ge.draw_bar_scales(AX[1][3], Xbar=1, Xbar_label='1s', Ybar=1e-12)

    AX[0][3].axis('off')
    # comparison
    ge.annotate(AX[1][3], '1$\Delta$F/F', (Episodes.t[-1], 0), xycoords='data', rotation=90)

    vse_shifts = Episodes.visual_stim.vse['t'][Episodes.visual_stim.vse['t']<Episodes.visual_stim.protocol['presentation-duration']]

    for ax,ax1 in zip(AX[0][:3], AX[1][:3]):
        ge.set_plot(ax, [], xlim=[Episodes.t[0], Episodes.t[-1]])
        ge.annotate(ax, 'ROIs', (0,0.5), rotation=90, ha='right', va='center')
        ge.annotate(ax, '1', (0,0), ha='right', size='x-small', va='center')
        ge.annotate(ax, '%i' % Episodes.dFoF.shape[1], (0,1), ha='right', size='x-small', va='center')
        for t in [0]+list(vse_shifts)+[Episodes.visual_stim.protocol['presentation-duration']]:
            ax.plot(t*np.ones(2), ax.get_ylim(), 'r--', lw=0.3)
            ax1.plot(t*np.ones(2), ax1.get_ylim(), 'r--', lw=0.3)

    return fig, AX



# def compute_DS_population_resp(filename, options,
                               # protocol_id=0,
                               # Nmax = 100000,
                               # stat_test_props=dict(interval_pre=[-2,0],
                                                    # interval_post=[1,3],
                                                    # test='ttest', positive=True),
                               # significance_threshold=0.01):

    # # load datafile
    # data = Data(filename)
    # Episodes = process_NWB.EpisodeResponse(data, 
                               # quantities=['dFoF', 'Pupil', 'Facemotion', 'Running-Speed'])
                                

    # full_resp = {'roi':[], 'angle_from_pref':[],
                 # 'Nroi_tot':data.iscell.sum(),
                 # 'post_level':[], 'evoked_level':[]}

    # # get levels of pupil and running-speed in the episodes (i.e. after realignement)
    # if 'Pupil' in data.nwbfile.processing:    
        # Pupil_episodes = EpisodeResponse(data, protocol_id=protocol_id, quantity='Pupil', **options)
        # full_resp['pupil_level'] = []
    # else:
        # Pupil_episodes = None
    # if 'Running-Speed' in data.nwbfile.acquisition:
        # Running_episodes = EpisodeResponse(data, protocol_id=protocol_id, quantity='Running-Speed', **options)
        # full_resp['speed_level'] = []
    # else:
        # Running_episodes = None

    # if Running_episodes is not None:
        # for key in Running_episodes.varied_parameters.keys():
            # full_resp[key] = []
    # elif Pupil_episodes is not None:
        # for key in Pupil_episodes.varied_parameters.keys():
            # full_resp[key] = []
    # else:
        # print(100*'-'+'\n /!\ Need at least one of the Pupil or Running modalities /!\ \n  '+100*'-')


    # for roi in np.arange(data.iscell.sum())[:Nmax]:
        # ROI_EPISODES = EpisodeResponse(data,
                                       # protocol_id=protocol_id,
                                       # quantity='CaImaging',
                                       # baseline_substraction=True, 
                                       # roiIndex = roi, **options)
        # # check if significant response in at least one direction and compute mean evoked resp
        # resp = {'significant':[], 'pre':[], 'post':[]}
        # for ia, angle in enumerate(ROI_EPISODES.varied_parameters['angle']):

            # stats = ROI_EPISODES.stat_test_for_evoked_responses(episode_cond=ROI_EPISODES.find_episode_cond('angle', ia),
                                                                # **stat_test_props)
            # resp['significant'].append(stats.significant(threshold=significance_threshold))
            # resp['pre'].append(np.mean(stats.x))
            # resp['post'].append(np.mean(stats.y))

        # if np.sum(resp['significant'])>0:
            # # if significant in at least one
            # imax = np.argmax(np.array(resp['post'])-np.array(resp['pre']))
            # amax = ROI_EPISODES.varied_parameters['angle'][imax]
            # # we compute the post response relative to the preferred orientation for all episodes
            # post_interval_cond = ROI_EPISODES.compute_interval_cond(stat_test_props['interval_post'])
            # pre_interval_cond = ROI_EPISODES.compute_interval_cond(stat_test_props['interval_pre'])
            # for iep, r in enumerate(ROI_EPISODES.resp):
                # full_resp['angle_from_pref'].append(shift_orientation_according_to_pref(ROI_EPISODES.angle[iep], amax))
                # full_resp['post_level'].append(ROI_EPISODES.resp[iep, post_interval_cond].mean())
                # full_resp['evoked_level'].append(full_resp['post_level'][-1]-ROI_EPISODES.resp[iep, pre_interval_cond].mean())
                # full_resp['roi'].append(roi)
                # # adding running and speed level in the "post" interval:
                # if Pupil_episodes is not None:
                    # full_resp['pupil_level'].append(Pupil_episodes.resp[iep, post_interval_cond].mean())
                # if Running_episodes is not None:
                    # full_resp['speed_level'].append(Running_episodes.resp[iep, post_interval_cond].mean())

    # # transform to numpy array for convenience
    # for key in full_resp:
        # full_resp[key] = np.array(full_resp[key])
        
    # #########################################################
    # ############ per cell analysis ##########################
    # #########################################################
    
    # angles = np.unique(full_resp['angle_from_pref'])

    # full_resp['per_cell'], full_resp['per_cell_post'] = [], []
    
    # for roi in np.unique(full_resp['roi']):
        
        # roi_cond = (full_resp['roi']==roi)
        
        # full_resp['per_cell'].append([])
        # full_resp['per_cell_post'].append([])
       
        # for ia, angle in enumerate(angles):
            # cond = (full_resp['angle_from_pref']==angle) & roi_cond
            # full_resp['per_cell'][-1].append(full_resp['evoked_level'][cond].mean())
            # full_resp['per_cell_post'][-1].append(full_resp['post_level'][cond].mean())
    
    # full_resp['per_cell'] = np.array(full_resp['per_cell'])
    # full_resp['per_cell_post'] = np.array(full_resp['per_cell_post'])
        
    # return full_resp


# def population_tuning_fig(full_resp):

    # Ncells = len(np.unique(full_resp['roi']))
    # Neps = len(full_resp['roi'])/Ncells   
    # angles = np.unique(full_resp['angle_from_pref'])

    # fig, AX = ge.figure(axes=(3,1), figsize=(1.5,1.5))
    
    # for ax in AX:
        # ge.annotate(ax, 'n=%i resp. cells (%.0f%% of rois)' % (Ncells, 
                                                    # 100.*Ncells/full_resp['Nroi_tot']), (1,1), va='top', ha='right')
    
    # ge.plot(angles, np.mean(full_resp['per_cell_post'], axis=0), 
            # sy = np.std(full_resp['per_cell_post'], axis=0),
            # color='grey', ms=2, m='o', ax=AX[0], no_set=True, lw=1)
    # ge.set_plot(AX[0], xlabel='angle ($^{o}$) w.r.t. pref. orient.', ylabel='post level (dF/F)',
               # xticks=[0,90,180,270])
    
    # ge.plot(angles, np.mean(full_resp['per_cell'], axis=0), 
            # sy = np.std(full_resp['per_cell'], axis=0),
            # color='grey', ms=2, m='o', ax=AX[1], no_set=True, lw=1)
    # ge.set_plot(AX[1], xlabel='angle ($^{o}$) w.r.t. pref. orient.', ylabel='evoked resp. ($\delta$ dF/F)',
               # xticks=[0,90,180,270])
    
    # ge.plot(angles, np.mean(full_resp['per_cell'].T/np.max(full_resp['per_cell'], axis=1).T, axis=1), 
            # sy = np.std(full_resp['per_cell'].T/np.max(full_resp['per_cell'], axis=1).T, axis=1),
            # color='grey', ms=2, m='o', ax=AX[2], no_set=True, lw=1)
    # ge.set_plot(AX[2], xlabel='angle ($^{o}$) w.r.t. pref. orient.', 
                # ylabel='norm. resp ($\delta$ dF/F)', yticks=[0,0.5,1],
               # xticks=[0,90,180,270])

    # return fig

# def compute_behavior_mod_population_tuning(full_resp,
                                           # resp_key='evoked_level',
                                           # pupil_threshold=2.5,
                                           # running_speed_threshold = 0.1):

    # Ncells = len(np.unique(full_resp['roi']))
    # Neps = len(full_resp['roi'])/Ncells

    # if 'speed_level' in full_resp:
        # running_cond = (np.abs(full_resp['speed_level'])>=running_speed_threshold)
    # else:
        # # always a running cond set to False if no running monitoring
        # running_cond = np.zeros(len(full_resp['roi']), dtype=bool) # False by default
        
    # if 'pupil_level' in full_resp:
        # dilated_cond = ~running_cond & (full_resp['pupil_level']>=pupil_threshold)
        # constricted_cond = ~running_cond & (full_resp['pupil_level']<pupil_threshold)

    # # add to full_resp
    # full_resp['Ncells'], full_resp['Neps'] = Ncells, Neps
    # full_resp['running_cond'] = running_cond
    # if 'pupil_level' in full_resp:
        # full_resp['dilated_cond'] = dilated_cond
        # full_resp['constricted_cond'] = constricted_cond
            
    # angles = np.unique(full_resp['angle_from_pref'])
    # curves = {'running_mean': [], 'running_std':[],
              # 'still_mean': [], 'still_std':[],
              # 'dilated_mean': [], 'dilated_std':[],
              # 'constricted_mean': [], 'constricted_std':[],
              # 'angles':angles}

    # for ia, angle in enumerate(angles):
        # cond = full_resp['angle_from_pref']==angle
        # # running
        # curves['running_mean'].append(full_resp[resp_key][cond & running_cond].mean())
        # curves['running_std'].append(full_resp[resp_key][cond & running_cond].std())
        # # still
        # curves['still_mean'].append(full_resp[resp_key][cond & ~running_cond].mean())
        # curves['still_std'].append(full_resp[resp_key][cond & ~running_cond].std())
        # if 'pupil_level' in full_resp:
            # # dilated pupil
            # curves['dilated_mean'].append(full_resp[resp_key][cond & dilated_cond].mean())
            # curves['dilated_std'].append(full_resp[resp_key][cond & dilated_cond].std())
            # # constricted pupil
            # curves['constricted_mean'].append(full_resp[resp_key][cond & constricted_cond].mean())
            # curves['constricted_std'].append(full_resp[resp_key][cond & constricted_cond].std())
        
    # curves['all'] = np.mean(full_resp['per_cell'], axis=0)

    # return curves

# def tuning_modulation_fig(curves, full_resp=None):

    # # running vs still --- raw evoked response
    # fig, ax = ge.figure(figsize=(1.5,1.5), right=6)
    # ge.plot(curves['angles'], curves['all'], label='all', color='grey', ax=ax, no_set=True, lw=2, alpha=.5)
    # ge.plot(curves['angles'], curves['running_mean'], 
            # color=ge.orange, ms=4, m='o', ax=ax, lw=2, label='running', no_set=True)
    # ge.plot(curves['angles'], curves['still_mean'], 
            # color=ge.blue, ms=4, m='o', ax=ax, lw=2, label='still', no_set=True)
    # ge.legend(ax, ncol=3, loc=(.3,1.))
    # ge.set_plot(ax, xlabel='angle ($^{o}$) w.r.t. pref. orient.', ylabel='evoked resp, ($\delta$ dF/F)   ',
               # xticks=[0,90,180,270])

    # if (full_resp is not None) and ('speed_level' in full_resp) and ('pupil_level' in full_resp):
        # inset = ge.inset(fig, [.8,.5,.16,.28])
        # ge.scatter(full_resp['pupil_level'][full_resp['running_cond']],full_resp['speed_level'][full_resp['running_cond']],
                   # ax=inset, no_set=True, color=ge.orange)
        # ge.scatter(full_resp['pupil_level'][~full_resp['running_cond']],full_resp['speed_level'][~full_resp['running_cond']],
                   # ax=inset, no_set=True, color=ge.blue)
        # ge.annotate(ax, 'n=%i cells\n' % full_resp['Ncells'], (0.,1.), ha='center')
        # ge.set_plot(inset, xlabel='pupil size (mm)', ylabel='run. speed (cm/s)     ',
                   # title='episodes (n=%i)   ' % full_resp['Neps'])
        # ge.annotate(inset, 'n=%i' % (np.sum(full_resp['running_cond'])/full_resp['Ncells']), (0.,1.), va='top', color=ge.orange)
        # ge.annotate(inset, '\nn=%i' % (np.sum(~full_resp['running_cond'])/full_resp['Ncells']), (0.,1.), va='top', color=ge.blue)

    # if len(curves['constricted_mean'])>0:
        # # constricted vs dilated --- raw evoked response
        # fig2, ax = ge.figure(figsize=(1.5,1.5), right=6)
        # ge.plot(curves['angles'], curves['all'], label='all', color='grey', ax=ax, no_set=True, lw=2, alpha=.5)
        # ge.plot(curves['angles'], curves['constricted_mean'], 
                # color=ge.green, ms=4, m='o', ax=ax, lw=2, label='constricted', no_set=True)
        # ge.plot(curves['angles'], curves['dilated_mean'], 
                # color=ge.purple, ms=4, m='o', ax=ax, lw=2, label='dilated', no_set=True)
        # ge.legend(ax, ncol=3, loc=(.1,1.))
        # ge.set_plot(ax, xlabel='angle ($^{o}$) w.r.t. pref. orient.', ylabel='evoked resp, ($\delta$ dF/F)   ',
                   # xticks=[0,90,180,270])

        # if (full_resp is not None) and ('speed_level' in full_resp) and ('pupil_level' in full_resp):
            # inset2 = ge.inset(fig2, [.8,.5,.16,.28])
            # ge.scatter(full_resp['pupil_level'][full_resp['dilated_cond']],
                       # full_resp['speed_level'][full_resp['dilated_cond']],
                       # ax=inset2, no_set=True, color=ge.purple)
            # ge.scatter(full_resp['pupil_level'][full_resp['constricted_cond']],
                       # full_resp['speed_level'][full_resp['constricted_cond']],
                       # ax=inset2, no_set=True, color=ge.green)
            # ge.annotate(ax, 'n=%i cells\n' % len(np.unique(full_resp['roi'])), (0.,1.), ha='right')
            # ge.set_plot(inset2, xlabel='pupil size (mm)', ylabel='run. speed (cm/s)     ', ylim=inset.get_ylim(),
                       # title='episodes (n=%i)   ' % full_resp['Neps'])
            # ge.annotate(inset2, 'n=%i' % (np.sum(full_resp['constricted_cond'])/full_resp['Ncells']), (0.,1.), va='top', color=ge.green)
            # ge.annotate(inset2, '\nn=%i' % (np.sum(full_resp['dilated_cond'])/full_resp['Ncells']), (0.,1.), va='top', color=ge.purple)
    # else:
        # fig2 = None
    
    # return fig, fig2

# def analysis_pdf(datafile, iprotocol=0, Nmax=1000000):

    # data = Data(datafile)
    
    # pdf_filename = os.path.join(summary_pdf_folder(datafile), '%s-behavioral-modulation.pdf' % data.protocols[iprotocol])

    # full_resp = compute_population_resp(datafile, protocol_id=iprotocol, Nmax=Nmax)
    
    # with PdfPages(pdf_filename) as pdf:
        # # print('   - behavioral-modulation analysis for ROI #%i / %i' % (roi+1, data.iscell.sum()))
        # fig = population_tuning_fig(full_resp)
        # pdf.savefig(fig)  # saves the current figure into a pdf page
        # plt.close()
        # fig1, fig2 = behavior_mod_population_tuning_fig(full_resp,
                                                        # running_speed_threshold = 0.1,
                                                        # pupil_threshold=2.5)

        # pdf.savefig(fig1)  # saves the current figure into a pdf page
        # plt.close(fig1)
        # pdf.savefig(fig2)  # saves the current figure into a pdf page
        # plt.close(fig2)

    # print('[ok] behavioral-modulation analysis saved as: "%s" ' % pdf_filename)


if __name__=='__main__':
    
    import argparse

    parser=argparse.ArgumentParser()
    parser.add_argument("datafile", type=str)
    parser.add_argument("--iprotocol", type=int, default=0, help='index for the protocol in case of multiprotocol in datafile')
    parser.add_argument('-nmax', "--Nmax", type=int, default=1000000)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    # if '.nwb' in args.datafile:
    #     analysis_pdf(args.datafile, iprotocol=args.iprotocol, Nmax=args.Nmax)
    # else:
    #     print('/!\ Need to provide a NWB datafile as argument ')

    # options = dict(subquantity='d(F-0.7*Fneu)',
                   # dt_sampling=1, prestim_duration=2, 
                   # verbose=False)

    # full_resp = compute_DS_population_resp(args.datafile, options, Nmax=args.Nmax)
    # if len(full_resp['roi'])>0:
        # fig = population_tuning_fig(full_resp)
        # curves = compute_behavior_mod_population_tuning(full_resp,
                                                        # running_speed_threshold = 0.2,
                                                        # pupil_threshold=2.1)
        # fig1, fig2 = tuning_modulation_fig(curves, full_resp=full_resp)

    # load data
    Episodes_NI = EpisodeResponse(args.datafile,
                                  protocol_id=args.iprotocol,
                                  quantities=['dFoF', 'Pupil', 'Running-Speed'],
                                  dt_sampling=30, # ms, to avoid to consume to much memory
                                  verbose=True, prestim_duration=1.5)

    ROI_SUMMARIES = [Episodes_NI.compute_summary_data(dict(\
                                interval_pre=[-Episodes_NI.visual_stim.protocol['presentation-interstim-period'],0],
                                interval_post=[0,Episodes_NI.visual_stim.protocol['presentation-duration']],
                                test='wilcoxon', positive=True),
                         response_args={'quantity':'dFoF', 
                                        'roiIndex':roi},
                         response_significance_threshold=0.01) for roi in range(Episodes_NI.dFoF.shape[1])]

    # plot
    plot_resp_dependency(Episodes_NI, 
                         running_threshold=0.5,
                         N_selected=20, selection_seed=20)
    ge.show()










