import sys, os, pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from datavyz import graph_env_manuscript as ge

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from physion.analysis.read_NWB import Data
from physion.analysis.process_NWB import EpisodeResponse
from analysis.tools import summary_pdf_folder
from analysis.orientation_direction_selectivity import shift_orientation_according_to_pref

def compute_DS_population_resp(filename, options,
                               protocol_id=0,
                               Nmax = 100000,
                               stat_test_props=dict(interval_pre=[-2,0],
                                                    interval_post=[1,3],
                                                    test='ttest', positive=True),
                               significance_threshold=0.01):

    # load datafile
    data = Data(filename)

    full_resp = {'roi':[], 'angle_from_pref':[], 'Nroi_tot':data.iscell.sum(),
                 'post_level':[], 'evoked_level':[]}

    # get levels of pupil and running-speed in the episodes (i.e. after realignement)
    if 'Pupil' in data.nwbfile.processing:    
        Pupil_episodes = EpisodeResponse(data, protocol_id=protocol_id, quantity='Pupil', **options)
        full_resp['pupil_level'] = []
    else:
        Pupil_episodes = None
    if 'Running-Speed' in data.nwbfile.acquisition:
        Running_episodes = EpisodeResponse(data, protocol_id=protocol_id, quantity='Running-Speed', **options)
        full_resp['speed_level'] = []
    else:
        Running_episodes = None

    if Running_episodes is not None:
        for key in Running_episodes.varied_parameters.keys():
            full_resp[key] = []
    elif Pupil_episodes is not None:
        for key in Pupil_episodes.varied_parameters.keys():
            full_resp[key] = []
    else:
        print(100*'-'+'\n /!\ Need at least one of the Pupil or Running modalities /!\ \n  '+100*'-')


    for roi in np.arange(data.iscell.sum())[:Nmax]:
        ROI_EPISODES = EpisodeResponse(data,
                                       protocol_id=protocol_id,
                                       quantity='CaImaging',
                                       baseline_substraction=True, 
                                       roiIndex = roi, **options)
        # check if significant response in at least one direction and compute mean evoked resp
        resp = {'significant':[], 'pre':[], 'post':[]}
        for ia, angle in enumerate(ROI_EPISODES.varied_parameters['angle']):

            stats = ROI_EPISODES.stat_test_for_evoked_responses(episode_cond=ROI_EPISODES.find_episode_cond('angle', ia),
                                                                **stat_test_props)
            resp['significant'].append(stats.significant(threshold=significance_threshold))
            resp['pre'].append(np.mean(stats.x))
            resp['post'].append(np.mean(stats.y))

        if np.sum(resp['significant'])>0:
            # if significant in at least one
            imax = np.argmax(np.array(resp['post'])-np.array(resp['pre']))
            amax = ROI_EPISODES.varied_parameters['angle'][imax]
            # we compute the post response relative to the preferred orientation for all episodes
            post_interval_cond = ROI_EPISODES.compute_interval_cond(stat_test_props['interval_post'])
            pre_interval_cond = ROI_EPISODES.compute_interval_cond(stat_test_props['interval_pre'])
            for iep, r in enumerate(ROI_EPISODES.resp):
                full_resp['angle_from_pref'].append(shift_orientation_according_to_pref(ROI_EPISODES.angle[iep], amax))
                full_resp['post_level'].append(ROI_EPISODES.resp[iep, post_interval_cond].mean())
                full_resp['evoked_level'].append(full_resp['post_level'][-1]-ROI_EPISODES.resp[iep, pre_interval_cond].mean())
                full_resp['roi'].append(roi)
                # adding running and speed level in the "post" interval:
                if Pupil_episodes is not None:
                    full_resp['pupil_level'].append(Pupil_episodes.resp[iep, post_interval_cond].mean())
                if Running_episodes is not None:
                    full_resp['speed_level'].append(Running_episodes.resp[iep, post_interval_cond].mean())

    # transform to numpy array for convenience
    for key in full_resp:
        full_resp[key] = np.array(full_resp[key])
        
    #########################################################
    ############ per cell analysis ##########################
    #########################################################
    
    angles = np.unique(full_resp['angle_from_pref'])

    full_resp['per_cell'], full_resp['per_cell_post'] = [], []
    
    for roi in np.unique(full_resp['roi']):
        
        roi_cond = (full_resp['roi']==roi)
        
        full_resp['per_cell'].append([])
        full_resp['per_cell_post'].append([])
       
        for ia, angle in enumerate(angles):
            cond = (full_resp['angle_from_pref']==angle) & roi_cond
            full_resp['per_cell'][-1].append(full_resp['evoked_level'][cond].mean())
            full_resp['per_cell_post'][-1].append(full_resp['post_level'][cond].mean())
    
    full_resp['per_cell'] = np.array(full_resp['per_cell'])
    full_resp['per_cell_post'] = np.array(full_resp['per_cell_post'])
        
    return full_resp


def population_tuning_fig(full_resp):

    Ncells = len(np.unique(full_resp['roi']))
    Neps = len(full_resp['roi'])/Ncells   
    angles = np.unique(full_resp['angle_from_pref'])

    fig, AX = ge.figure(axes=(3,1), figsize=(1.5,1.5))
    
    for ax in AX:
        ge.annotate(ax, 'n=%i resp. cells (%.0f%% of rois)' % (Ncells, 
                                                    100.*Ncells/full_resp['Nroi_tot']), (1,1), va='top', ha='right')
    
    ge.plot(angles, np.mean(full_resp['per_cell_post'], axis=0), 
            sy = np.std(full_resp['per_cell_post'], axis=0),
            color='grey', ms=2, m='o', ax=AX[0], no_set=True, lw=1)
    ge.set_plot(AX[0], xlabel='angle ($^{o}$) w.r.t. pref. orient.', ylabel='post level (dF/F)',
               xticks=[0,90,180,270])
    
    ge.plot(angles, np.mean(full_resp['per_cell'], axis=0), 
            sy = np.std(full_resp['per_cell'], axis=0),
            color='grey', ms=2, m='o', ax=AX[1], no_set=True, lw=1)
    ge.set_plot(AX[1], xlabel='angle ($^{o}$) w.r.t. pref. orient.', ylabel='evoked resp. ($\delta$ dF/F)',
               xticks=[0,90,180,270])
    
    ge.plot(angles, np.mean(full_resp['per_cell'].T/np.max(full_resp['per_cell'], axis=1).T, axis=1), 
            sy = np.std(full_resp['per_cell'].T/np.max(full_resp['per_cell'], axis=1).T, axis=1),
            color='grey', ms=2, m='o', ax=AX[2], no_set=True, lw=1)
    ge.set_plot(AX[2], xlabel='angle ($^{o}$) w.r.t. pref. orient.', 
                ylabel='norm. resp ($\delta$ dF/F)', yticks=[0,0.5,1],
               xticks=[0,90,180,270])

    return fig

def compute_behavior_mod_population_tuning(full_resp,
                                           resp_key='evoked_level',
                                           pupil_threshold=2.5,
                                           running_speed_threshold = 0.1):

    Ncells = len(np.unique(full_resp['roi']))
    Neps = len(full_resp['roi'])/Ncells

    if 'speed_level' in full_resp:
        running_cond = (np.abs(full_resp['speed_level'])>=running_speed_threshold)
    else:
        # always a running cond set to False if no running monitoring
        running_cond = np.zeros(len(full_resp['roi']), dtype=bool) # False by default
        
    if 'pupil_level' in full_resp:
        dilated_cond = ~running_cond & (full_resp['pupil_level']>=pupil_threshold)
        constricted_cond = ~running_cond & (full_resp['pupil_level']<pupil_threshold)

    # add to full_resp
    full_resp['Ncells'], full_resp['Neps'] = Ncells, Neps
    full_resp['running_cond'] = running_cond
    if 'pupil_level' in full_resp:
        full_resp['dilated_cond'] = dilated_cond
        full_resp['constricted_cond'] = constricted_cond
            
    angles = np.unique(full_resp['angle_from_pref'])
    curves = {'running_mean': [], 'running_std':[],
              'still_mean': [], 'still_std':[],
              'dilated_mean': [], 'dilated_std':[],
              'constricted_mean': [], 'constricted_std':[],
              'angles':angles}

    for ia, angle in enumerate(angles):
        cond = full_resp['angle_from_pref']==angle
        # running
        curves['running_mean'].append(full_resp[resp_key][cond & running_cond].mean())
        curves['running_std'].append(full_resp[resp_key][cond & running_cond].std())
        # still
        curves['still_mean'].append(full_resp[resp_key][cond & ~running_cond].mean())
        curves['still_std'].append(full_resp[resp_key][cond & ~running_cond].std())
        if 'pupil_level' in full_resp:
            # dilated pupil
            curves['dilated_mean'].append(full_resp[resp_key][cond & dilated_cond].mean())
            curves['dilated_std'].append(full_resp[resp_key][cond & dilated_cond].std())
            # constricted pupil
            curves['constricted_mean'].append(full_resp[resp_key][cond & constricted_cond].mean())
            curves['constricted_std'].append(full_resp[resp_key][cond & constricted_cond].std())
        
    curves['all'] = np.mean(full_resp['per_cell'], axis=0)

    return curves

def tuning_modulation_fig(curves, full_resp=None):

    # running vs still --- raw evoked response
    fig, ax = ge.figure(figsize=(1.5,1.5), right=6)
    ge.plot(curves['angles'], curves['all'], label='all', color='grey', ax=ax, no_set=True, lw=2, alpha=.5)
    ge.plot(curves['angles'], curves['running_mean'], 
            color=ge.orange, ms=4, m='o', ax=ax, lw=2, label='running', no_set=True)
    ge.plot(curves['angles'], curves['still_mean'], 
            color=ge.blue, ms=4, m='o', ax=ax, lw=2, label='still', no_set=True)
    ge.legend(ax, ncol=3, loc=(.3,1.))
    ge.set_plot(ax, xlabel='angle ($^{o}$) w.r.t. pref. orient.', ylabel='evoked resp, ($\delta$ dF/F)   ',
               xticks=[0,90,180,270])

    if (full_resp is not None) and ('speed_level' in full_resp) and ('pupil_level' in full_resp):
        inset = ge.inset(fig, [.8,.5,.16,.28])
        ge.scatter(full_resp['pupil_level'][full_resp['running_cond']],full_resp['speed_level'][full_resp['running_cond']],
                   ax=inset, no_set=True, color=ge.orange)
        ge.scatter(full_resp['pupil_level'][~full_resp['running_cond']],full_resp['speed_level'][~full_resp['running_cond']],
                   ax=inset, no_set=True, color=ge.blue)
        ge.annotate(ax, 'n=%i cells\n' % full_resp['Ncells'], (0.,1.), ha='center')
        ge.set_plot(inset, xlabel='pupil size (mm)', ylabel='run. speed (cm/s)     ',
                   title='episodes (n=%i)   ' % full_resp['Neps'])
        ge.annotate(inset, 'n=%i' % (np.sum(full_resp['running_cond'])/full_resp['Ncells']), (0.,1.), va='top', color=ge.orange)
        ge.annotate(inset, '\nn=%i' % (np.sum(~full_resp['running_cond'])/full_resp['Ncells']), (0.,1.), va='top', color=ge.blue)

    if len(curves['constricted_mean'])>0:
        # constricted vs dilated --- raw evoked response
        fig2, ax = ge.figure(figsize=(1.5,1.5), right=6)
        ge.plot(curves['angles'], curves['all'], label='all', color='grey', ax=ax, no_set=True, lw=2, alpha=.5)
        ge.plot(curves['angles'], curves['constricted_mean'], 
                color=ge.green, ms=4, m='o', ax=ax, lw=2, label='constricted', no_set=True)
        ge.plot(curves['angles'], curves['dilated_mean'], 
                color=ge.purple, ms=4, m='o', ax=ax, lw=2, label='dilated', no_set=True)
        ge.legend(ax, ncol=3, loc=(.1,1.))
        ge.set_plot(ax, xlabel='angle ($^{o}$) w.r.t. pref. orient.', ylabel='evoked resp, ($\delta$ dF/F)   ',
                   xticks=[0,90,180,270])

        if (full_resp is not None) and ('speed_level' in full_resp) and ('pupil_level' in full_resp):
            inset2 = ge.inset(fig2, [.8,.5,.16,.28])
            ge.scatter(full_resp['pupil_level'][full_resp['dilated_cond']],
                       full_resp['speed_level'][full_resp['dilated_cond']],
                       ax=inset2, no_set=True, color=ge.purple)
            ge.scatter(full_resp['pupil_level'][full_resp['constricted_cond']],
                       full_resp['speed_level'][full_resp['constricted_cond']],
                       ax=inset2, no_set=True, color=ge.green)
            ge.annotate(ax, 'n=%i cells\n' % len(np.unique(full_resp['roi'])), (0.,1.), ha='right')
            ge.set_plot(inset2, xlabel='pupil size (mm)', ylabel='run. speed (cm/s)     ', ylim=inset.get_ylim(),
                       title='episodes (n=%i)   ' % full_resp['Neps'])
            ge.annotate(inset2, 'n=%i' % (np.sum(full_resp['constricted_cond'])/full_resp['Ncells']), (0.,1.), va='top', color=ge.green)
            ge.annotate(inset2, '\nn=%i' % (np.sum(full_resp['dilated_cond'])/full_resp['Ncells']), (0.,1.), va='top', color=ge.purple)
    else:
        fig2 = None
    
    return fig, fig2

def analysis_pdf(datafile, iprotocol=0, Nmax=1000000):

    data = Data(datafile)
    
    pdf_filename = os.path.join(summary_pdf_folder(datafile), '%s-behavioral-modulation.pdf' % data.protocols[iprotocol])

    full_resp = compute_population_resp(datafile, protocol_id=iprotocol, Nmax=Nmax)
    
    with PdfPages(pdf_filename) as pdf:
        # print('   - behavioral-modulation analysis for ROI #%i / %i' % (roi+1, data.iscell.sum()))
        fig = population_tuning_fig(full_resp)
        pdf.savefig(fig)  # saves the current figure into a pdf page
        plt.close()
        fig1, fig2 = behavior_mod_population_tuning_fig(full_resp,
                                                        running_speed_threshold = 0.1,
                                                        pupil_threshold=2.5)

        pdf.savefig(fig1)  # saves the current figure into a pdf page
        plt.close(fig1)
        pdf.savefig(fig2)  # saves the current figure into a pdf page
        plt.close(fig2)

    print('[ok] behavioral-modulation analysis saved as: "%s" ' % pdf_filename)


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

    options = dict(subquantity='d(F-0.7*Fneu)',
                   dt_sampling=1, prestim_duration=2, 
                   verbose=False)

    full_resp = compute_DS_population_resp(args.datafile, options, Nmax=args.Nmax)
    if len(full_resp['roi'])>0:
        fig = population_tuning_fig(full_resp)
        curves = compute_behavior_mod_population_tuning(full_resp,
                                                        running_speed_threshold = 0.2,
                                                        pupil_threshold=2.1)
        fig1, fig2 = tuning_modulation_fig(curves, full_resp=full_resp)
    ge.show()










