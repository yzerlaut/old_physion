# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Behavioral modulation of direction selectivity

# %%
# general python modules
import sys, os, pprint
import numpy as np
import matplotlib.pylab as plt

# *_-= physion =-_*
physion_folder = os.path.join(os.path.expanduser('~'), 'work', 'physion') # UPDATE to your folder location
# -- physion core
sys.path.append(os.path.join(physion_folder, 'physion'))
from analysis.read_NWB import Data, scan_folder_for_NWBfiles
from dataviz.show_data import MultimodalData, EpisodeResponse
# -- physion data visualization
sys.path.append(os.path.join(physion_folder, 'dataviz', 'datavyz'))
from datavyz import ge


# %%
def selectivity_index(angles, resp):
    """
    computes the selectivity index: (Pref-Orth)/(Pref+Orth)
    clipped in [0,1]
    """
    imax = np.argmax(resp)
    iop = np.argmin(((angles[imax]+90)%(180)-angles)**2)
    if (resp[imax]>0):
        return min([1,max([0,(resp[imax]-resp[iop])/(resp[imax]+resp[iop])])])
    else:
        return 0

def shift_orientation_according_to_pref(angle, 
                                        pref_angle=0, 
                                        start_angle=-45, 
                                        angle_range=360):
    new_angle = (angle-pref_angle)%angle_range
    if new_angle>=angle_range+start_angle:
        return new_angle-angle_range
    else:
        return new_angle



# %%
data_folder =  os.path.join(os.path.expanduser('~'), 'DATA', 'taddy_GluN3KO')
FILES, _, _ = scan_folder_for_NWBfiles(data_folder)
EPISODES = EpisodeResponse(FILES[-1],
                           quantities=['dFoF', 'Pupil', 'Running-Speed'],
                           protocol_id=0,
                           verbose=True)
Nep = EPISODES.dFoF.shape[0]

# %% [markdown]
# ## Look for episodes of different behavioral level

# %%
fig, ax = ge.figure(figsize=(1.2,1.5))

threshold = 0.2 # cm/s

running = np.mean(EPISODES.RunningSpeed, axis=1)>threshold

ge.scatter(np.mean(EPISODES.pupilSize, axis=1)[running], 
           np.mean(EPISODES.RunningSpeed, axis=1)[running],
           ax=ax, no_set=True, color=ge.blue, ms=5)
ge.scatter(np.mean(EPISODES.pupilSize, axis=1)[~running], 
           np.mean(EPISODES.RunningSpeed, axis=1)[~running],
           ax=ax, no_set=True, color=ge.orange, ms=5)
ge.set_plot(ax, xlabel='pupil size (mm)', ylabel='run. speed (cm/s)')
ax.plot(ax.get_xlim(), threshold*np.ones(2), 'k--', lw=0.5)
ge.annotate(ax, 'n=%i' % np.sum(running), (0,1), va='top', color=ge.blue)
ge.annotate(ax, '\nn=%i' % np.sum(~running), (0,1), va='top', color=ge.orange)

# %%
Nsamples = 20 #EPISODES.data.nROIs

fig, AX = ge.figure((len(EPISODES.varied_parameters['angle']), Nsamples),
                    figsize=(.8,.9), right=10)

stat_test_props=dict(interval_pre=[-1,0], interval_post=[1,2],
                     test='ttest', positive=True, verbose=False)
response_significance_threshold = 0.01


responsive_cells = []
for i, r in enumerate(np.arange(EPISODES.data.iscell.sum())):
    # decide whether a cell is visually responsive on all episodes
    responsive = False
    for a, angle in enumerate(EPISODES.varied_parameters['angle']):
        stats = EPISODES.stat_test_for_evoked_responses(episode_cond=EPISODES.find_episode_cond('angle', a),
                                                        response_args=dict(quantity='dFoF', roiIndex=r),
                                                        **stat_test_props)
        if r not in responsive_cells and \
                stats.significant(threshold=response_significance_threshold):
            responsive_cells.append(r) # we just need one responsive angle to turn this True
    
    
for i, r in enumerate(np.random.choice(np.arange(EPISODES.data.iscell.sum()), 
                                       Nsamples, replace=False)):
    
    # STILL trial-average
    EPISODES.plot_trial_average(column_key='angle',
                                condition=EPISODES.find_episode_cond(key='contrast', value=1.) & ~running,
                                quantity='dFoF',
                                ybar=1., ybarlabel='1dF/F',
                                xbar=1., xbarlabel='1s',
                                roiIndex=r,
                                color=ge.orange,
                                AX=[AX[i]], no_set=True, verbose=False)
    
    # RUNNING trial-average
    EPISODES.plot_trial_average(column_key='angle',
                                condition=EPISODES.find_episode_cond(key='contrast', value=1.) & running,
                                quantity='dFoF',
                                ybar=1., ybarlabel='1dF/F',
                                xbar=1., xbarlabel='1s',
                                roiIndex=r,
                                color=ge.blue,
                                AX=[AX[i]], no_set=False,verbose=False)


    ge.annotate(AX[i][0], 'roi #%i  ' % (r+1), (0,0), ha='right')
    
    # SHOW summary angle dependence
    inset = ge.inset(AX[i][-1], (2, 0.2, 1.2, 0.8))
    
    ########################################################################
    ###### RUNNING 
    ########################################################################
    
    angles, y, sy = [], [], []
    
    for a, angle in enumerate(EPISODES.varied_parameters['angle']):

        stats = EPISODES.stat_test_for_evoked_responses(episode_cond=EPISODES.find_episode_cond('angle', a) & running,
                                                        response_args=dict(quantity='dFoF', roiIndex=r),
                                                        **stat_test_props)

        if stats.r!=0:
            angles.append(angle)
            y.append(np.mean(stats.y-stats.x))    
            sy.append(np.std(stats.y-stats.x))  
            
    ge.plot(angles, np.array(y), sy=np.array(sy), ax=inset,
            m='o', ms=2, lw=1, color=ge.blue, no_set=True)
    

    ########################################################################
    ###### STILL
    ########################################################################

    angles, y, sy = [], [], []

    for a, angle in enumerate(EPISODES.varied_parameters['angle']):

        stats = EPISODES.stat_test_for_evoked_responses(episode_cond=EPISODES.find_episode_cond('angle', a) & ~running,
                                                        response_args=dict(quantity='dFoF', roiIndex=r),
                                                        **stat_test_props)
        
        if stats.r!=0:
            angles.append(angle)
            y.append(np.mean(stats.y-stats.x))    # means "delta"
            sy.append(np.std(stats.y-stats.x))    # std "delta"
            
    ge.plot(angles, np.array(y), sy=np.array(sy), ax=inset,
            m='o', ms=2, lw=1, color=ge.orange, no_set=True)


    ge.set_plot(inset, 
                ylabel='< $\Delta$ dF/F>  ', xlabel='angle ($^{o}$)',
                #xticks=angles, 
                size='small')


# %%
stat_test_props=dict(interval_pre=[-1,0], interval_post=[1,2],
                     test='ttest', positive=True)
response_significance_threshold = 0.01

RUN_RESPONSES, STILL_RESPONSES, RESPONSES = [], [], []
shifted_angle = EPISODES.varied_parameters['angle']-EPISODES.varied_parameters['angle'][1]

for roi in responsive_cells:
    
    # ALL    
    # ----------------------------------------
    cell_resp = EPISODES.compute_summary_data(stat_test_props,
                                              response_significance_threshold=response_significance_threshold,
                                              response_args=dict(quantity='dFoF', roiIndex=roi))
    
    condition = np.ones(len(cell_resp['angle']), dtype=bool) # no condition
    # condition = (cell_resp['contrast']==1.) # if specific condition
    
    ipref = np.argmax(cell_resp['value'][condition])
    prefered_angle = cell_resp['angle'][condition][ipref]
    
    RESPONSES.append(np.zeros(len(shifted_angle)))
        
    for a, angle in enumerate(cell_resp['angle'][condition]):
            
        new_angle = shift_orientation_according_to_pref(angle, 
                                                        pref_angle=prefered_angle, 
                                                        start_angle=shifted_angle[0], 
                                                        angle_range=180)
        iangle = np.argwhere(shifted_angle==new_angle)[0][0]
        RESPONSES[-1][iangle] = cell_resp['value'][a]
    
    
    # RUNNING
    # ----------------------------------------
    cell_resp = EPISODES.compute_summary_data(stat_test_props,
                                              episode_cond=running,
                                              response_significance_threshold=response_significance_threshold,
                                              response_args=dict(quantity='dFoF', roiIndex=roi),
                                              verbose=False)
    
    RUN_RESPONSES.append(np.zeros(len(shifted_angle)))
        
    for a, angle in enumerate(cell_resp['angle'][condition]):
            
        new_angle = shift_orientation_according_to_pref(angle, 
                                                        pref_angle=prefered_angle, 
                                                        start_angle=shifted_angle[0], 
                                                        angle_range=180)
        iangle = np.argwhere(shifted_angle==new_angle)[0][0]
        RUN_RESPONSES[-1][iangle] = cell_resp['value'][a]

    # STILL
    # ----------------------------------------
    cell_resp = EPISODES.compute_summary_data(stat_test_props,
                                              episode_cond=~running,
                                              response_significance_threshold=response_significance_threshold,
                                              response_args=dict(quantity='dFoF', roiIndex=roi),
                                              verbose=False)
            
    STILL_RESPONSES.append(np.zeros(len(shifted_angle)))
        
    for a, angle in enumerate(cell_resp['angle'][condition]):
            
        new_angle = shift_orientation_according_to_pref(angle, 
                                                        pref_angle=prefered_angle, 
                                                        start_angle=shifted_angle[0], 
                                                        angle_range=180)
        iangle = np.argwhere(shifted_angle==new_angle)[0][0]
        STILL_RESPONSES[-1][iangle] = cell_resp['value'][a]


# %%
fig, AX = ge.figure(axes=(2,1), wspace=2)
# raw
ge.scatter(shifted_angle, np.mean(RESPONSES, axis=0), 
           sy=np.std(RESPONSES, axis=0), 
           ms=2, lw=0.5, ax=AX[0],
           color=ge.grey, no_set=True)
ge.scatter(shifted_angle, np.mean(STILL_RESPONSES, axis=0), 
           sy=np.std(STILL_RESPONSES, axis=0), 
           ms=3, lw=1, ax=AX[0],
           color=ge.orange, no_set=True)
ge.scatter(shifted_angle, np.mean(RUN_RESPONSES, axis=0), 
           sy=np.std(RUN_RESPONSES, axis=0),
           ms=3, lw=1, ax=AX[0],
           xlabel='angle from pref. ($^o$)', ylabel='$\Delta$F/F', color=ge.blue,
           title='raw resp.')

# peak normalized
N_RESP = [resp/np.max(resp) for resp in RESPONSES]
N_RESP_RUN = [run/np.max(resp) for run, resp in zip(RUN_RESPONSES, RESPONSES)]
N_RESP_STILL = [still/np.max(resp) for still, resp in zip(STILL_RESPONSES, RESPONSES)]
ge.scatter(shifted_angle, np.mean(N_RESP, axis=0), 
           sy=np.std(N_RESP, axis=0),
           ms=2, lw=1, color=ge.grey, no_set=True, ax=AX[1])
ge.scatter(shifted_angle, np.mean(N_RESP_STILL, axis=0), 
           sy=np.std(N_RESP_STILL, axis=0),
           ms=3, lw=1, color=ge.orange, no_set=True, ax=AX[1])
ge.scatter(shifted_angle, np.mean(N_RESP, axis=0), 
           sy=np.std(N_RESP, axis=0),
           ms=3, lw=1, ax=AX[1], axes_args={'yticks':[0,1]}, color=ge.blue,
           xlabel='angle from pref. ($^o$)', ylabel='n. $\Delta$F/F', title='peak normalized')

# %%
