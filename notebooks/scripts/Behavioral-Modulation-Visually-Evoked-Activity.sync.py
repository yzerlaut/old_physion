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
EPISODES = EpisodeResponse(FILES[0],
                           quantities=['dFoF', 'Pupil', 'Running-Speed'],
                           protocol_id=0,
                           verbose=True)
Nep = EPISODES.dFoF.shape[0]

# %% [markdown]
# ## Look for episodes of different behavioral level

# %%
fig, ax = ge.figure(figsize=(1.2,1.5))

threshold = 0.1 # cm/s

running = np.mean(EPISODES.RunningSpeed, axis=1)>threshold

ge.scatter(np.mean(EPISODES.pupilSize, axis=1)[running], 
           np.mean(EPISODES.RunningSpeed, axis=1)[running],
           ax=ax, no_set=True, color=ge.green, ms=5)
ge.scatter(np.mean(EPISODES.pupilSize, axis=1)[~running], 
           np.mean(EPISODES.RunningSpeed, axis=1)[~running],
           ax=ax, no_set=True, color=ge.orange, ms=5)
ge.set_plot(ax, xlabel='pupil size (mm)', ylabel='run. speed (cm/s)')
ax.plot(ax.get_xlim(), threshold*np.ones(2), 'k--', lw=0.5)
ge.annotate(ax, 'n=%i' % np.sum(running), (0,1), va='top', color=ge.green)
ge.annotate(ax, '\nn=%i' % np.sum(~running), (0,1), va='top', color=ge.orange)


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
Nsamples = 10

fig, AX = ge.figure((len(EPISODES.varied_parameters['angle']), Nsamples),
                    figsize=(.8,.9), right=10)

stat_test_props=dict(interval_pre=[-1,0], interval_post=[1,2],
                     test='ttest', positive=True)
response_significance_threshold = 0.01

for i, r in enumerate(np.random.choice(np.arange(EPISODES.data.iscell.sum()), 
                                       Nsamples, replace=False)):
    
    # SHOW trial-average
    EPISODES.plot_trial_average(column_key='angle',
                                condition=EPISODES.find_episode_cond(key='contrast', value=1.) & running,
                                quantity='dFoF',
                                ybar=1., ybarlabel='1dF/F',
                                xbar=1., xbarlabel='1s',
                                roiIndex=r,
                                color=ge.green,
                                AX=[AX[i]], no_set=True)

    # SHOW trial-average
    EPISODES.plot_trial_average(column_key='angle',
                                condition=EPISODES.find_episode_cond(key='contrast', value=1.) & ~running,
                                quantity='dFoF',
                                ybar=1., ybarlabel='1dF/F',
                                xbar=1., xbarlabel='1s',
                                roiIndex=r,
                                color=ge.orange,
                                AX=[AX[i]], no_set=False)

    ge.annotate(AX[i][0], 'roi #%i  ' % (r+1), (0,0), ha='right')
    
    # SHOW summary angle dependence
    inset = ge.inset(AX[i][-1], (2, 0.2, 1.2, 0.8))

    ########################################################################
    ###### NOT RUNNING 
    ########################################################################

    angles, y, sy, responsive_angles = [], [], [], []
    responsive = False
    
    for a, angle in enumerate(EPISODES.varied_parameters['angle']):

        stats = EPISODES.stat_test_for_evoked_responses(episode_cond=EPISODES.find_episode_cond('angle', a) & ~running,
                                                        response_args=dict(quantity='dFoF', roiIndex=r),
                                                        **stat_test_props)
        
        if stats.r!=0:
            angles.append(angle)
            y.append(np.mean(stats.y))    # means "post"
            sy.append(np.std(stats.y))    # std "post"

            if stats.significant(threshold=response_significance_threshold):
                responsive = True
                responsive_angles.append(angle)
            
    ge.plot(angles, np.array(y), sy=np.array(sy), ax=inset,
            axes_args=dict(ylabel='<post dF/F>         ', xlabel='angle ($^{o}$)',
                           xticks=angles, size='small'), 
            m='o', ms=2, lw=1, color=ge.orange, no_set=True)

    ########################################################################
    ###### RUNNING 
    ########################################################################
    
    angles, y, sy, responsive_angles = [], [], [], []
    responsive = False
    
    for a, angle in enumerate(EPISODES.varied_parameters['angle']):

        stats = EPISODES.stat_test_for_evoked_responses(episode_cond=EPISODES.find_episode_cond('angle', a) & running,
                                                        response_args=dict(quantity='dFoF', roiIndex=r),
                                                        **stat_test_props)

        if stats.r!=0:
            angles.append(angle)
            y.append(np.mean(stats.y))    # means "post"
            sy.append(np.std(stats.y))    # std "post"

            if stats.significant(threshold=response_significance_threshold):
                responsive = True
                responsive_angles.append(angle)
            
    ge.plot(angles, np.array(y), sy=np.array(sy), ax=inset,
            axes_args=dict(ylabel='<post dF/F>         ', xlabel='angle ($^{o}$)',
                           xticks=angles, size='small'), 
            m='o', ms=2, lw=1, color=ge.green, no_set=False)
        
    

# %%
