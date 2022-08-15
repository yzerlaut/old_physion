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
# # Orientation and direction selectivity analysis

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


# %% [markdown]
# ## Some useful functions

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



# %% [markdown]
# ## Load and plot raw data

# %%
data_folder =  os.path.join(os.path.expanduser('~'), 'DATA', 'taddy_GluN3KO')
FILES, _, _ = scan_folder_for_NWBfiles(data_folder)

# %%
EPISODES = EpisodeResponse(FILES[0],
                           quantities=['dFoF'],
                           protocol_id=0,
                           verbose=True)

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
                                #condition=EPISODES.find_episode_cond(key='contrast', value=1.),
                                #color_key='contrast',
                                quantity='dF/F',
                                ybar=1., ybarlabel='1dF/F',
                                xbar=1., xbarlabel='1s',
                                roiIndex=r,
                                AX=[AX[i]], no_set=False)
    ge.annotate(AX[i][0], 'roi #%i  ' % (r+1), (0,0), ha='right')
    
    # SHOW summary angle dependence
    inset = ge.inset(AX[i][-1], (2, 0.2, 1.2, 0.8))
    
    angles, y, sy, responsive_angles = [], [], [], []
    responsive = False
    
    for a, angle in enumerate(EPISODES.varied_parameters['angle']):

        stats = EPISODES.stat_test_for_evoked_responses(episode_cond=EPISODES.find_episode_cond('angle', a),
                                                        response_args=dict(quantity='dFoF', roiIndex=r),
                                                        **stat_test_props)
        
        angles.append(angle)
        y.append(np.mean(stats.y))    # means "post"
        sy.append(np.std(stats.y))    # std "post"
        
        if stats.significant(threshold=response_significance_threshold):
            responsive = True
            responsive_angles.append(angle)
            
    ge.plot(angles, np.array(y), sy=np.array(sy), ax=inset,
            axes_args=dict(ylabel='<post dF/F>         ', xlabel='angle ($^{o}$)',
                           xticks=angles, size='small'), 
            m='o', ms=2, lw=1)

    SI = selectivity_index(angles, y)
    ge.annotate(inset, 'SI=%.2f ' % SI, (0, 1), ha='right', weight='bold', fontsize=8,
                color=('k' if responsive else 'lightgray'))
    ge.annotate(inset, ('responsive' if responsive else 'unresponsive'), (1, 1), ha='right',
                weight='bold', fontsize=6, color=(plt.cm.tab10(2) if responsive else plt.cm.tab10(3)))

# %% [markdown]
# ## Summary data

# %%
print(EPISODES.varied_parameters['angle'])
shifted_angle = EPISODES.varied_parameters['angle']-EPISODES.varied_parameters['angle'][1]
print(shifted_angle)

# %%
stat_test_props=dict(interval_pre=[-1,0], interval_post=[1,2],
                     test='ttest', positive=True)
response_significance_threshold = 0.01

RESPONSES = []

for roi in np.arange(EPISODES.data.iscell.sum()):
    
    cell_resp = EPISODES.compute_summary_data(response_significance_threshold=response_significance_threshold,
                                              response_args=dict(quantity='dFoF', roiIndex=roi),
                                              stat_test_props=stat_test_props)
    
    condition = np.ones(len(cell_resp['angle']), dtype=bool) # no condition
    # condition = (cell_resp['contrast']==1.) # if specific condition
    
    if np.sum(cell_resp['significant'][condition]):
        
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

# %%
fig, AX = ge.figure(axes=(2,1), wspace=2)
# raw
ge.scatter(shifted_angle, np.mean(RESPONSES, axis=0), 
           sy=np.std(RESPONSES, axis=0), ms=4, lw=1, ax=AX[0],
           xlabel='angle ($^o$)', ylabel='$\Delta$F/F', title='raw resp.')
# peak normalized
N_RESP = [resp/np.max(resp) for resp in RESPONSES]
ge.scatter(shifted_angle, np.mean(N_RESP, axis=0), sy=np.std(N_RESP, axis=0),
           ms=4, lw=1, ax=AX[1], axes_args={'yticks':[0,1]},
           xlabel='angle ($^o$)', ylabel='n. $\Delta$F/F', title='peak normalized')
