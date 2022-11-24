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
# # Build a session summary file for a given experiment

# %%
# general python modules
import sys, os, pprint, datetime, tempfile
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
from datavyz import graph_env
ge = graph_env('manuscript')

# %%
filename = os.path.join(os.path.expanduser('~'), 'DATA', 'JO-VIP-CB1', 'Assembled', '2022_10_26-18-13-13.nwb')

# %% [markdown]
# ## Metadata

# %%
# let's pick the first one
data = Data(filename)


# %%
def metadata_fig(data):

    metadata = dict(data.metadata)

    string = """
    Datafile: "%s",                                                  analyzed on: %s
    Mouse ID: "%s" """ % (os.path.basename(data.filename), datetime.date.today(), data.metadata['subject_ID'])
    i=1
    while 'Protocol-%i'%i in data.metadata:
        string += '\n          -  Protocol-%i: %s' % (i, data.metadata['Protocol-%i'%i])
        i+=1
    # write on fig
    fig, ax = ge.figure(figsize=(5,1), left=0)
    ge.annotate(ax, string, (0,0), size='small')
    ax.axis('off')
    return fig, ax


fig, ax = metadata_fig(data)
fig.savefig(os.path.join(tempfile.tempdir, 'metadata.png'), dpi=300)

# %%
data = MultimodalData(filename)

# %% [markdown]
# ## FOVs and ROIs

# %%
fig, ax = ge.figure(figsize=(4,10))
data.show_CaImaging_FOV(key='meanImg', ax=ax, NL=4, with_annotation=False,
                        roiIndices=np.arange(data.nROIs));

# %%
fig, AX = ge.figure(axes=(3,1), 
                    figsize=(0.85,1),
                    wspace=0.1, bottom=0, left=0, right=0, top=0.6)

data.show_CaImaging_FOV(key='meanImg', ax=AX[0], NL=4, with_annotation=False)
data.show_CaImaging_FOV(key='max_proj', ax=AX[1], NL=3, with_annotation=False)
data.show_CaImaging_FOV(key='meanImg', ax=AX[2], NL=4, with_annotation=False,
                        roiIndices=np.arange(data.nROIs))
for ax, title in zip(AX, ['meanImg', 'max_proj', 'n=%iROIs' % data.nROIs]):
    ax.set_title(title, fontsize=6)
fig.savefig(os.path.join(tempfile.tempdir, 'FOV.png'), dpi=300)

# %% [markdown]
# ## Raw data plot

# %%
fig, ax = ge.figure(figsize=(2.6,3.2), bottom=0, top=0.2, left=0.3, right=1.7)

_, ax = data.plot_raw_data(data.tlim, 
                  settings={'Locomotion':dict(fig_fraction=1, subsampling=2, color=ge.blue),
                            'FaceMotion':dict(fig_fraction=1, subsampling=2, color=ge.purple),
                            'Pupil':dict(fig_fraction=1, subsampling=2, color=ge.red),
                            'CaImaging':dict(fig_fraction=4, subsampling=2, 
                                             subquantity='dF/F', color=ge.green,
                                             roiIndices=np.random.choice(data.nROIs,5)),
                            'CaImagingRaster':dict(fig_fraction=2, subsampling=4,
                                                   roiIndices='all',
                                                   normalization='per-line',
                                                   subquantity='dF/F')},
                            Tbar=120, ax=ax)

ax.annotate('full recording: %.1fmin  ' % ((data.tlim[1]-data.tlim[0])/60), (1,1), 
             ha='right', xycoords='axes fraction', size=8)

fig.savefig(os.path.join(tempfile.tempdir, 'raw0.png'), dpi=300)

# %% [markdown]
# ## Showing the visual stimulation

# %%
print(data.protocols)

# %%
# fetching the grey screen protocol time
igrey = np.flatnonzero(data.nwbfile.stimulus['protocol_id'].data[:]==1) # grey
tgrey = data.nwbfile.stimulus['time_start_realigned'].data[igrey]+\
                data.nwbfile.stimulus['time_duration'].data[igrey]/2.

# fetching the black screen protocol time
iblank = np.flatnonzero(data.nwbfile.stimulus['protocol_id'].data[:]==2) # blank
tblank = data.nwbfile.stimulus['time_start_realigned'].data[iblank]+\
                data.nwbfile.stimulus['time_duration'].data[iblank]/2.

# %%
fig, ax = ge.figure(figsize=(2.6,3.2), bottom=0, top=0.2, left=0.3, right=1.7)

tlim = [50, 900]
_, ax = data.plot_raw_data(tlim, 
                  settings={'Locomotion':dict(fig_fraction=1, subsampling=1, color=ge.blue),
                            'FaceMotion':dict(fig_fraction=1, subsampling=1, color=ge.purple),
                            'Pupil':dict(fig_fraction=1, subsampling=1, color=ge.red),
                            'CaImaging':dict(fig_fraction=4, subsampling=1, 
                                             subquantity='dF/F', color=ge.green,
                                             roiIndices=np.random.choice(data.nROIs,5)),
                            'CaImagingRaster':dict(fig_fraction=2, subsampling=1,
                                                   roiIndices='all',
                                                   normalization='per-line',
                                                   subquantity='dF/F'),
                            'VisualStim':dict(fig_fraction=0, color='black',
                                              with_screen_inset=False)},
                            Tbar=60, ax=ax)
ge.annotate(ax, 'grey screen', (tgrey, 1.02), xycoords='data', ha='center', va='bottom', italic=True)
ge.annotate(ax, 'black screen', (tblank, 1.02), xycoords='data', ha='center', va='bottom', italic=True)

fig.savefig(os.path.join(tempfile.tempdir, 'raw1.png'), dpi=300)

# %%
tlim = [15, 35]

fig, ax = ge.figure(figsize=(2.6,3.2), bottom=0, top=0.2, left=0.3, right=1.7)

_, ax = data.plot_raw_data(tlim, 
                  settings={'Photodiode':dict(fig_fraction=1, subsampling=1, color=ge.gray),
                            'Locomotion':dict(fig_fraction=1, subsampling=1, color=ge.blue),
                            'FaceMotion':dict(fig_fraction=1, subsampling=1, color=ge.purple),
                            'Pupil':dict(fig_fraction=1, subsampling=1, color=ge.red),
                            'CaImaging':dict(fig_fraction=4, subsampling=1, 
                                             subquantity='dF/F', color=ge.green,
                                             roiIndices=np.random.choice(data.nROIs,5)),
                            'CaImagingRaster':dict(fig_fraction=2, subsampling=1,
                                                   roiIndices='all',
                                                   normalization='per-line',
                                                   subquantity='dF/F'),
                            'VisualStim':dict(fig_fraction=0, color='black',
                                              with_screen_inset=True)},
                            Tbar=1, ax=ax)

fig.savefig(os.path.join(tempfile.tempdir, 'raw2.png'), dpi=300)

# %%
tlim = [1530, 1595]

fig, ax = ge.figure(figsize=(2.6,3.2), bottom=0, top=0.2, left=0.3, right=1.7)
_, ax = data.plot_raw_data(tlim, 
                  settings={'Photodiode':dict(fig_fraction=1, subsampling=1, color=ge.gray),
                            'Locomotion':dict(fig_fraction=1, subsampling=1, color=ge.blue),
                            'FaceMotion':dict(fig_fraction=1, subsampling=1, color=ge.purple),
                            'Pupil':dict(fig_fraction=1, subsampling=1, color=ge.red),
                            'CaImaging':dict(fig_fraction=4, subsampling=1, 
                                             subquantity='dF/F', color=ge.green,
                                             roiIndices=np.random.choice(data.nROIs,5)),
                            'CaImagingRaster':dict(fig_fraction=2, subsampling=1,
                                                   roiIndices='all',
                                                   normalization='per-line',
                                                   subquantity='dF/F'),
                            'VisualStim':dict(fig_fraction=0, color='black',
                                              with_screen_inset=True)},
                            Tbar=2, ax=ax)

fig.savefig(os.path.join(tempfile.tempdir, 'raw3.png'), dpi=300)

# %% [markdown]
# ## Activity under different light conditions

# %%
data.build_dFoF()

# %%
# we take 10 second security around each
tfull_wStim_start = 10

# fetching the grey screen protocol interval
igrey = np.flatnonzero(data.nwbfile.stimulus['protocol_id'].data[:]==1) # grey
tgrey_start = 10+data.nwbfile.stimulus['time_start_realigned'].data[igrey][0]
tgrey_stop = tgrey_start-10+data.nwbfile.stimulus['time_duration'].data[igrey][0]

# fetching the black screen protocol interval
iblank = np.flatnonzero(data.nwbfile.stimulus['protocol_id'].data[:]==2) # blank
tblank_start = 10+data.nwbfile.stimulus['time_start_realigned'].data[iblank][0]
tblank_stop = tblank_start-10+data.nwbfile.stimulus['time_duration'].data[iblank][0]

# fetching the interval with visual stimulation after the last blank
tStim_start = tblank_stop+10
tStim_stop = tStim_start + data.nwbfile.stimulus['time_duration'].data[iblank][0] # same length

# %%
RESP = {}

from scipy.stats import skew

for key, interval in zip(['black', 'grey', 'wStim'],
                         [(tblank_start, tblank_stop),
                          (tgrey_start, tgrey_stop),
                          (tStim_start, tStim_stop)]):
    
    time_cond = (data.t_dFoF>interval[0]) & (data.t_dFoF<interval[1])
    RESP[key+'-mean'], RESP[key+'-std'], RESP[key+'-skew'] = [], [], []
    for roi in range(data.nROIs):
        RESP[key+'-mean'].append(data.dFoF[roi,time_cond].mean())
        RESP[key+'-std'].append(data.dFoF[roi,time_cond].std())
        RESP[key+'-skew'].append(skew(data.dFoF[roi,time_cond]))
        
for key in RESP:
    RESP[key] = np.array(RESP[key])

# %%
fig, [ax1, ax2, ax3] = ge.figure(axes=(3,1), figsize=(1.1,1.1), wspace=1.5)

COLORS = ['k', 'grey', 'lightgray']
for i, key in enumerate(['black', 'grey', 'wStim']):
    
    parts = ax1.violinplot([RESP[key+'-mean']], [i], showextrema=False, showmedians=False)#, color=COLORS[i])
    parts['bodies'][0].set_facecolor(COLORS[i])
    parts['bodies'][0].set_alpha(1)
    ax1.plot([i], [np.median(RESP[key+'-mean'])], 'r_')
    
    parts = ax2.violinplot([RESP[key+'-mean']/RESP['black-mean']], [i], 
                           showextrema=False, showmedians=False)#, color=COLORS[i])
    parts['bodies'][0].set_facecolor(COLORS[i])
    parts['bodies'][0].set_alpha(1)
    ax2.plot([i], [np.mean(RESP[key+'-mean']/RESP['black-mean'])], 'r_')

    parts = ax3.violinplot([RESP[key+'-skew']], [i], showextrema=False, showmedians=False)#, color=COLORS[i])
    parts['bodies'][0].set_facecolor(COLORS[i])
    parts['bodies'][0].set_alpha(1)
    ax3.plot([i], [np.median(RESP[key+'-skew'])], 'r_')
    
ge.set_plot(ax1, ylabel='mean $\Delta$F/F', ylim=[0.2,1.8], 
            xticks=range(3), xticks_labels=['black', 'grey', 'wStim'], xticks_rotation=50)
ge.set_plot(ax2, ylabel='mean $\Delta$F/F\n norm. to "black"',
            xticks=range(3), xticks_labels=['black', 'grey', 'wStim'], xticks_rotation=50)
ge.set_plot(ax3, ylabel='$\Delta$F/F skewness',
            xticks=range(3), xticks_labels=['black', 'grey', 'wStim'], xticks_rotation=50)

fig.savefig(os.path.join(tempfile.tempdir, 'light-cond.png'), dpi=300)

# %% [markdown]
# ## Trial-average data

# %%
episodes = EpisodeResponse(filename,
                           protocol_id=data.get_protocol_id('ff-drifiting-gratings-4orientation-5contrasts-log-spaced-10repeats'),
                           quantities=['dFoF'],
                           prestim_duration=3,
                           verbose=True)

# %%
SIGNIFICANT_ROIS = []
for roi in range(data.nROIs):
    summary_data = episodes.compute_summary_data(dict(interval_pre=[0,1],
                                                      interval_post=[1,2],
                                                      test='anova',
                                                      positive=True),
                                                      response_args={'quantity':'dFoF', 'roiIndex':roi},
                                                      response_significance_threshold=0.01)
    if np.sum(summary_data['significant'])>0:
        SIGNIFICANT_ROIS.append(roi)

# %%
X = [100*len(SIGNIFICANT_ROIS)/data.nROIs,100-100*len(SIGNIFICANT_ROIS)/data.nROIs]
fig, ax = ge.pie(X,
       ext_labels=['responsive\n%.1f%%  (n=%i)'%(X[0], len(SIGNIFICANT_ROIS)), 'non  \nresp.'],
       COLORS=[ge.green, ge.grey],
               fig_args=dict(figsize=(1.2,1.2), bottom=0, right=4, left=0.4, top=0.8))
ge.title(ax, 'drifting grating stim.')
fig.savefig(os.path.join(tempfile.tempdir, 'resp-fraction.png'), dpi=300)

# %%
fig, _ = episodes.plot_trial_average(column_key='contrast', 
                                     xbar=1, xbarlabel='1s', 
                                     ybar=0.4, ybarlabel='0.4$\Delta$F/F',
                                     row_key='angle', 
                                     with_screen_inset=True,
                                     with_std_over_rois=True, 
                                     with_annotation=True)
ge.annotate(fig, 'response average (s.d. over all ROIs)\n', (0.5, 0), ha='center')
fig.savefig(os.path.join(tempfile.tempdir, 'TA-all.png'), dpi=300)

# %%
for i, roi in enumerate(SIGNIFICANT_ROIS):
    fig, _ = episodes.plot_trial_average(column_key='contrast', roiIndex=roi,
                                         color_key='angle', 
                                         xbar=1, xbarlabel='1s', 
                                         ybar=1, ybarlabel='1$\Delta$F/F',
                                         with_std=True, with_annotation=True)
    
    ge.annotate(fig, 'example %i: responsive ROI' % (i+1), (0.5, 0.2), ha='center', size='small')
    fig.savefig(os.path.join(tempfile.tempdir, 'TA-%i.png' % roi), dpi=300)

# %% [markdown]
# ## Session Summary PDF

# %% [markdown]
# ### Page 1 - Raw Data

# %%
# -- PIL to build a session summary pdf
from PIL import Image
width, height = int(8.27 * 300), int(11.7 * 300) # A4 at 300dpi : (2481, 3510)

# let's create the A4 page
page = Image.new('RGB', (width, height), 'white')

KEYS = ['metadata',
        'raw0', 'raw1', 'raw2', 'raw3',
        'FOV']

LOCS = [(200, 70),
        (150, 550), (150, 1300), (150, 2000), (150, 2700),
        (1000, 330)]
for key, loc in zip(KEYS, LOCS):
    
    fig = Image.open(os.path.join(tempfile.tempdir, '%s.png' % key))
    page.paste(fig, box=loc)
    fig.close()

page.save(os.path.join(os.path.expanduser('~'), 'Desktop', 'session-summary-1.pdf'))

# %% [markdown]
# ### Page 2 - Analysis

# %%
# -- PIL to build a session summary pdf
from PIL import Image
width, height = int(8.27 * 300), int(11.7 * 300) # A4 at 300dpi : (2481, 3510)

# let's create the A4 page
page = Image.new('RGB', (width, height), 'white')

KEYS = ['light-cond', 'resp-fraction',
        'TA-all', 
        'TA-14', 'TA-63', 'TA-77']

LOCS = [(400, 70), (1000, 550),
        (150, 900),
        (150, 2250), (150, 2600), (150, 2950)]
for key, loc in zip(KEYS, LOCS):
    
    fig = Image.open(os.path.join(tempfile.tempdir, '%s.png' % key))
    page.paste(fig, box=loc)
    fig.close()

page.save(os.path.join(os.path.expanduser('~'), 'Desktop', 'session-summary-2.pdf'))

# %%
