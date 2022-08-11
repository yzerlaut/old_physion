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
# # Preprocessing of Calcium signals

# %%
# general modules
import pynwb, os, sys
import numpy as np
import matplotlib.pylab as plt

# custom modules
sys.path.append('..')
from physion.dataviz.show_data import MultimodalData

# %% [markdown]
# ## Loading a datafile with Calcium Imaging data

# %%
filename = '/home/yann/DATA/CaImaging/PSD-95_GCamp6s/2021_04_29/2021_04_29-15-46-23.nwb'
data = MultimodalData(filename)
fig, ax = plt.subplots(1, figsize=(6,6))
_ = data.show_CaImaging_FOV(key='meanImg', NL=2, ax=ax)

# %%
fig, ax = plt.subplots(1, figsize=(15,5))
_ = data.plot_raw_data([50.3, 251], 
              settings={'Photodiode':dict(fig_fraction=.1, subsampling=10, color='grey'),
                        'Locomotion':dict(fig_fraction=1, subsampling=10, color='b'),
                        'CaImaging':dict(fig_fraction=4, 
                                         quantity='CaImaging', subquantity='Neuropil', color='green',
                                                   roiIndices=[2, 6, 9, 10, 13, 15, 16, 17, 38, 41]),
                        'VisualStim':dict(fig_fraction=0, size=0.04, color='black')},                    
              Tbar=2, ax=ax)

# %% [markdown]
# ## Looking at the raw ROI fluorescence and neuropil signals

# %%
from physion.analysis.read_NWB import Data
data = Data(filename)

for ROI in range(10)[::2]:
    fig, ax = plt.subplots(figsize=(15,1.5))
    F = data.Fluorescence.data[data.validROI_indices[ROI],:]
    Neuropil = data.Neuropil.data[data.validROI_indices[ROI],:]
    plt.plot(data.Fluorescence.timestamps[:], Neuropil, label='Neuropil', color='orange')
    plt.plot(data.Fluorescence.timestamps[:], F, label='ROI Fluo.')
    plt.title('ROI#%i' % (ROI+1))
    plt.legend()
    plt.ylabel('Fluo. (a.u.)')
plt.xlabel('time (s)');

# %% [markdown]
# ## Computing the $\Delta$F/F
#
# - we evaluate the baseline signal with a sliding percentile, that we smooth

# %%
from scipy.ndimage import filters
from physion.Ca_imaging.tools import sliding_percentile 

data = Data(filename)

sliding_window = int(120./data.CaImaging_dt) # 2 min sliding window
percentile = 5 # 5th percentile
for ROI in range(10)[::2]:
    fig, [ax,ax2] = plt.subplots(2,figsize=(15,2))
    F = data.Fluorescence.data[data.validROI_indices[ROI],:]
    Fmin = sliding_percentile(F, percentile, sliding_window) # sliding percentile
    Fmin = filters.gaussian_filter1d(Fmin, sliding_window) # + smoothing
    DFoF = (F-Fmin)/Fmin
    ax.plot(data.Fluorescence.timestamps[:], F, label='F')
    ax2.plot(data.Fluorescence.timestamps[:], DFoF, color='g')
    ax.plot(data.Fluorescence.timestamps[:], Fmin, label='sliding min')
    ax.set_title('ROI#%i' % (ROI+1))
    ax.legend()
    ax.set_ylabel('Fluo. (a.u.)')
    ax2.set_ylabel('$\Delta$F/F')
plt.xlabel('time (s)');

# %% [markdown]
# #### Implementation in the module: "compute_CaImaging_trace" function

# %%
from physion.Ca_imaging.tools import compute_CaImaging_trace

data = Data(filename)

for ROI in range(10)[::2]:
    fig, [ax,ax2] = plt.subplots(2,figsize=(15,2))
    F = data.Fluorescence.data[data.validROI_indices[ROI],:]
    DFoF, Fmin = compute_CaImaging_trace(data, 'dF/F', [ROI],
                                         Tsliding=120, percentile=5., with_baseline=True)
    ax.plot(data.Fluorescence.timestamps[:], F, label='F')
    ax2.plot(data.Fluorescence.timestamps[:], DFoF.mean(axis=0), color='g')
    ax.plot(data.Fluorescence.timestamps[:], Fmin.mean(axis=0), label='sliding min')
    ax.set_title('ROI#%i' % (ROI+1))
    ax.legend()
    ax.set_ylabel('Fluo. (a.u.)')
    ax2.set_ylabel('$\Delta$F/F')
plt.xlabel('time (s)');

# %% [markdown]
# ## Deconvolution

# %%
from suite2p.extraction.dcnv import oasis # NEED THE DECONVOLUTION CODE FROM SUITE2P
Tau = 0.7
Factor=0.85
Dcnv = oasis(data.Fluorescence.data[:]-Factor*data.Neuropil.data[:], 5000, 1./data.CaImaging_dt, Tau)

for ROI in range(10)[::2]:
    fig, [ax,ax2] = plt.subplots(2, figsize=(15,1.5))
    ax.plot(data.Fluorescence.timestamps[:], data.Fluorescence.data[ROI,:], color='green')
    ax2.plot(data.Fluorescence.timestamps[:], Dcnv[ROI,:], label='custom')
    ax2.plot(data.Fluorescence.timestamps[:], data.Deconvolved.data[ROI,:], label='suite2p defaults')
    ax.set_ylabel('Fluo. (a.u.)')
    ax2.set_ylabel('Deconv.')
    ax2.legend()

# %%
data.metadata['subject_props']['virus']

# %% [markdown]
# #### Implementation in the module: "compute_CaImaging_trace" function

# %%
