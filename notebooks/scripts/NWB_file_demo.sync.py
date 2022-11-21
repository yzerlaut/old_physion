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
# # Datafiles in the NWB format
#
# We use the NWB data standard to store and share our multimodal neurophysiological recordings:
#
# From the NWB  website https://www.nwb.org/:
#
# > Neurodata Without Borders: Neurophysiology (NWB:N) is a data standard for neurophysiology, providing neuroscientists with a common standard to share, archive, use, and build analysis tools for neurophysiology data. NWB:N is designed to store a variety of neurophysiology data, including data from intracellular and extracellular electrophysiology experiments, data from optical physiology experiments, and tracking and stimulus data.
#
# We rely on the python API to create such files, see:
#
# https://pynwb.readthedocs.io/en/stable/
#
# The script to build such data files is [build_NWB.py](./build_NWB.py) and we update those files (e.g. to add processed data) the script [update_NWB.py](./update_NWB.py)
#
#

# %%
# loading the python NWB API
import pynwb
import os
import numpy as np # numpy for numerical analysis
import matplotlib.pylab as plt # + matplotlib for vizualization

# %%
# loading an example file
#filename = os.path.join(os.path.expanduser('~'), 'DATA', 'tofix', '2022_07_07', '2022_07_07-17-19-14.nwb')
filename = os.path.join(os.path.expanduser('~'), 'DATA', 'JO-VIP-CB1', '2022_10_26-17-08-32.nwb')
io = pynwb.NWBHDF5IO(os.path.join(os.path.expanduser('~'), filename), 'r')
nwbfile = io.read() # don't forget to close afterwards !! (io.close() )

# %%
# let's have a look at what is inside
print(nwbfile)

# %% [markdown]
# ## Acquisition fields

# %%
nwbfile.acquisition

# %% [markdown]
# ### FaceCamera

# %%
fig, AX = plt.subplots(1,3, figsize=(8,3))
for i, ax in zip(np.linspace(0, nwbfile.acquisition['FaceCamera'].data.shape[0]-1, 3, dtype=int), AX):
    ax.imshow(nwbfile.acquisition['FaceCamera'].data[i, :, :])
    ax.axis('off')
    ax.set_title("t=%.1fs" % nwbfile.acquisition['FaceCamera'].timestamps[i])

# %%
nwbfile.processing['Pupil'].data_interfaces['cx']

# %% [markdown]
# ### Pupil

# %%
fig, AX = plt.subplots(1,6, figsize=(16,3))
for i, ax in zip(np.linspace(0, nwbfile.acquisition['Pupil'].data.shape[0]-1, 6, dtype=int), AX):
    ax.imshow(nwbfile.acquisition['Pupil'].data[i, :, :])
    ax.axis('off')
    ax.set_title("t=%.1fs" % nwbfile.acquisition['Pupil'].timestamps[i])

# %%

# %% [markdown]
# ### Calcium-Imaging time series

# %%
nwbfile.acquisition.keys()

# %%
fig, AX = plt.subplots(1,6, figsize=(16,3))
for i, ax in zip(np.linspace(0, nwbfile.acquisition['CaImaging-TimeSeries'].data.shape[0]-1, 6, dtype=int), AX):
    ax.imshow(nwbfile.acquisition['CaImaging-TimeSeries'].data[i, :, :])
    ax.axis('off')
    ax.set_title("t=%.1fs" % nwbfile.acquisition['CaImaging-TimeSeries'].timestamps[i])

# %% [markdown]
# ### Locomotion

# %%
locomotion = nwbfile.acquisition['Running-Speed']
t = np.arange(locomotion.data.shape[0])/locomotion.rate
fig, ax = plt.subplots(1, figsize=(16,3))
ax.plot(t, locomotion.data[:])

# %% [markdown]
# ## Processing fields

# %%
print(nwbfile.processing)

# %% [markdown]
# ### Pupil size and position

# %%
tdata

# %% [markdown]
# ### Optical physiology output

# %%
# "backgrounds_0" key
fig, AX = plt.subplots(1,4, figsize=(16,4))
for key, ax in zip(nwbfile.processing['ophys'].data_interfaces['Backgrounds_0'].images, AX):
    ax.imshow(nwbfile.processing['ophys'].data_interfaces['Backgrounds_0'].images[key][:])
    ax.axis('off')
    ax.set_title(key)
meanImg = nwbfile.processing['ophys'].data_interfaces['Backgrounds_0'].images['meanImg'][:] # we store this

# %%
# Image Segmentation output
#print(nwbfile.processing['ophys'].data_interfaces['ImageSegmentation'].plane_segmentations)

# %%
# fetch quantities
Segmentation = nwbfile.processing['ophys'].data_interfaces['ImageSegmentation'].plane_segmentations['PlaneSegmentation']
pixel_masks_index = Segmentation.columns[0].data[:]
pixel_masks = Segmentation.columns[1].data[:]
iscell = Segmentation.columns[2].data[:,0].astype(bool)

# %%
iscell

# %%
plt.figure(figsize=(10,8))
for i in np.arange(len(iscell))[iscell]:
    indices = np.arange(pixel_masks_index[i],
                        pixel_masks_index[i+1] if (i<len(iscell)-1) else len(pixel_masks))
    x, y = [pixel_masks[ii][1] for ii in indices], [pixel_masks[ii][0] for ii in indices]
    plt.scatter(x, y, color='w', alpha=0.05, s=1)
plt.imshow(meanImg)

# %%
plt.plot(nwbfile.processing['ophys'].data_interfaces['Neuropil'].roi_response_series['Neuropil'].data[:,0])

# %%
# Fluorescence and Neuropil
dt = 1./nwbfile.processing['ophys'].data_interfaces['Neuropil'].roi_response_series['Neuropil'].rate # in s
t = dt*np.arange(nwbfile.processing['ophys'].data_interfaces['Neuropil'].roi_response_series['Neuropil'].data.shape[1])
Fluorescence = nwbfile.processing['ophys'].data_interfaces['Fluorescence'].roi_response_series['Fluorescence']
Neuropil = nwbfile.processing['ophys'].data_interfaces['Neuropil'].roi_response_series['Neuropil']
Deconvolved = nwbfile.processing['ophys'].data_interfaces['Deconvolved'].roi_response_series['Deconvolved']
# we plot just the first 5000 time samples

plt.figure(figsize=(16,30))
for k, i in enumerate(np.arange(len(iscell))[iscell]):
    #fluo_resp = Deconvolved.data[i,:5000] 
    fluo_resp = Fluorescence.data[i,:5000]
    nrp_resp = Neuropil.data[i,:5000]
    plt.plot(t[:5000],k+(fluo_resp-fluo_resp.min())/(fluo_resp.max()-fluo_resp.min()))
    plt.plot(t[:5000],k+(nrp_resp-fluo_resp.min())/(fluo_resp.max()-fluo_resp.min()), 'k', lw=0.3)
plt.plot([0,10], [k+1,k+1], 'k-', lw=2)
plt.annotate('10s', (0,k+1.1), size=14)
plt.axis('off')

# %%
Neuropil = nwbfile.processing['ophys'].data_interfaces['Neuropil'].roi_response_series['Neuropil']
Neuropil.timestamps

# %%
io.close()

# %% [markdown]
# ### Red channel in Ophys

# %%
filename = '/home/yann/UNPROCESSED/2021_06_17/2021_06_17-12-05-31.nwb'
import pynwb
import matplotlib.pylab as plt
import numpy as np
io = pynwb.NWBHDF5IO(filename, 'r')
nwbfile = io.read() # don't forget to close afterwards !! (io.close() )

# "backgrounds_0" key
fig, AX = plt.subplots(1,5, figsize=(16,4))
for key, ax in zip(nwbfile.processing['ophys'].data_interfaces['Backgrounds_0'].images, AX):
    ax.imshow(nwbfile.processing['ophys'].data_interfaces['Backgrounds_0'].images[key][:])
    ax.axis('off')
    ax.set_title(key)
meanImg = nwbfile.processing['ophys'].data_interfaces['Backgrounds_0'].images['meanImg'][:] # we store this
meanImg2 = nwbfile.processing['ophys'].data_interfaces['Backgrounds_0'].images['meanImg_chan2'][:] # we store this

# %%
Segmentation = nwbfile.processing['ophys'].data_interfaces['ImageSegmentation'].plane_segmentations['PlaneSegmentation']
pixel_masks_index = Segmentation.columns[0].data[:]
pixel_masks = Segmentation.columns[1].data[:]
iscell = Segmentation.columns[2].data[:,0].astype(bool)
redcell = Segmentation.columns[3].data[:,0].astype(bool)

# %%
plt.figure(figsize=(10,8))
t = np.linspace(0, 2*np.pi, 50)
for i in np.arange(len(iscell))[redcell]:
    indices = np.arange(pixel_masks_index[i], pixel_masks_index[i+1])
    x, y = [pixel_masks[ii][1] for ii in indices], [pixel_masks[ii][0] for ii in indices]
    plt.scatter(np.mean(x)+np.cos(t)*10,np.mean(y)+np.sin(t)*10, color='r', s=1)
plt.imshow(meanImg2)

# %%
io.close()

# %% [markdown]
# ### Stimulation parameters

# %%

# loading an example file
filename = '/home/yann.zerlaut/DATA/14juin/2022_06_14-18-54-45.nwb'
io = pynwb.NWBHDF5IO(os.path.join(os.path.expanduser('~'), filename), 'r')
nwbfile = io.read() # don't forget to close afterwards !! (io.close() )
print(nwbfile.stimulus)

# %%
nwbfile.stimulus['protocol_id'].data[:]

# %%
nwbfile.stimulus['patch-delay'].data[:]

# %%
nwbfile.stimulus['radius'].data[:]

# %%
