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

# %%
bug = ('keep this line for having jupyter_ascending.vim working')

# %% [markdown]
# # Notebook template

# %% [markdown]
# ## Load all required modules

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
# ## Inspect a data folder

# %%
data_folder =  os.path.join(os.path.expanduser('~'), 'DATA', 'taddy_GluN3KO')
FILES, _, _ = scan_folder_for_NWBfiles(data_folder) 
print(FILES)

# %% [markdown]
# ## Load a datafile

# %%
# let's pick the first one
data = Data(FILES[0])

# %%
# the different protocols are listed in data.protocols
for i, protocol in enumerate(data.protocols):
    print('  - protocol #%i: "%s"' % (i+1, protocol))

# %%
# metadata in the data.metadata dictionary
pprint.pprint(data.metadata)
#print(data.metadata.keys())

# %% [markdown]
# ## Plot the raw data

# %%
mdata = MultimodalData(FILES[0])

# %%
mdata.plot_raw_data()

# %% [markdown]
# # Plot the trial average|

# %%
episodes = EpisodeResponse(FILES[0],
                           protocol_id=data.get_protocol_id('ff-gratings-8orientation-2contrasts-10repeats'),
                           quantities=['dFoF'],
                           prestim_duration=3,
                           verbose=True)

# %%
episodes.plot_trial_average(column_key='angle',
                            #roiIndex=2,
                            with_std=False);

# %%
