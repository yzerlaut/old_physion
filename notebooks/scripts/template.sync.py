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

bug = ('keep this line for having jupyter_ascending.vim working')

# %% [markdown]
# # Notebook template

# %% [markdown]
# ## Load all required modules

# %%
# general python modules
import sys, os
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
# ## Load a datafile

# %%
data_folder =  os.path.join(os.path.expanduser('~'), 'DATA')
FILES, _, _ = scan_folder_for_NWBfiles(data_folder) 
data = Data(FILES[0])

# %%
print(data)

# %% [markdown]
# ## Plot the raw data

# %%
mdata = MultimodalData(FILES[0])

# %%
mdata.plot_raw_data()


