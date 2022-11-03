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


# -- PIL to build a session summary pdf
from PIL import Image
width, height = int(8.27 * 300), int(11.7 * 300) # A4 at 300dpi : (2481, 3510)

# let's create the A4 page
page = Image.new('RGB', (width, height), 'white')

page.save(os.path.join(os.path.expanduser('~'), 'Desktop', 'session-summary.pdf'))

# %%
filename = os.path.join(os.path.expanduser('~'), 'DATA', 'JO-VIP-CB1', '2022_10_26-18-13-13.nwb')

# %%
# let's pick the first one
data = Data(filename)

# %%
# the different protocols are listed in data.protocols
for i, protocol in enumerate(data.protocols):
    print('  - protocol #%i: "%s"' % (i+1, protocol))

# %%
episodes = EpisodeResponse(filename,
                           protocol_id=data.get_protocol_id('ff-drifiting-gratings-4orientation-5contrasts-log-spaced-10repeats'),
                           quantities=['dFoF'],
                           prestim_duration=3,
                           verbose=True)

# %%
fig, _ = episodes.plot_trial_average(column_key='contrast', color_key='angle', 
                                     with_std=False, with_annotation=True)

# %%
import tempfile

# %%
tempfile.gettempdir()

# %%


def metadata_fig(data):

    metadata = dict(data.metadata)

    string = """
    Datafile: "%s"
    Mouse ID: "%s" """ % (filename, data.metadata['subject_ID'])
    i=1
    while 'Protocol-%i'%i in data.metadata:
        string += '\n          -  Protocol-%i: %s' % (i, data.metadata['Protocol-%i'%i])
        i+=1
    # write on fig
    fig, ax = ge.figure(figsize=(5,1), left=0)
    ge.annotate(ax, string, (0,0), size='small')
    ax.axis('off')
    return fig, ax
metadata_fig(data)

# %%
data.
