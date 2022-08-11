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
# # Visual Stimulation design and procedures

# %%
import sys, os
sys.path.append(os.path.join(os.path.expanduser('~'), 'work', 'physion'))

from physion.dataviz.datavyz.datavyz import graph_env_manuscript as ge
import numpy as np


# %%
def cm_to_angle(distance,
                distance_from_eye=15.):
    # distance_from_eye in cm
    return 180./np.pi*np.arctan(distance/distance_from_eye)

def cm_to_angle_lin(distance, distance_from_eye=15.):
    return cm_to_angle(1, distance_from_eye=distance_from_eye)*distance

def angle_to_cm(angle,
                distance_from_eye=15.):
    # distance_from_eye in cm
    return distance_from_eye*np.tan(np.pi*angle/180.)

screen_width = 48. # cm
screen_height = 27. # cm
distance_from_eye = 15. # cm

max_height = cm_to_angle(screen_height/2.)
max_width = cm_to_angle(screen_width/2.)

angles = np.linspace(0, 1.3*max_width, 100) #
positions = angle_to_cm(angles, distance_from_eye=distance_from_eye)

fig, ax = ge.figure()
ax.plot(angles, positions, color='k')
ge.plot(angles, angle_to_cm(1)*angles, ls='--', ax=ax, no_set=True, color=ge.red)
ge.annotate(ax, 'lin. approx.', (0.9,0.35), color=ge.red)

ax.plot(np.ones(2)*max_height, [0,positions.max()], ':', color=ge.green, lw=1)
ge.annotate(ax, 'max.\naltitude', (max_height,positions.max()), color=ge.green, xycoords='data', va='top', ha='right')

ax.plot(np.ones(2)*max_width, [0,positions.max()], ':', color=ge.blue, lw=1)
ge.annotate(ax, 'max.\nazimuth', (max_width,positions.max()), color=ge.blue, xycoords='data', ha='center')

ge.set_plot(ax, 
            ylabel='distance (cm) \nfrom center',
            xlabel='angle (deg)\nfrom center',
            ylim=[0,positions.max()])

ge.save_on_desktop(fig)

# %%
import matplotlib.pylab as plt
fig, ax = plt.subplots(figsize=(screen_width/20.,screen_height/20.))
ge.draw_bar_scales(ax, Ybar=angle_to_cm(20), Xbar=angle_to_cm(20), Xbar_label='20$^o$', Ybar_label='20$^o$')
ax.set_xlim(np.arange(-1,3,2)*screen_width/2.)
ax.set_ylim(np.arange(-1,3,2)*screen_height/2.)
ax.set_xlabel('x-coord (cm)')
ax.set_ylabel('z-coord (cm)')
ax.set_title('screen coordinates')
ge.save_on_desktop(fig)

# %%
print('visual field covered:')
print('azimuth:', cm_to_angle(screen_width/2.))
print('altitude:', cm_to_angle(screen_height/2.))

# %% [markdown]
# ## stim design

# %%
# start from a simple protocol to load all required data
import json, os
with open(os.path.join('..', 'physion', 'exp', 'protocols', 'demo', 'uniform-bg.json')) as j:
    protocol = json.load(j)
    
from physion.visual_stim.stimuli import build_stim, init_bg_image
protocol['no-window'] = True
stim = build_stim(protocol)


# %%
def compute_new_image_with_dots(self, index, dot_size, Ndots,
                                away_from_edges_factor=4):
    dot_size_pix = int(np.round(self.angle_to_pix(dot_size),0))
    Nx = int(self.x.shape[0]/dot_size_pix)
    Nz = int(self.x.shape[1]/dot_size_pix)
    dots = []
    img = init_bg_image(self, index)
    for n in range(Ndots):
        ix, iz = (np.random.choice(np.arange(away_from_edges_factor, Nx-away_from_edges_factor),1)[0],
                  np.random.choice(range(away_from_edges_factor,Nz-away_from_edges_factor),1)[0])
        img[dot_size_pix*ix:dot_size_pix*(ix+1),
            dot_size_pix*iz:dot_size_pix*(iz+1)] = 1
    return img
        
img = compute_new_image_with_dots(stim, 0, 5, 8)

# %%

# %%
fig, ax = ge.figure(figsize=(4,4))

radius = 20
img = init_bg_image(stim, 0)
img += np.exp(-(stim.x**2+stim.z**2)/2/radius**2)
ge.image(img, ax=ax)

# %% [markdown]
# ## Drawing stimulus for figures

# %%
# start from a simple protocol to load all required data
import json, os
with open(os.path.join('..', 'physion', 'exp', 'protocols', 'demo', 'increasing-contrast.json')) as j:
    protocol = json.load(j)
    
from physion.visual_stim.stimuli import build_stim, init_bg_image
protocol['no-window'] = True
stim = build_stim(protocol)

fig, AX = ge.figure(axes=(10,1), wspace=0.5)
for i in range(10):
    AX[i].imshow(-stim.get_image(i).T, vmin=-1, vmax=1, cmap=ge.binary)
    AX[i].axis('off')
ge.save_on_desktop(fig, 'fig.svg')

# %%
# start from a simple protocol to load all required data
import json, os
with open(os.path.join('..', 'physion', 'exp', 'protocols', 'demo', 'moving-dots.json')) as j:
    protocol = json.load(j)
    
from physion.visual_stim.stimuli import build_stim, init_bg_image
protocol['no-window'] = True
stim = build_stim(protocol)

fig, ax = ge.figure(axes=(1,1), wspace=0.5)
ax.imshow(-stim.get_image(1).T, vmin=-1, vmax=1, cmap=ge.binary)
ax.axis('off')
ge.save_on_desktop(fig, 'fig.svg')

# %%
