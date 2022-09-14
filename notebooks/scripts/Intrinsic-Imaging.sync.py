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
# --- load standard modules
import pprint, os, sys
import numpy as np
import matplotlib.pylab as plt
# --- physion modules
sys.path.append(os.path.join(os.path.expanduser('~'), 'work', 'physion'))
import physion
from physion.dataviz.datavyz.datavyz import graph_env
ge = graph_env('manuscript')

# %%
# --- load a data folder
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', '2022_09_13', '15-26-26')
maps = np.load(os.path.join(datafolder, 'draft-maps.npy'), allow_pickle=True).item()
params, (t, data) = physion.intrinsic.Analysis.load_raw_data(datafolder,
                                                             'up', run_id=1)


# %% [markdown]
# ## raw Intrinsic Imaging data

# %%
# plot raw data
def show_raw_data(t, data, params, maps,
                  pixel=(200,200)):
    
    fig, AX = ge.figure(axes_extents=[[[5,1]],[[5,1]],[[1,1] for i in range(5)]],
                        wspace=1.2, figsize=(1,1.1))

    AX[0][0].plot(t, data[:,pixel[0], pixel[1]], 'k', lw=1)
    ge.set_plot(AX[0][0], ylabel='pixel\n intensity (a.u.)', xlabel='time (s)',
                xlim=[t[0], t[-1]])
    ge.annotate(AX[0][0], 'pixel: (%i, %i) ' % pixel, (1,1), ha='right', color='r', size='x-small')

    AX[1][0].plot(params['STIM']['up-times'], params['STIM']['up-angle'], 'k', lw=1)
    ge.set_plot(AX[1][0], ['left'], 
                ylabel='bar stim.\n angle ($^o$)',
                xlim=[t[0], t[-1]])

    ge.image(np.rot90(maps['vasculature'], k=1), ax=AX[2][0],
             title='green light')

    AX[2][1].scatter([pixel[0]], [pixel[1]], s=100, color='none', edgecolor='r', lw=2)
    ge.image(np.rot90(data[0,:,:], k=1), ax=AX[2][1],
             title='t=%.1fs' % t[0])

    AX[2][2].scatter([pixel[0]], [pixel[1]], s=100, color='none', edgecolor='r', lw=2)
    ge.image(np.rot90(data[-1,:,:], k=1), ax=AX[2][2],
             title='t=%.1fs' % t[-1])

    spectrum = np.fft.fft(data[:,pixel[0], pixel[1]], axis=0)
    
    power, phase = np.abs(spectrum), np.angle(spectrum)

    AX[2][3].plot(np.arange(1, len(power)), power[1:], color=ge.gray, lw=1)
    AX[2][3].plot([params['Nrepeat']], [power[params['Nrepeat']]], 'o', color=ge.blue, ms=4)
    ge.annotate(AX[2][3], '  stim. freq.', (params['Nrepeat'], power[params['Nrepeat']]), 
                color=ge.blue, xycoords='data', ha='left')

    AX[2][4].plot(np.arange(1, len(power)), phase[1:], color=ge.gray, lw=1)
    AX[2][4].plot([params['Nrepeat']], [phase[params['Nrepeat']]], 'o', color=ge.blue, ms=4)

    ge.set_plot(AX[2][3], xscale='log', yscale='log', 
                xlim=[.99,101], ylim=[power[1:].max()/120.,1.5*power[1:].max()],
                xlabel='freq (sample unit)', ylabel='power (a.u.)')
    ge.set_plot(AX[2][4], xscale='log', 
                xlim=[.99,101], xlabel='freq (sample unit)', ylabel='phase (Rd)')

show_raw_data(t, data, params, maps, pixel=(170,160))

# %% [markdown]
# ## Compute retinotopic maps

# %%
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', '2022_09_13', '15-26-26')
maps = {}
maps = physion.intrinsic.Analysis.compute_retinotopic_maps(datafolder, 'altitude', maps=maps)
maps = physion.intrinsic.Analysis.compute_retinotopic_maps(datafolder, 'azimuth', maps=maps)
np.save(os.path.join(datafolder, 'draft-maps.npy'), maps)

# %% [markdown]
# ### Visualizing retinotopic maps

# %%
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', '2022_09_13', '15-26-26')
maps = np.load(os.path.join(datafolder, 'draft-maps.npy'), allow_pickle=True).item()
fig = physion.intrinsic.Analysis.plot_phase_power_maps(maps, 'up')
# just one stim: "up"

# %%
# azimuth = "up" - "down" protocols
fig = physion.intrinsic.Analysis.plot_retinotopic_maps(maps, 'altitude')
fig.suptitle('altitude map');

# %%
# azimuth = "right" - "left" protocols
fig = physion.intrinsic.Analysis.plot_retinotopic_maps(maps, 'azimuth')
fig.suptitle('azimuth map');

# %% [markdown]
# # Visual Area segmentation
#
# Copied from:
#     
# https://nbviewer.org/github/zhuangjun1981/retinotopic_mapping/blob/master/retinotopic_mapping/examples/signmap_analysis/retinotopic_mapping_example.ipynb
#     
#     
# /!\ Cite the original implementation/work by Jun Zhuang: *Zhuang et al., Elife 2017*
#
# https://elifesciences.org/articles/18372
#

# %%
# --- load standard modules
import pprint, os, sys
import numpy as np
import matplotlib.pylab as plt

# --- physion modules
sys.path.append(os.path.join(os.path.expanduser('~'), 'work', 'physion'))
import physion
from physion.dataviz.datavyz.datavyz import graph_env
ge = graph_env('manuscript')

import physion.intrinsic.RetinotopicMapping as rm

# %%
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', '2022_09_13', '15-26-26')
maps = np.load(os.path.join(datafolder, 'draft-maps.npy'), allow_pickle=True).item()
trial_data = physion.intrinsic.Analysis.build_trial_data(maps)

# %%
params = {
          'phaseMapFilterSigma': 1,
          'signMapFilterSigma': 3.,
          'signMapThr': 0.35,
          'eccMapFilterSigma': 15.0,
          'splitLocalMinCutStep': 5.,
          'closeIter': 3,
          'openIter': 3,
          'dilationIter': 15,
          'borderWidth': 1,
          'smallPatchThr': 100,
          'visualSpacePixelSize': 0.5,
          'visualSpaceCloseIter': 15,
          'splitOverlapThr': 1.1,
          'mergeOverlapThr': 0.1
          }

# %% [markdown]
# ### Generating visual sign map

# %%
trial = rm.RetinotopicMappingTrial(**trial_data,
                                   params=params)

# %%
_ = trial._getSignMap(isPlot=True)
plt.show()

# %% [markdown]
# ### Binarizing filtered visual signmap

# %%
_ = trial._getRawPatchMap(isPlot=True)

# %% [markdown]
# ### Generating raw patches

# %%
_ = trial._getRawPatches(isPlot=True)

# %% [markdown]
# ### Generating determinant map

# %%
_ = trial._getDeterminantMap(isPlot=True)

# %% [markdown]
# ### Generating eccentricity map for each patch

# %%
_ = trial._getEccentricityMap(isPlot=True)

# %%
_ = trial._splitPatches(isPlot=True)

# %%
_ = trial._mergePatches(isPlot=True)

# %%
_ = trial.plotFinalPatchBorders2()

# %% [markdown]
# # Demo of FFT analysis of periodic stimulation
# Strategy of Kalatsky & Stryker, _Neuron_ (2003). See: http://doi.org/10.1016/s0896-6273(03)00286-1

# %%
# demo of analysis motivation
import numpy as np

max_sample = 20

def from_angle_to_fraction(angle):
    return -(angle-np.pi)/2./np.pi
    
def demo_fig(max_sample,
             nrepeat=10,
             n_sample_per_repeat= 137,
             signal_amplitude=0.8,
             noise_fraction=0.2,
             slow_fraction=0.2,
             seed=10):
    
    np.random.seed(seed)
    fig, AX = ge.figure(axes_extents=[[[2,1]],[[1,1],[1,1]]], figsize=(1.4,1), wspace=0.7, hspace=0.2)
    
    x = -np.exp(-(np.arange(n_sample_per_repeat)-max_sample)**2/30)+3500
    X = np.concatenate([x for n in range(nrepeat)])
    X += noise_fraction*np.random.randn(len(X))+slow_fraction*np.sin(np.linspace(0, 10, len(X)))
    
    AX[0][0].plot(X, 'k-')
    AX[0][0].plot(x, lw=3, color=ge.orange)
    ge.draw_bar_scales(AX[0][0], Xbar=n_sample_per_repeat, Xbar_label='1repeat',
                       Ybar=1, Ybar_label='signal', remove_axis=True)
    ge.title(AX[0][0], 'for one repeat, event @ sample: %i /%i , %i repeats' % (max_sample, n_sample_per_repeat, nrepeat), size='small')
    spectrum = np.fft.fft(X)
    power, phase = np.abs(spectrum)/len(x), np.angle(spectrum)
    AX[1][0].plot(power)
    AX[1][0].plot([nrepeat], [power[nrepeat]], 'o', color=ge.red)
    AX[1][1].plot(phase)
    AX[1][1].plot([nrepeat], [phase[nrepeat]], 'o', color=ge.red)
    ge.title(AX[1][1], 'est. event @ sample=%i' % (from_angle_to_fraction(phase[nrepeat])*n_sample_per_repeat), color=ge.red)
    ge.set_plot(AX[1][0], xscale='log', yscale='log', ylim=[power.min(), 2*power[nrepeat]],
                xlabel='freq', ylabel='power')
    ge.set_plot(AX[1][1], xscale='log', xlabel='freq', ylabel='phase (Rd)', yticks=[-np.pi, 0, np.pi],
               yticks_labels=['-$\pi$', '0', '$\pi$'])
    return fig

demo_fig(20, seed=3);

# %%
demo_fig(96);

# %%
ge.save_on_desktop(demo_fig(96, noise_fraction=0.8, nrepeat=15), 'fig.png')

# %%
