# general modules
import pynwb, os, sys, pathlib
import numpy as np
import matplotlib.pylab as plt

# custom modules
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from dataviz import plots
from analysis.read_NWB import Data
from dataviz.tools import *
from Ca_imaging.tools import compute_CaImaging_trace
from visual_stim.psychopy_code.stimuli import build_stim

# we define a data object fitting this analysis purpose
class MultimodalData(Data):
    
    def __init__(self, filename, verbose=False, with_visual_stim=False):
        """ opens data file """
        super().__init__(filename, verbose=verbose)
        if with_visual_stim:
            self.init_visual_stim()
        else:
            self.visual_stim = None
            
    def init_visual_stim(self):
        self.metadata['load_from_protocol_data'], self.metadata['no-window'] = True, True
        self.visual_stim = build_stim(self.metadata, no_psychopy=True)
        
    def add_Photodiode(self, tlim, ax,
                       fig_fraction_start=0., fig_fraction=1., subsampling=10, color='grey'):
        i1, i2 = convert_times_to_indices(*tlim, self.nwbfile.acquisition['Photodiode-Signal'])
        x = convert_index_to_time(range(i1,i2), self.nwbfile.acquisition['Photodiode-Signal'])[::subsampling]
        y = self.nwbfile.acquisition['Photodiode-Signal'].data[i1:i2][::subsampling]
        ax.plot(x, (y-y.min())/(y.max()-y.min())*fig_fraction+fig_fraction_start, color=color)
        
    def add_Locomotion(self, tlim, ax,
                       fig_fraction_start=0., fig_fraction=1., subsampling=10, color='red'):
        i1, i2 = convert_times_to_indices(*tlim, self.nwbfile.acquisition['Running-Speed'])
        x = convert_index_to_time(range(i1,i2), self.nwbfile.acquisition['Running-Speed'])[::subsampling]
        y = self.nwbfile.acquisition['Running-Speed'].data[i1:i2][::subsampling]
        ax.plot(x, (y-y.min())/(y.max()-y.min())*fig_fraction+fig_fraction_start, color=color)
        
    def add_Pupil(self, tlim, ax,
                  fig_fraction_start=0., fig_fraction=1., subsampling=1, color='red'):
        i1, i2 = convert_times_to_indices(*tlim, self.nwbfile.processing['Pupil'].data_interfaces['cx'])
        t = self.nwbfile.processing['Pupil'].data_interfaces['sx'].timestamps[i1:i2]
        diameter = self.nwbfile.processing['Pupil'].data_interfaces['sx'].data[i1:i2]*\
                               self.nwbfile.processing['Pupil'].data_interfaces['sy'].data[i1:i2]
        x, y = t[::subsampling], diameter[::subsampling]
        ax.plot(x, (y-y.min())/(y.max()-y.min())*fig_fraction+fig_fraction_start, color=color)
    
    def add_CaImaging(self, tlim, ax,
                      fig_fraction_start=0., fig_fraction=1., color='green',
                      quantity='CaImaging', subquantity='Fluorescence', roiIndices='all',
                      vicinity_factor=1):
        if (type(roiIndices)==str) and roiIndices=='all':
            roiIndices = np.arange(np.sum(self.iscell))
        if color=='tab':
            COLORS = [plt.cm.tab10(n%10) for n in range(len(roiIndices))]
        else:
            COLORS = [str(color) for n in range(len(roiIndices))]

        dF = compute_CaImaging_trace(self, subquantity, roiIndices) # validROI indices inside !!
        i1 = convert_time_to_index(tlim[0], self.Neuropil, axis=1)
        i2 = convert_time_to_index(tlim[1], self.Neuropil, axis=1)
        tt = self.Neuropil.timestamps[np.arange(i1,i2)]
        if vicinity_factor>1:
            ymax_factor = fig_fraction*(1-1./vicinity_factor)
        else:
            ymax_factor = fig_fraction/len(roiIndices)
        for n, ir in zip(range(len(roiIndices))[::-1], roiIndices[::-1]):
            y = dF[n, np.arange(i1,i2)]
            ypos = n*fig_fraction/len(roiIndices)/vicinity_factor+fig_fraction_start
            if subquantity in ['dF/F', 'dFoF']:
                ax.plot(tt, y/2.*ymax_factor+ypos, color=COLORS[n], lw=1)
                ax.plot(tlim[0]*np.ones(2), np.arange(2)/2.*ymax_factor+ypos, color=COLORS[n], lw=1)
            elif y.max()>y.min():
                rescaled_y = (y-y.min())/(y.max()-y.min())
                ax.plot(tt, rescaled_y*ymax_factor+ypos, color=COLORS[n], lw=1)

            ax.annotate('ROI#%i'%(ir+1), (tlim[1], ypos), color=COLORS[n], fontsize=8)
        if subquantity in ['dF/F', 'dFoF']:
            ax.annotate('1$\Delta$F/F', (tlim[0], fig_fraction_start), ha='right',
                        rotation=90, color='k', fontsize=9)

    def add_CaImagingSum(self, tlim, ax,
                         fig_fraction_start=0., fig_fraction=1., color='green',
                         quantity='CaImaging', subquantity='Fluorescence'):
        i1 = convert_time_to_index(tlim[0], self.Neuropil, axis=1)
        i2 = convert_time_to_index(tlim[1], self.Neuropil, axis=1)
        tt = self.Neuropil.timestamps[np.arange(i1,i2)]
        y = compute_CaImaging_trace(self, subquantity, np.arange(np.sum(self.iscell))).sum(axis=0)[np.arange(i1,i2)]
        ax.plot(tt, (y-y.min())/(y.max()-y.min())*fig_fraction+fig_fraction_start, color=color)
        ax.annotate('Sum', (tlim[1], fig_fraction_start), fontsize=8)        
            
    def add_VisualStim(self, tlim, ax,
                       fig_fraction_start=0., fig_fraction=0.05,
                       color='k'):
        if self.visual_stim is None:
            self.init_visual_stim()
        cond = (self.nwbfile.stimulus['time_start_realigned'].data[:]>tlim[0]) &\
            (self.nwbfile.stimulus['time_stop_realigned'].data[:]<tlim[1])
        ylevel = fig_fraction_start+fig_fraction
        sx, sy = self.visual_stim.screen['resolution']
        ax_pos = ax.get_position()
        for i in np.arange(self.nwbfile.stimulus['time_start_realigned'].num_samples)[cond]:
            tstart = self.nwbfile.stimulus['time_start_realigned'].data[i]
            tstop = self.nwbfile.stimulus['time_stop_realigned'].data[i]
            ax.plot([tstart, tstop], [ylevel, ylevel], color=color)
            ax.fill_between([tstart, tstop], [0,0], np.zeros(2)+ylevel, lw=0, alpha=0.05, color=color)
            x0, y0 = ax_pos.x0+ax_pos.width*(tstart-tlim[0])/(tlim[1]-tlim[0]), ax_pos.y0+ax_pos.height
            width = ax_pos.width/2./np.sum(cond)
            axi = plt.axes([x0, ax_pos.y1, width, width*2*sy/sx])
            # add arrow if drifting stim
            if (self.nwbfile.stimulus['frame_run_type'].data[i]=='drifting'):
                arrow={'direction':self.nwbfile.stimulus['angle'].data[i],
                       'center':(0,0), 'length':25, 'width_factor':0.1, 'color':'red'}
            else:
                arrow = None
            # add VSE if in stim (TO BE DONE)
            if True:
                vse = None
            self.visual_stim.show_frame(i, ax=axi, label=None, arrow=arrow)
    
    def plot(self, 
             tlim=[0,100],
             settings={'Photodiode':dict(fig_fraction=.1, subsampling=10, color='grey'),
                       'Locomotion':dict(fig_fraction=1, subsampling=10, color='b'),
                       'Pupil':dict(fig_fraction=2, subsampling=10, color='red'),
                       'CaImaging':dict(fig_fraction=4, 
                                        quantity='CaImaging', subquantity='Fluorescence', color='green',
                                        roiIndices='all'),
                       'VisualStim':dict(fig_fraction=0, color='black')},                    
                figsize=(15,6), Tbar=20,
                ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, figsize=figsize)
            plt.subplots_adjust(left=0, right=1., top=.9)
        else:
            fig = None
        fig_fraction_full, fstart = np.sum([settings[key]['fig_fraction'] for key in settings]), 0
        for key in settings:
            settings[key]['fig_fraction_start'] = fstart
            settings[key]['fig_fraction'] = settings[key]['fig_fraction']/fig_fraction_full
            fstart += settings[key]['fig_fraction']
        for key in settings:
            getattr(self, 'add_%s' % key)(tlim, ax, **settings[key])
            ax.annotate(key, (tlim[0],
                           settings[key]['fig_fraction_start']+settings[key]['fig_fraction']/2.0),
                        color=(settings[key]['color'] if settings[key]['color']!='tab' else 'k'),
                        fontsize=8, ha='right')
        ax.axis('off')
        ax.plot([tlim[1]-Tbar, tlim[1]], [0,0], lw=2, color='k')
        ax.set_xlim([tlim[0]-0.01*(tlim[1]-tlim[0]),tlim[1]+0.01*(tlim[1]-tlim[0])])
        ax.set_ylim([-0.05,1.05])
        ax.annotate(' %is' % Tbar, [tlim[1], 0], color='k', fontsize=9)
        
        return fig, ax
        

    def show_CaImaging_FOV(self, key='meanImg', NL=1, cmap='viridis', ax=None):
        if ax is None:
            fig, ax = plt.subplots(1)
        else:
            fig = None
        img = self.nwbfile.processing['ophys'].data_interfaces['Backgrounds_0'].images[key][:]
        img = (img-img.min())/(img.max()-img.min())
        img = np.power(img, 1/NL)
        ax.imshow(img, vmin=0, vmax=1, cmap=cmap, aspect='equal', interpolation='none')
        ax.axis('off')
        ax.set_title(key)
        return fig, ax
    
     
if __name__=='__main__':
    
    filename = os.path.join(os.path.expanduser('~'), 'DATA', '2021_03_11-17-32-34.nwb')
    data = MultimodalData(filename)
    data.plot([250, 300], 
              settings={'Photodiode':dict(fig_fraction=.1, subsampling=10, color='grey'),
                        'Locomotion':dict(fig_fraction=1, subsampling=10, color='b'),
                        'Pupil':dict(fig_fraction=2, subsampling=10, color='red'),
                        'CaImaging':dict(fig_fraction=4, 
                                         quantity='CaImaging', subquantity='Fluorescence', color='green',
                                                   roiIndices=[2, 6, 9, 10, 13, 15, 16, 17, 38, 41]),
                        'VisualStim':dict(fig_fraction=0, size=0.05, color='black')},                    
              Tbar=10)
    plt.show()
