# general modules
import pynwb, os, sys, pathlib
import numpy as np
import matplotlib.pylab as plt

# custom modules
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from dataviz import tools as dv_tools
from analysis import read_NWB, process_NWB, stat_tools
from visual_stim.stimuli import build_stim

# datavyz submodule
try:
    from datavyz import graph_env_manuscript as ge
except ModuleNotFoundError:
    print('--------------------------------------------')
    print('  "datavyz" submodule not found')
    print('  -> install with "pip install ./physion/dataviz/datavyz/."')
    print('             (after a "git submodule init; git submodule update" if not already done) ')


# we define a data object fitting this analysis purpose
class MultimodalData(read_NWB.Data):
    
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

    ###-------------------------------------
    ### ----- RAW DATA PLOT components -----
    ###-------------------------------------

    def shifted_start(self, tlim, frac_shift=0.01):
        return tlim[0]-0.01*(tlim[1]-tlim[0])
    
    def plot_scaled_signal(self, ax, t, signal, tlim, scale_bar, ax_fraction_extent, ax_fraction_start,
                           color='#1f77b4', scale_unit_string='%.1f'):
        # generic function to add scaled signal

        try:
            scale_range = np.max([signal.max()-signal.min(), scale_bar])
            min_signal = signal.min()
        except ValueError:
            scale_range = scale_bar
            min_signal = 0

        ax.plot(t, ax_fraction_start+(signal-min_signal)*ax_fraction_extent/scale_range, color=color, lw=1)
        if scale_unit_string!='':
            ax.plot(self.shifted_start(tlim)*np.ones(2), ax_fraction_start+scale_bar*np.arange(2)*ax_fraction_extent/scale_range, color=color, lw=1)
        if '%' in scale_unit_string:
            ge.annotate(ax, str(scale_unit_string+' ') % scale_bar, (self.shifted_start(tlim), ax_fraction_start), ha='right', color=color, va='center', xycoords='data')
        elif scale_unit_string!='':
            ge.annotate(ax, scale_unit_string, (self.shifted_start(tlim), ax_fraction_start), ha='right', color=color, va='center', xycoords='data')

    def add_name_annotation(self, ax, name, tlim, ax_fraction_extent, ax_fraction_start,
                            color='k', rotation=0):
        ge.annotate(ax, ' '+name, (tlim[1], ax_fraction_extent/2.+ax_fraction_start), xycoords='data', color=color, va='center', rotation=rotation)
        
    def add_Photodiode(self, tlim, ax,
                       fig_fraction_start=0., fig_fraction=1., subsampling=10, color='#808080', name='photodiode'):
        i1, i2 = dv_tools.convert_times_to_indices(*tlim, self.nwbfile.acquisition['Photodiode-Signal'])
        t = dv_tools.convert_index_to_time(range(i1,i2), self.nwbfile.acquisition['Photodiode-Signal'])[::subsampling]
        y = self.nwbfile.acquisition['Photodiode-Signal'].data[i1:i2][::subsampling]
        
        self.plot_scaled_signal(ax, t, y, tlim, 1e-5, fig_fraction, fig_fraction_start, color=color, scale_unit_string='')        
        self.add_name_annotation(ax, name, tlim, fig_fraction, fig_fraction_start, color=color)

    def add_Electrophy(self, tlim, ax,
                       fig_fraction_start=0., fig_fraction=1., subsampling=2, color='k',
                       name='LFP'):
        i1, i2 = dv_tools.convert_times_to_indices(*tlim, self.nwbfile.acquisition['Electrophysiological-Signal'])
        t = dv_tools.convert_index_to_time(range(i1,i2), self.nwbfile.acquisition['Electrophysiological-Signal'])[::subsampling]
        y = self.nwbfile.acquisition['Electrophysiological-Signal'].data[i1:i2][::subsampling]

        self.plot_scaled_signal(ax, t, y, tlim, 0.2, fig_fraction, fig_fraction_start, color=color, scale_unit_string='%.1fmV')        
        self.add_name_annotation(ax, name, tlim, fig_fraction, fig_fraction_start, color=color)

    def add_Locomotion(self, tlim, ax,
                       fig_fraction_start=0., fig_fraction=1., subsampling=2,
                       speed_scale_bar=1, # cm/s
                       color='#1f77b4', name='run. speed'):
        if not hasattr(self, 'running_speed'):
            self.build_running_speed()
        i1, i2 = dv_tools.convert_times_to_indices(*tlim, self.nwbfile.acquisition['Running-Speed'])
        x, y = self.t_running_speed[i1:i2][::subsampling], self.running_speed[i1:i2][::subsampling]

        self.plot_scaled_signal(ax, x, y, tlim, speed_scale_bar, fig_fraction, fig_fraction_start, color=color, scale_unit_string='%.1fcm/s')        
        self.add_name_annotation(ax, name, tlim, fig_fraction, fig_fraction_start, color=color)
        
    def add_Pupil(self, tlim, ax,
                  fig_fraction_start=0., fig_fraction=1., subsampling=2,
                  pupil_scale_bar = 0.5, # scale bar in mm
                  color='red', name='pupil diam.'):
        i1, i2 = dv_tools.convert_times_to_indices(*tlim, self.nwbfile.processing['Pupil'].data_interfaces['cx'])
        if not hasattr(self, 'pupil_diameter'):
            self.build_pupil_diameter()
        x, y = self.t_pupil[i1:i2][::subsampling], self.pupil_diameter[i1:i2][::subsampling]

        self.plot_scaled_signal(ax, x, y, tlim, pupil_scale_bar, fig_fraction, fig_fraction_start, color=color, scale_unit_string='%.1fmm')        
        self.add_name_annotation(ax, name, tlim, fig_fraction, fig_fraction_start, color=color)

    def add_GazeMovement(self, tlim, ax,
                         fig_fraction_start=0., fig_fraction=1., subsampling=2,
                         gaze_scale_bar = 0.2, # scale bar in mm
                         color='#ff7f0e', name='gaze mov.'):
        i1, i2 = dv_tools.convert_times_to_indices(*tlim, self.nwbfile.processing['Pupil'].data_interfaces['cx'])
        if not hasattr(self, 'gaze_movement'):
            self.build_gaze_movement()
        
        x, y = self.t_pupil[i1:i2][::subsampling], self.gaze_movement[i1:i2][::subsampling]

        self.plot_scaled_signal(ax, x, y, tlim, gaze_scale_bar, fig_fraction, fig_fraction_start, color=color, scale_unit_string='%.1fmm')
        self.add_name_annotation(ax, name, tlim, fig_fraction, fig_fraction_start, color=color)

    def add_FaceMotion(self, tlim, ax,
                       fig_fraction_start=0., fig_fraction=1., subsampling=2, color='#9467bd', name='facemotion'):
        if not hasattr(self, 'facemotion'):
            self.build_facemotion()
        i1, i2 = dv_tools.convert_times_to_indices(*tlim, self.nwbfile.processing['FaceMotion'].data_interfaces['face-motion'])
        x, y = self.t_facemotion[i1:i2][::subsampling], self.facemotion[i1:i2][::subsampling]

        self.plot_scaled_signal(ax, x, y, tlim, 1., fig_fraction, fig_fraction_start, color=color, scale_unit_string='') # no scale bar here
        self.add_name_annotation(ax, name, tlim, fig_fraction, fig_fraction_start, color=color)

        
    def add_CaImagingRaster(self, tlim, ax, raster=None,
                            fig_fraction_start=0., fig_fraction=1., color='green',
                            subquantity='Fluorescence', roiIndices='all', subquantity_args={},
                            cmap=plt.cm.binary,
                            normalization='None', subsampling=1,
                            name='\nROIs'):

        if subquantity=='Fluorescence' and (raster is None):
            if (roiIndices=='all'):
                raster = self.Fluorescence.data[:,:]
            else:
                raster = self.Fluorescence.data[roiIndices,:]
                
        elif (subquantity in ['dFoF', 'dF/F']) and (raster is None):
            if not hasattr(self, 'dFoF'):
                self.build_dFoF(**subquantity_args)
            if (roiIndices=='all'):
                raster = self.dFoF[:,:]
            else:
                raster = self.dFoF[roiIndices,:]
                
            roiIndices = np.arange(self.iscell.sum())
        elif (roiIndices=='all') and (subquantity in ['dFoF', 'dF/F']):
            roiIndices = np.arange(self.nROIs)
            
        if normalization in ['per line', 'per-line', 'per cell', 'per-cell']:
            raster = np.array([(raster[i,:]-np.min(raster[i,:]))/(np.max(raster[i,:])-np.min(raster[i,:])) for i in range(raster.shape[0])])
            
        indices=np.arange(*dv_tools.convert_times_to_indices(*tlim, self.Neuropil, axis=1))[::subsampling]
        
        ax.imshow(raster[:,indices], origin='lower', cmap=cmap,
                  aspect='auto', interpolation='none', vmin=0, vmax=1,
                  extent=(dv_tools.convert_index_to_time(indices[0], self.Neuropil),
                          dv_tools.convert_index_to_time(indices[-1], self.Neuropil),
                          fig_fraction_start, fig_fraction_start+fig_fraction))

        if normalization in ['per line', 'per-line', 'per cell', 'per-cell']:
            _, axb = ge.bar_legend(ax,
                          # X=[0,1], bounds=[0,1],
                          continuous=False, colormap=cmap,
                          colorbar_inset=dict(rect=[-.04,
                                           fig_fraction_start+.2*fig_fraction,
                                           .01,
                                           .6*fig_fraction], facecolor=None),
                          color_discretization=100, no_ticks=True, labelpad=4.,
                          label=('$\Delta$F/F' if (subquantity in ['dFoF', 'dF/F']) else ' fluo.'),
                          fontsize='small')
            ge.annotate(axb, ' max', (1,1), size='x-small')
            ge.annotate(axb, ' min', (1,0), size='x-small', va='top')
            
        self.add_name_annotation(ax, name, tlim, fig_fraction, fig_fraction_start, rotation=90)

        ge.annotate(ax, '1', (tlim[1], fig_fraction_start), xycoords='data')
        ge.annotate(ax, '%i' % raster.shape[0],
                    (tlim[1], fig_fraction_start+fig_fraction), va='top', xycoords='data')
        
        
    def add_CaImaging(self, tlim, ax,
                      fig_fraction_start=0., fig_fraction=1., color='green',
                      subquantity='Fluorescence', roiIndices='all', dFoF_args={},
                      vicinity_factor=1, subsampling=1, name='[Ca] imaging'):

        if (subquantity in ['dF/F', 'dFoF']) and (not hasattr(self, 'dFoF')):
            self.build_dFoF(**dFoF_args)
            
        if (type(roiIndices)==str) and roiIndices=='all':
            roiIndices = self.valid_roiIndices
            
        if color=='tab':
            COLORS = [plt.cm.tab10(n%10) for n in range(len(roiIndices))]
        else:
            COLORS = [str(color) for n in range(len(roiIndices))]

        i1, i2 = dv_tools.convert_times_to_indices(*tlim, self.Neuropil, axis=1)
        t = np.array(self.Neuropil.timestamps[:])[np.arange(i1,i2)][::subsampling]
        
        for n, ir in zip(range(len(roiIndices))[::-1], roiIndices[::-1]):
            ypos = n*fig_fraction/len(roiIndices)/vicinity_factor+fig_fraction_start # bottom position
            
            if (subquantity in ['dF/F', 'dFoF']):
                y = self.dFoF[n, np.arange(i1,i2)][::subsampling]
                self.plot_scaled_signal(ax, t, y, tlim, 1., fig_fraction/len(roiIndices), ypos, color=color,
                                        scale_unit_string=('%.0f$\Delta$F/F' if (n==0) else ' '))
            else:
                y = self.Fluorescence.data[n, np.arange(i1,i2)][::subsampling]
                self.plot_scaled_signal(ax, t, y, tlim, 1., fig_fraction/len(roiIndices), ypos, color=color,
                                        scale_unit_string=('fluo (a.u.)' if (n==0) else ''))

            self.add_name_annotation(ax, ' ROI#%i'%(ir+1), tlim, fig_fraction/len(roiIndices), ypos, color=color)
            
        # ge.annotate(ax, name, (self.shifted_start(tlim), fig_fraction/2.+fig_fraction_start), color=color,
        #             xycoords='data', ha='right', va='center', rotation=90)
            

    def add_CaImagingSum(self, tlim, ax,
                         fig_fraction_start=0., fig_fraction=1., color='green',
                         quantity='CaImaging', subquantity='Fluorescence', subsampling=1,
                         name='Sum [Ca]'):
        
        if (subquantity in ['dF/F', 'dFoF']) and (not hasattr(self, 'dFoF')):
            self.build_dFoF()
            
        i1, i2 = dv_tools.convert_times_to_indices(*tlim, self.Neuropil, axis=1)
        t = np.array(self.Neuropil.timestamps[:])[np.arange(i1,i2)][::subsampling]
        
        if (subquantity in ['dF/F', 'dFoF']):
            y = self.dFoF.sum(axis=0)[np.arange(i1,i2)][::subsampling]
        else:
            y = self.Fluorescence.data[:,:].sum(axis=0)[np.arange(i1,i2)][::subsampling]

        self.plot_scaled_signal(ax, t, y, tlim, 1., fig_fraction, fig_fraction_start, color=color,
                                scale_unit_string=('%.0fdF/F' if subquantity in ['dF/F', 'dFoF'] else ''))
        self.add_name_annotation(ax, name, tlim, fig_fraction, fig_fraction_start, color=color)

        
    def add_VisualStim(self, tlim, ax,
                       fig_fraction_start=0., fig_fraction=0.05, size=0.1,
                       color='k', name='visual stim.'):
        if self.visual_stim is None:
            self.init_visual_stim()
        cond = (self.nwbfile.stimulus['time_start_realigned'].data[:]>tlim[0]) &\
            (self.nwbfile.stimulus['time_stop_realigned'].data[:]<tlim[1])
        ylevel = fig_fraction_start+fig_fraction/2.
        sx, sy = self.visual_stim.screen['resolution']
        ax_pos = ax.get_position()
        for i in np.arange(self.nwbfile.stimulus['time_start_realigned'].num_samples)[cond]:
            tstart = self.nwbfile.stimulus['time_start_realigned'].data[i]
            tstop = self.nwbfile.stimulus['time_stop_realigned'].data[i]
            # ax.plot([tstart, tstop], [ylevel, ylevel], color=color)
            ax.fill_between([tstart, tstop], [0,0], np.zeros(2)+ylevel,
                            lw=0, alpha=0.05, color=color)
            axi = ax.inset_axes([tstart, 1.01, (tstop-tstart), size], transform=ax.transData)
            axi.axis('equal')
            self.visual_stim.plot_stim_picture(i, ax=axi)
        ge.annotate(ax, ' '+name, (tlim[1], fig_fraction+fig_fraction_start), color=color, xycoords='data')

        
    def show_VisualStim(self, tlim,
                        Npanels=8,
                        enhance=False):
        
        if self.visual_stim is None:
            self.init_visual_stim()

        fig, AX = ge.figure(axes=(Npanels,1),
                            figsize=(1.6/2., 0.9/2.), top=3, bottom=2, wspace=.2)

        label={'degree':20,
               'shift_factor':0.03,
               'lw':0.5, 'fontsize':7}
        
        for i, ti in enumerate(np.linspace(*tlim, Npanels)):
            iEp = self.find_episode_from_time(ti)
            tEp = self.nwbfile.stimulus['time_start_realigned'].data[iEp]
            if iEp>=0:
                self.visual_stim.show_frame(iEp, ax=AX[i],
                                            time_from_episode_start=ti-tEp,
                                            label=label,
                                            enhance=enhance)
            AX[i].set_title('%.1fs' % ti, fontsize=6)
            AX[i].axis('off')
            label=None
            
        return fig, AX

    
    def plot_raw_data(self, 
                      tlim=[0,100],
                      settings={'Photodiode':dict(fig_fraction=.1, subsampling=10, color='grey'),
                                'Locomotion':dict(fig_fraction=1, subsampling=10, color='b'),
                                'FaceMotion':dict(fig_fraction=1, subsampling=10, color='purple'),
                                'Pupil':dict(fig_fraction=2, subsampling=10, color='red'),
                                'CaImaging':dict(fig_fraction=4, 
                                                 quantity='CaImaging', subquantity='Fluorescence', color='green',
                                                 roiIndices='all'),
                                'VisualStim':dict(fig_fraction=0, color='black')},                    
                      figsize=(3,3), Tbar=0., zoom_area=None,
                      ax=None):

        if ax is None:
            fig, ax = ge.figure(figsize=figsize, bottom=.3, left=.5)
        else:
            fig = None
            
        fig_fraction_full, fstart = np.sum([settings[key]['fig_fraction'] for key in settings]), 0
        
        for key in settings:
            settings[key]['fig_fraction_start'] = fstart
            settings[key]['fig_fraction'] = settings[key]['fig_fraction']/fig_fraction_full
            fstart += settings[key]['fig_fraction']
            
        for key in settings:
            getattr(self, 'add_%s' % key)(tlim, ax, **settings[key])

        # time scale bar
        if Tbar==0.:
            Tbar = np.max([int((tlim[1]-tlim[0])/30.), 1])

        ax.plot([self.shifted_start(tlim), self.shifted_start(tlim)+Tbar], [1.,1.], lw=1, color='k')
        ax.annotate((' %is' % Tbar if Tbar>=1 else  '%.1fs' % Tbar) ,
                    [self.shifted_start(tlim), 1.02], color='k', fontsize=9)
        
        ax.axis('off')
        ax.set_xlim([self.shifted_start(tlim)-0.01*(tlim[1]-tlim[0]),tlim[1]+0.01*(tlim[1]-tlim[0])])
        ax.set_ylim([-0.05,1.05])

        if zoom_area is not None:
            ax.fill_between(zoom_area, [0,0], [1,1],  color='k', alpha=.2, lw=0)
        
        return fig, ax

    
    ###-------------------------------------
    ### ----- IMAGING PLOT components -----
    ###-------------------------------------

    def show_CaImaging_FOV(self, key='meanImg', NL=1, cmap='viridis', ax=None, roiIndex=None, with_roi_zoom=False):
        
        if ax is None:
            fig, ax = ge.figure()
        else:
            fig = None
            
        img = self.nwbfile.processing['ophys'].data_interfaces['Backgrounds_0'].images[key][:]
        img = (img-img.min())/(img.max()-img.min())
        img = np.power(img, 1/NL)
        ax.imshow(img, vmin=0, vmax=1, cmap=cmap, aspect='equal', interpolation='none')
        ax.axis('off')
        
        if roiIndex is not None:
            indices = np.arange((self.pixel_masks_index[roiIndex-1] if roiIndex>0 else 0),
                                (self.pixel_masks_index[roiIndex] if roiIndex<len(self.valid_roiIndices) else len(self.pixel_masks_index)))
            x = np.mean([self.pixel_masks[ii][1] for ii in indices])
            sx = np.std([self.pixel_masks[ii][1] for ii in indices])
            y = np.mean([self.pixel_masks[ii][0] for ii in indices])
            sy = np.std([self.pixel_masks[ii][1] for ii in indices])
            # ellipse = plt.Circle((x, y), sx, sy)
            ellipse = plt.Circle((x, y), 1.5*(sx+sy), edgecolor='r', facecolor='none', lw=3)
            ax.add_patch(ellipse)
            if with_roi_zoom:
                ax.set_xlim([x-10*sx, x+10*sx])
                ax.set_ylim([y-10*sy, y+10*sy])

        ge.title(ax, key)
        
        return fig, ax


    
###------------------------------------------
### ----- Trial Average plot components -----
###------------------------------------------


class EpisodeResponse(process_NWB.EpisodeResponse):

    def __init__(self, filename,
                 protocol_id=0,
                 quantities=['dFoF'],
                 quantities_args=[{}],
                 prestim_duration=None,
                 verbose=False,
                 with_visual_stim=False):
        """ plot Episode Response """

        # load data first
        self.data = MultimodalData(filename,
                                   with_visual_stim=with_visual_stim,
                                   verbose=verbose)

        # initialize episodes
        super().__init__(self.data,
                         protocol_id=protocol_id,
                         quantities=quantities,
                         quantities_args=quantities_args,
                         prestim_duration=prestim_duration,
                         verbose=verbose)
        
    def plot_trial_average(self,
                           # episodes props
                           quantity='dFoF', roiIndex=None, roiIndices='all',
                           interpolation='linear',
                           baseline_substraction=False,
                           condition=None,
                           COL_CONDS=None, column_keys=[], column_key='',
                           ROW_CONDS=None, row_keys=[], row_key='',
                           COLOR_CONDS = None, color_keys=[], color_key='',
                           fig_preset=' ',
                           xbar=0., xbarlabel='',
                           ybar=0., ybarlabel='',
                           with_std=True,
                           with_screen_inset=False,
                           with_stim=True,
                           with_axis=False,
                           with_stat_test=False, stat_test_props=dict(interval_pre=[-1,0],
                                                                      interval_post=[1,2],
                                                                      test='wilcoxon',
                                                                      positive=True),
                           with_annotation=False,
                           color='k',
                           label='',
                           ylim=None, xlim=None,
                           fig=None, AX=None, no_set=True, verbose=False):

        response_args = dict(roiIndex=roiIndex, roiIndices=roiIndices)


        if with_screen_inset and (self.data.visual_stim is None):
            print('initializing stim [...]')
            self.data.init_visual_stim()
        
        if condition is None:
            condition = np.ones(np.sum(self.protocol_cond_in_full_data), dtype=bool)
        elif len(condition)==len(self.protocol_cond_in_full_data):
            condition = condition[self.protocol_cond_in_full_data]
            
        # ----- building conditions ------

        # columns
        if column_key!='':
            COL_CONDS = self.data.get_stimulus_conditions([np.sort(np.unique(self.data.nwbfile.stimulus[column_key].data[self.protocol_cond_in_full_data]))], [column_key], self.protocol_id)
        elif len(column_keys)>0:
            COL_CONDS = self.data.get_stimulus_conditions([np.sort(np.unique(self.data.nwbfile.stimulus[key].data[self.protocol_cond_in_full_data])) for key in column_keys],
                                                       column_keys, self.protocol_id)
        elif (COL_CONDS is None):
            COL_CONDS = [np.ones(np.sum(self.protocol_cond_in_full_data), dtype=bool)]

        # rows
        if row_key!='':
            ROW_CONDS = self.data.get_stimulus_conditions([np.sort(np.unique(self.data.nwbfile.stimulus[row_key].data[self.protocol_cond_in_full_data]))], [row_key], self.protocol_id)
        elif row_keys!='':
            ROW_CONDS = self.data.get_stimulus_conditions([np.sort(np.unique(self.data.nwbfile.stimulus[key].data[self.protocol_cond_in_full_data])) for key in row_keys],
                                                       row_keys, self.protocol_id)
        elif (ROW_CONDS is None):
            ROW_CONDS = [np.ones(np.sum(self.protocol_cond_in_full_data), dtype=bool)]
            
        # colors
        if color_key!='':
            COLOR_CONDS = self.data.get_stimulus_conditions([np.sort(np.unique(self.data.nwbfile.stimulus[color_key].data[self.protocol_cond_in_full_data]))], [color_key], self.protocol_id)
        elif color_keys!='':
            COLOR_CONDS = self.data.get_stimulus_conditions([np.sort(np.unique(self.data.nwbfile.stimulus[key].data[self.protocol_cond_in_full_data])) for key in color_keys],
                                                       color_keys, self.protocol_id)
        elif COLOR_CONDS is None:
            COLOR_CONDS = [np.ones(np.sum(self.protocol_cond_in_full_data), dtype=bool)]

        if (len(COLOR_CONDS)>1):
            COLORS = [ge.tab10((c%10)/10.) for c in np.arange(len(COLOR_CONDS))]
        else:
            COLORS = [color for ic in range(len(COLOR_CONDS))]
            
        # single-value
        # condition = [...]
                
        if (fig is None) and (AX is None):
            fig, AX = ge.figure(axes=(len(COL_CONDS), len(ROW_CONDS)),
                                **dv_tools.FIGURE_PRESETS[fig_preset])
            no_set=False
        else:
            no_set=no_set

        # response reshape in 
        response = self.get_response(**response_args)
        
        self.ylim = [np.inf, -np.inf]
        for irow, row_cond in enumerate(ROW_CONDS):
            for icol, col_cond in enumerate(COL_CONDS):
                for icolor, color_cond in enumerate(COLOR_CONDS):
                    
                    cond = np.array(condition & col_cond & row_cond & color_cond)[:response.shape[0]]
                    
                    if response[cond,:].shape[0]>0:

                        my = response[cond, :].mean(axis=0)
                        if with_std:
                            sy = response[cond, :].std(axis=0)
                            ge.plot(self.t, my, sy=sy,
                                    ax=AX[irow][icol], color=COLORS[icolor], lw=1)
                            self.ylim = [min([self.ylim[0], np.min(my-sy)]),
                                         max([self.ylim[1], np.max(my+sy)])]
                        else:
                            AX[irow][icol].plot(self.t, my,
                                                color=COLORS[icolor], lw=1)
                            self.ylim = [min([self.ylim[0], np.min(my)]),
                                         max([self.ylim[1], np.max(my)])]

                            
                    if with_screen_inset:
                        inset = ge.inset(AX[irow][icol], [.83, .9, .3, .25])
                        self.data.visual_stim.plot_stim_picture(self.index_from_start[cond][0],
                                                           ax=inset)
                        
                    if with_annotation:
                        
                        # column label
                        if (len(COL_CONDS)>1) and (irow==0) and (icolor==0):
                            s = ''
                            for i, key in enumerate(self.varied_parameters.keys()):
                                if (key==column_key) or (key in column_keys):
                                    s+=format_key_value(key, getattr(self, key)[cond][0])+',' # should have a unique value
                            # ge.annotate(AX[irow][icol], s, (1, 1), ha='right', va='bottom', size='small')
                            ge.annotate(AX[irow][icol], s[:-1], (0.5, 1), ha='center', va='bottom', size='small')
                        # row label
                        if (len(ROW_CONDS)>1) and (icol==0) and (icolor==0):
                            s = ''
                            for i, key in enumerate(self.varied_parameters.keys()):
                                if (key==row_key) or (key in row_keys):
                                    try:
                                        s+=format_key_value(key, getattr(self, key)[cond][0])+', ' # should have a unique value
                                    except IndexError:
                                        pass
                            ge.annotate(AX[irow][icol], s[:-2], (0, 0), ha='right', va='bottom', rotation=90, size='small')
                        # n per cond
                        ge.annotate(AX[irow][icol], ' n=%i'%np.sum(cond)+'\n'*icolor,
                                    (.99,0), color=COLORS[icolor], size='xx-small',
                                    ha='left', va='bottom')
                        # color label
                        if (len(COLOR_CONDS)>1) and (irow==0) and (icol==0):
                            s = ''
                            for i, key in enumerate(self.varied_parameters.keys()):
                                if (key==color_key) or (key in color_keys):
                                    s+=20*' '+icolor*18*' '+format_key_value(key, getattr(self, key)[cond][0])
                                    ge.annotate(fig, s+'  ', (1,0), color=COLORS[icolor], ha='right', va='bottom', size='small')
                    
        if with_stat_test:
            for irow, row_cond in enumerate(ROW_CONDS):
                for icol, col_cond in enumerate(COL_CONDS):
                    for icolor, color_cond in enumerate(COLOR_CONDS):
                        
                        cond = np.array(condition & col_cond & row_cond & color_cond)[:response.shape[0]]
                        results = self.stat_test_for_evoked_responses(episode_cond=cond, response_args=response_args, **stat_test_props)

                        ps, size = results.pval_annot()
                        AX[irow][icol].annotate(icolor*'\n'+ps, ((stat_test_props['interval_post'][0]+stat_test_props['interval_pre'][1])/2.,
                                                                 self.ylim[0]), va='top', ha='center', size=size-1, xycoords='data', color=COLORS[icolor])
                        AX[irow][icol].plot(stat_test_props['interval_pre'], self.ylim[0]*np.ones(2), 'k-', lw=1)
                        AX[irow][icol].plot(stat_test_props['interval_post'], self.ylim[0]*np.ones(2), 'k-', lw=1)
                            
        if xlim is None:
            self.xlim = [self.t[0], self.t[-1]]
        else:
            self.xlim = xlim
            
        if ylim is not None:
            self.ylim = ylim

            
        for irow, row_cond in enumerate(ROW_CONDS):
            for icol, col_cond in enumerate(COL_CONDS):
                if not no_set:
                    ge.set_plot(AX[irow][icol],
                                spines=(['left', 'bottom'] if with_axis else []),
                                # xlabel=(self.xbarlabel.text() if with_axis else ''),
                                # ylabel=(self.ybarlabel.text() if with_axis else ''),
                                ylim=self.ylim, xlim=self.xlim)

                if with_stim:
                    AX[irow][icol].fill_between([0, np.mean(self.time_duration)],
                                        self.ylim[0]*np.ones(2), self.ylim[1]*np.ones(2),
                                        color='grey', alpha=.2, lw=0)

        if not with_axis and not no_set:
            ge.draw_bar_scales(AX[0][0],
                               Xbar=xbar, Xbar_label=xbarlabel,
                               Ybar=ybar,  Ybar_label=ybarlabel,
                               Xbar_fraction=0.1, Xbar_label_format='%.1f',
                               Ybar_fraction=0.2, Ybar_label_format='%.1f',
                               loc='top-left')

        if label!='':
            ge.annotate(fig, label, (0,0), color=color, ha='left', va='bottom')

        if with_annotation:
            S = ''
            if hasattr(self, 'rawFluo') or hasattr(self, 'dFoF') or hasattr(self, 'neuropil'):
                if roiIndex is not None:
                    S+='roi #%i' % roiIndex
                elif roiIndices in ['sum', 'mean', 'all']:
                    S+='mean: n=%i rois' % len(self.data.valid_roiIndices)
                else:
                    S+='mean: n=%i rois\n' % len(roiIndices)
            # for i, key in enumerate(self.varied_parameters.keys()):
            #     if 'single-value' in getattr(self, '%s_plot' % key).currentText():
            #         S += ', %s=%.2f' % (key, getattr(self, '%s_values' % key).currentText())
            ge.annotate(fig, S, (0,0), color='k', ha='left', va='bottom', size='small')
            
        return fig, AX

    ####

    def plot_evoked_pattern(self, 
                            pattern_cond, 
                            quantity='rawFluo',
                            rois=None,
                            with_stim_inset=True,
                            with_mean_trace=False,
                            factor_for_traces=2,
                            raster_norm='full',
                            Tbar=1, Nbar=None,
                            min_dFof_range=4,
                            figsize=(1.3,.3), axR=None, axT=None):

        resp = np.array(getattr(self, quantity))

        if Nbar is None:
            Nbar = int(resp.shape[1]/4)
        if rois is None:
            rois = np.random.choice(np.arange(resp.shape[1]), 5, replace=False)

        if (axR is None) or (axT is None):
            fig, [axR, axT] = ge.figure(axes_extents=[[[1,3]],
                                                      [[1,int(3*factor_for_traces)]]], 
                                        figsize=figsize, left=0.3,
                                        top=(12 if with_stim_inset else 1),
                                        right=3)
        else:
            fig = None

        
        first_pattern_resp_index = np.arange(len(pattern_cond))[pattern_cond][0]
        if with_stim_inset:
            if self.data.visual_stim is None:
                self.data.init_visual_stim()
            stim_inset = ge.inset(axR, [0.2,1.3,0.6,0.6])
            self.data.visual_stim.plot_stim_picture(first_pattern_resp_index,
                                                    ax=stim_inset,
                                                    enhance=True,
                                                    vse=True)
            if hasattr(self.data.visual_stim, 'get_vse'):
                vse = self.data.visual_stim.get_vse(first_pattern_resp_index)

        # mean response for raster
        mean_resp = resp[pattern_cond,:,:].mean(axis=0)
        if raster_norm=='full':
            mean_resp = (mean_resp-mean_resp.min(axis=1).reshape(resp.shape[1],1))
        else:
            pass

        # raster
        axR.imshow(mean_resp,
                   cmap=ge.binary,
                   aspect='auto', interpolation='none',
                   vmin=0, vmax=2, 
                   #origin='lower',
                   extent = (self.t[0], self.t[-1],
                             0, resp.shape[1]))

        ge.set_plot(axR, [], xlim=[self.t[0], self.t[-1]])
        axR.plot([self.t[0], self.t[0]], [0, Nbar], 'k-', lw=2)
        ge.annotate(axR, '%i rois' % Nbar, (self.t[0], 0), rotation=90, xycoords='data', ha='right')
        ge.annotate(axR, 'n=%i trials' % np.sum(pattern_cond), (self.t[-1], resp.shape[1]),
                    xycoords='data', ha='right', size='x-small')

        # raster_bar_inset = ge.inset(axR, [0.2,1.3,0.6,0.6])
        ge.bar_legend(axR, 
                      colorbar_inset=dict(rect=[1.1,.1,.04,.8], facecolor=None),
                      colormap=ge.binary,
                      bar_legend_args={},
                      label='n. $\Delta$F/F',
                      bounds=None,
                      ticks = None,
                      ticks_labels=None,
                      no_ticks=False,
                      orientation='vertical')

        for ir, r in enumerate(rois):
            roi_resp = resp[pattern_cond, r, :]
            roi_resp = roi_resp-roi_resp.mean()
            scale = max([min_dFof_range, np.max(roi_resp)])
            roi_resp /= scale
            axT.plot([self.t[-1], self.t[-1]], [.25+ir, .25+ir+1./scale], 'k-', lw=2)

            if with_mean_trace:
                ge.plot(self.t, ir+roi_resp.mean(axis=0), 
                        sy=roi_resp.std(axis=0),ax=axT, no_set=True)
            ge.annotate(axT, 'roi#%i' % (r+1), (self.t[0], ir), xycoords='data',
                        rotation=90, ha='right', size='xx-small')
            for iep in range(np.sum(pattern_cond)):
                axT.plot(self.t, ir+roi_resp[iep,:], color=ge.tab10(iep/(np.sum(pattern_cond)-1)), lw=.5)
        ge.annotate(axT, '1$\Delta$F/F', (self.t[-1], 0), xycoords='data',
                    rotation=90, size='small')
        ge.set_plot(axT, [], xlim=[self.t[0], self.t[-1]])
        ge.draw_bar_scales(axT, Xbar=Tbar, Xbar_label=str(Tbar)+'s', Ybar=1e-12)

        ge.bar_legend(axT, X=np.arange(10),
                      colorbar_inset=dict(rect=[1.1,1-.8/factor_for_traces,
                                                .04,.8/factor_for_traces], facecolor=None),
                      colormap=ge.tab10,
                      label='trial ID',
                      no_ticks=True,
                      orientation='vertical')

        if hasattr(self.data.visual_stim, 'get_vse'):
            for t in [0]+vse['t']:
                axR.plot([t,t], axR.get_ylim(), 'r-', lw=0.3)
                axT.plot([t,t], axT.get_ylim(), 'r-', lw=0.3)
                
        return fig
    
    ###-------------------------------------------
    ### ----- Single Trial population response  --
    ###-------------------------------------------

    def single_trial_rasters(self,
                             protocol_id=0,
                             quantity='Photodiode-Signal', subquantity='dF/F',
                             Nmax=10000000,
                             condition=None,
                             row_key = 'repeat',
                             column_key=None, column_keys=None, 
                             dt_sampling=10, # ms
                             interpolation='linear',
                             baseline_substraction=False,
                             with_screen_inset=False,
                             Tsubsampling=1,
                             fig_preset='raster-preset',
                             fig=None, AX=None, verbose=False, Tbar=2):

        ALL_ROIS = []
        for roi in np.arange(np.min([Nmax, np.sum(self.iscell)])):
            # ----- building episodes of single cell response ------
            if verbose:
                print('computing roi #', roi, ' for single trial raster plot')
            ALL_ROIS.append(process_NWB.EpisodeResponse(self,
                                                        protocol_id=protocol_id,
                                                        quantity=quantity,
                                                        subquantity=subquantity,
                                                        roiIndex=roi,
                                                        dt_sampling=dt_sampling,
                                                        prestim_duration=2,
                                                        verbose=verbose))
        
        # ----- protocol cond ------
        Pcond = self.get_protocol_cond(self.protocol_id)

        # build column conditions
        if column_key is not None:
            column_keys = [column_key]
        if column_keys is None:
            column_keys = [k for k in ALL_ROIS[0].varied_parameters.keys() if k!='repeat']
        COL_CONDS = self.data.get_stimulus_conditions([np.sort(np.unique(self.data.nwbfile.stimulus[key].data[Pcond])) for key in column_keys],
                                                 column_keys, protocol_id)

        # build row conditions
        ROW_CONDS = self.data.get_stimulus_conditions([np.sort(np.unique(self.data.nwbfile.stimulus[row_key].data[Pcond]))],
                                                 [row_key], protocol_id)

        if with_screen_inset and (self.data.visual_stim is None):
            print('initializing stim [...]')
            self.init_visual_stim()
        
        if condition is None:
            condition = np.ones(np.sum(Pcond), dtype=bool)
        elif len(condition)==len(Pcond):
            condition = condition[Pcond]

            
        if (fig is None) and (AX is None):
            fig, AX = ge.figure(axes=(len(COL_CONDS), len(ROW_CONDS)),
                                **dv_tools.FIGURE_PRESETS[fig_preset])
            no_set=False
        else:
            no_set=True
        
        for irow, row_cond in enumerate(ROW_CONDS):
            for icol, col_cond in enumerate(COL_CONDS):
                
                cond = np.array(condition & col_cond & row_cond)[:ALL_ROIS[0].resp.shape[0]]

                if np.sum(cond)==1:
                    resp = np.zeros((len(ALL_ROIS), ALL_ROIS[0].resp.shape[1]))
                    for roi in range(len(ALL_ROIS)):
                        norm = (ALL_ROIS[roi].resp[cond,:].max()-ALL_ROIS[roi].resp[cond,:].min())
                        if norm>0:
                            resp[roi,:] = (ALL_ROIS[roi].resp[cond,:]-ALL_ROIS[roi].resp[cond,:].min())/norm
                        AX[irow][icol].imshow(resp[:,::Tsubsampling],
                                              cmap=plt.cm.binary,
                                              aspect='auto', interpolation='none',
                                              vmin=0, vmax=1, origin='lower',
                                              extent = (ALL_ROIS[0].t[0], ALL_ROIS[0].t[-1],
                                                        0, len(ALL_ROIS)-1))
                        
                    # row label
                    if (icol==0):
                        ge.annotate(AX[irow][icol],
                                    format_key_value('repeat', getattr(ALL_ROIS[0], 'repeat')[cond][0]),
                                    (0, 0), ha='right', va='bottom', rotation=90, size='small')
                    # column label
                    if (irow==0):
                        s = ''
                        for i, key in enumerate(column_keys):
                            s+=format_key_value(key, getattr(ALL_ROIS[0], key)[cond][0])+', '
                        ge.annotate(AX[irow][icol], s[:-2], (1, 1), ha='right', va='bottom', size='small')
                        if with_screen_inset:
                            inset = ge.inset(AX[0][icol], [0.2, 1.2, .8, .8])
                            if 'center-time' in self.data.nwbfile.stimulus:
                                t0 = self.data.nwbfile.stimulus['center-time'].data[np.argwhere(cond)[0][0]]
                            else:
                                t0 = 0
                            self.data.visual_stim.show_frame(ALL_ROIS[0].index_from_start[np.argwhere(cond)[0][0]],
                                                             time_from_episode_start=t0,
                                                             ax=inset,
                                                             label=({'degree':15,
                                                                     'shift_factor':0.03,
                                                                     'lw':0.5, 'fontsize':7} if (icol==1) else None))
                                            
                AX[irow][icol].axis('off')

        # dF/F bar legend
        ge.bar_legend(AX[0][0],
                      continuous=False, colormap=plt.cm.binary,
                      colorbar_inset=dict(rect=[-.8, -.2, 0.1, 1.], facecolor=None),
                      color_discretization=100, no_ticks=True, labelpad=4.,
                      label='norm F', fontsize='small')

        ax_time = ge.inset(AX[0][0], [0., 1.1, 1., 0.1])
        ax_time.plot([ALL_ROIS[0].t[0],ALL_ROIS[0].t[0]+Tbar], [0,0], 'k-', lw=1)
        ge.annotate(ax_time, '%is' % Tbar, (ALL_ROIS[0].t[0],0), xycoords='data')
        ax_time.set_xlim((ALL_ROIS[0].t[0],ALL_ROIS[0].t[-1]))
        ax_time.axis('off')

        ge.annotate(AX[0][0],'%i ROIs ' % len(ALL_ROIS), (0,1), ha='right', va='top')

        return fig, AX
    

def format_key_value(key, value):
    if key in ['angle','direction']:
        return '$\\theta$=%.0f$^{o}$' % value
    elif key=='x-center':
        return '$x$=%.0f$^{o}$' % value
    elif key=='y-center':
        return '$y$=%.0f$^{o}$' % value
    elif key=='radius':
        return '$r$=%.0f$^{o}$' % value
    elif key=='size':
        return '$s$=%.0f$^{o}$' % value
    elif key=='contrast':
        return '$c$=%.1f' % value
    elif key=='repeat':
        return 'trial #%i' % (value+1)
    elif key=='center-time':
        return '$t_0$:%.1fs' % value
    elif key=='Image-ID':
        return 'im#%i' % value
    elif key=='VSE-seed':
        return 'vse#%i' % value
    elif key=='light-level':
        if value==0:
            return 'grey'
        elif value==1:
            return 'white'
        else:
            return 'lum.=%.1f' % value
    elif key=='dotcolor':
        if value==-1:
            return 'black dot'
        elif value==0:
            return 'grey dot'
        elif value==1:
            return 'white dot'
        else:
            return 'dot=%.1f' % value
    elif key=='color':
        if value==-1:
            return 'black'
        elif value==0:
            return 'grey'
        elif value==1:
            return 'white'
        else:
            return 'color=%.1f' % value
    elif key=='speed':
        return 'v=%.0f$^{o}$/s' % value
    elif key=='protocol_id':
        return 'p.#%i' % (value+1)
    else:
        return '%s=%.2f' % (key, value)

    
     
if __name__=='__main__':

    import argparse

    parser=argparse.ArgumentParser()
    parser.add_argument("datafile", type=str)
    parser.add_argument('-o', "--ops", default='raw', help='')
    parser.add_argument("--tlim", type=float, nargs='*', default=[10, 50], help='')
    parser.add_argument('-e', "--episode", type=int, default=0)
    parser.add_argument('-nmax', "--Nmax", type=int, default=20)
    parser.add_argument("--Npanels", type=int, default=8)
    parser.add_argument('-roi', "--roiIndex", type=int, default=0)
    parser.add_argument('-pid', "--protocol_id", type=int, default=0)
    parser.add_argument('-q', "--quantity", type=str, default='dFoF')
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")

    args = parser.parse_args()
    

    if args.ops=='raw':
        data = MultimodalData(args.datafile)
        data.plot_raw_data(args.tlim, 
                  settings={'CaImagingRaster':dict(fig_fraction=4, subsampling=1,
                                                   roiIndices='all',
                                                   normalization='per-line',
                                                   subquantity='dF/F'),
                            'CaImaging':dict(fig_fraction=3, subsampling=1, 
                                             subquantity='dF/F', color='#2ca02c',
                                             roiIndices=np.sort(np.random.choice(np.arange(np.sum(data.iscell)), np.min([args.Nmax, data.iscell.sum()]), replace=False))),
                            'Locomotion':dict(fig_fraction=1, subsampling=1, color='#1f77b4'),
                            'Pupil':dict(fig_fraction=2, subsampling=1, color='#d62728'),
                            'GazeMovement':dict(fig_fraction=1, subsampling=1, color='#ff7f0e'),
                            'Photodiode':dict(fig_fraction=.5, subsampling=1, color='grey'),
                            'VisualStim':dict(fig_fraction=.5, color='black')},
                            Tbar=5)
        
    elif args.ops=='trial-average':
        episodes = EpisodeResponse(args.datafile,
                                   protocol_id=args.protocol_id,
                                   quantities=[args.quantity])
        fig, AX = episodes.plot_trial_average(quantity=args.quantity,
                                              roiIndex=args.roiIndex,
                                              column_key=list(episodes.varied_parameters.keys())[0],
                                              xbar=1, xbarlabel='1s', ybar=1, ybarlabel='1dF/F',
                                              with_stat_test=True,
                                              with_annotation=True,
                                              with_screen_inset=True,                                          
                                              fig_preset='raw-traces-preset', color='#1f77b4', label='test\n')

    elif args.ops=='evoked-raster':
        episodes = EpisodeResponse(args.datafile,
                                   protocol_id=args.protocol_id,
                                   quantities=[args.quantity])

        VP = [key for key in episodes.varied_parameters if key!='repeat'] # varied parameters except rpeat

        # single stim
        episodes.plot_evoked_pattern(episodes.find_episode_cond(np.array(VP),
                                                                np.zeros(len(VP), dtype=int)),
                                     quantity=args.quantity)
        
        
    elif args.ops=='visual-stim':
        data = MultimodalData(args.datafile)
        fig, AX = data.show_VisualStim(args.tlim, Npanels=args.Npanels, enhance=True)
        fig2 = data.visual_stim.plot_stim_picture(args.episode, enhance=True)
        print('interval [%.1f, %.1f] ' % (data.nwbfile.stimulus['time_start_realigned'].data[args.episode],
                                          data.nwbfile.stimulus['time_stop_realigned'].data[args.episode]))
        
    elif args.ops=='FOV':
        fig, ax = data.show_CaImaging_FOV('meanImg', NL=3, cmap=ge.get_linear_colormap('k', 'lightgreen'))
    else:
        print(' option not recognized !')
        
    ge.show()




