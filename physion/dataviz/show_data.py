# general modules
import pynwb, os, sys, pathlib
import numpy as np
import matplotlib.pylab as plt

# custom modules
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from dataviz import tools as dv_tools
from analysis.read_NWB import Data
from analysis import stat_tools, process_NWB
from Ca_imaging.tools import compute_CaImaging_trace, compute_CaImaging_raster
from visual_stim.psychopy_code.stimuli import build_stim
from datavyz import graph_env_manuscript as ge

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

    ###-------------------------------------
    ### ----- RAW DATA PLOT components -----
    ###-------------------------------------

    def shifted_start(self, tlim, frac_shift=0.01):
        return tlim[0]-0.01*(tlim[1]-tlim[0])
    
    def plot_scaled_signal(self, ax, t, signal, tlim, scale_bar, ax_fraction_extent, ax_fraction_start,
                           color=ge.blue, scale_unit_string='%.1f'):
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
            ge.annotate(ax, str(scale_unit_string+' ') % scale_bar, (self.shifted_start(tlim), ax_fraction_start), ha='right', color=color, va='center', xycoords='data')

    def add_name_annotation(self, ax, name, tlim, ax_fraction_extent, ax_fraction_start,
                            color='k', rotation=0):
        ge.annotate(ax, ' '+name, (tlim[1], ax_fraction_extent/2.+ax_fraction_start), xycoords='data', color=color, va='center', rotation=rotation)
        
    def add_Photodiode(self, tlim, ax,
                       fig_fraction_start=0., fig_fraction=1., subsampling=10, color=ge.grey, name='photodiode'):
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
                       color=ge.blue, name='run. speed'):
        i1, i2 = dv_tools.convert_times_to_indices(*tlim, self.nwbfile.acquisition['Running-Speed'])
        t = dv_tools.convert_index_to_time(range(i1,i2), self.nwbfile.acquisition['Running-Speed'])[::subsampling]
        y = self.nwbfile.acquisition['Running-Speed'].data[i1:i2][::subsampling]

        self.plot_scaled_signal(ax, t, y, tlim, speed_scale_bar, fig_fraction, fig_fraction_start, color=color, scale_unit_string='%.1fcm/s')        
        self.add_name_annotation(ax, name, tlim, fig_fraction, fig_fraction_start, color=color)
        
    def add_FaceMotion(self, tlim, ax,
                       fig_fraction_start=0., fig_fraction=1., subsampling=2, color=ge.purple, name='facemotion'):
        i1, i2 = dv_tools.convert_times_to_indices(*tlim, self.nwbfile.processing['FaceMotion'].data_interfaces['face-motion'])
        t = self.nwbfile.processing['FaceMotion'].data_interfaces['face-motion'].timestamps[i1:i2]
        motion = self.nwbfile.processing['FaceMotion'].data_interfaces['face-motion'].data[i1:i2]
        x, y = t[::subsampling], motion[::subsampling]

        self.plot_scaled_signal(ax, x, y, tlim, 1., fig_fraction, fig_fraction_start, color=color, scale_unit_string='') # no scale bar here
        self.add_name_annotation(ax, name, tlim, fig_fraction, fig_fraction_start, color=color)

    def add_Pupil(self, tlim, ax,
                  fig_fraction_start=0., fig_fraction=1., subsampling=2,
                  pupil_scale_bar = 0.5, # scale bar in mm
                  color='red', name='pupil diam.'):
        i1, i2 = dv_tools.convert_times_to_indices(*tlim, self.nwbfile.processing['Pupil'].data_interfaces['cx'])
        if not hasattr(self, 't_pupil'):
            self.build_pupil_diameter()
        x, y = self.t_pupil[::subsampling], self.pupil_diameter[::subsampling]

        self.plot_scaled_signal(ax, x, y, tlim, pupil_scale_bar, fig_fraction, fig_fraction_start, color=color, scale_unit_string='%.1fmm')        
        self.add_name_annotation(ax, name, tlim, fig_fraction, fig_fraction_start, color=color)

        
    def add_GazeMovement(self, tlim, ax,
                         fig_fraction_start=0., fig_fraction=1., subsampling=2,
                         gaze_scale_bar = 0.2, # scale bar in mm
                         color=ge.orange, name='gaze mov.'):
        i1, i2 = dv_tools.convert_times_to_indices(*tlim, self.nwbfile.processing['Pupil'].data_interfaces['cx'])
        t = self.nwbfile.processing['Pupil'].data_interfaces['sx'].timestamps[i1:i2]
        cx = self.nwbfile.processing['Pupil'].data_interfaces['cx'].data[i1:i2]
        cy = self.nwbfile.processing['Pupil'].data_interfaces['cy'].data[i1:i2]
        mov = np.sqrt((cx-np.mean(cx))**2+(cy-np.mean(cy))**2)
        
        x, y = t[::subsampling], mov[::subsampling]

        self.plot_scaled_signal(ax, x, y, tlim, gaze_scale_bar, fig_fraction, fig_fraction_start, color=color, scale_unit_string='%.1fmm')
        self.add_name_annotation(ax, name, tlim, fig_fraction, fig_fraction_start, color=color)

        
    def add_CaImagingRaster(self, tlim, ax,
                            fig_fraction_start=0., fig_fraction=1., color='green',
                            quantity='CaImaging', subquantity='Fluorescence', roiIndices='all',
                            cmap=plt.cm.binary,
                            normalization='None', subsampling=1,
                            name='\nROIs'):

        raster = compute_CaImaging_raster(self, subquantity,
                                          roiIndices=roiIndices,
                                          normalization=normalization) # validROI indices inside !!
        indices=np.arange(*dv_tools.convert_times_to_indices(*tlim, self.Neuropil, axis=1))[::subsampling]
        
        ax.imshow(raster[:,indices], origin='lower', cmap=cmap,
                  aspect='auto', interpolation='none',
                  extent=(dv_tools.convert_index_to_time(indices[0], self.Neuropil),
                          dv_tools.convert_index_to_time(indices[-1], self.Neuropil),
                          fig_fraction_start, fig_fraction_start+fig_fraction))

        if normalization in ['per line', 'per-line', 'per cell', 'per-cell']:
            ge.bar_legend(ax,
                          # X=[0,1], bounds=[0,1],
                          continuous=False, colormap=cmap,
                          colorbar_inset=dict(rect=[-.04,
                                           fig_fraction_start+.2*fig_fraction,
                                           .01,
                                           .6*fig_fraction], facecolor=None),
                          color_discretization=100, no_ticks=True, labelpad=4.,
                          label='norm F', fontsize='small')
        
        self.add_name_annotation(ax, name, tlim, fig_fraction, fig_fraction_start, rotation=90)

        ge.annotate(ax, '1', (tlim[1], fig_fraction_start), xycoords='data')
        ge.annotate(ax, '%i' % raster.shape[0],
                    (tlim[1], fig_fraction_start+fig_fraction), va='top', xycoords='data')
        
        
    def add_CaImaging(self, tlim, ax,
                      fig_fraction_start=0., fig_fraction=1., color='green',
                      quantity='CaImaging', subquantity='Fluorescence', roiIndices='all',
                      vicinity_factor=1, subsampling=1, name='[Ca] imaging'):
        
        if (type(roiIndices)==str) and roiIndices=='all':
            roiIndices = np.arange(np.sum(self.iscell))
        if color=='tab':
            COLORS = [plt.cm.tab10(n%10) for n in range(len(roiIndices))]
        else:
            COLORS = [str(color) for n in range(len(roiIndices))]

        dF = compute_CaImaging_trace(self, subquantity, roiIndices) # validROI indices inside !!
        i1, i2 = dv_tools.convert_times_to_indices(*tlim, self.Neuropil, axis=1)
        t = np.array(self.Neuropil.timestamps[:])[np.arange(i1,i2)][::subsampling]
        
        for n, ir in zip(range(len(roiIndices))[::-1], roiIndices[::-1]):
            ypos = n*fig_fraction/len(roiIndices)/vicinity_factor+fig_fraction_start # bottom position
            y = dF[n, np.arange(i1,i2)][::subsampling]

            self.plot_scaled_signal(ax, t, y, tlim, 1., fig_fraction/len(roiIndices), ypos, color=color,
                                    scale_unit_string=('%.0fdF/F' if ((n==0) and subquantity in ['dF/F', 'dFoF']) else ''))
            
            self.add_name_annotation(ax, ' ROI#%i'%(ir+1), tlim, fig_fraction/len(roiIndices), ypos, color=color)
            
        # ge.annotate(ax, name, (self.shifted_start(tlim), fig_fraction/2.+fig_fraction_start), color=color,
        #             xycoords='data', ha='right', va='center', rotation=90)
            

    def add_CaImagingSum(self, tlim, ax,
                         fig_fraction_start=0., fig_fraction=1., color='green',
                         quantity='CaImaging', subquantity='Fluorescence', subsampling=1,
                         name='Sum [Ca]'):
        i1 = dv_tools.convert_time_to_index(tlim[0], self.Neuropil, axis=1)
        i2 = dv_tools.convert_time_to_index(tlim[1], self.Neuropil, axis=1)
        t = np.array(self.Neuropil.timestamps[:])[np.arange(i1,i2)][::subsampling]
        y = compute_CaImaging_trace(self, subquantity, np.arange(np.sum(self.iscell))).sum(axis=0)[np.arange(i1,i2)][::subsampling]

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
                        Npanels=8):
        
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
                # arrow = self.visual_stim.get_arrow(iEp, self,
                #             arrow_props={'length':25, 'width_factor':0.1})
                self.visual_stim.show_frame(iEp, ax=AX[i],
                                            time_from_episode_start=ti-tEp,
                                            label=label)
            # else:
            #     self.visual_stim.show_interstim(AX[i])
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
                      figsize=(15,6), Tbar=0.,
                      ax=None, ax_raster=None):

        if ('CaImaging' in settings) and ('raster' in settings['CaImaging']) and (ax_raster is None):
            fig, [ax, ax_raster] = ge.figure(axes=(1,2), figsize=(3,1), bottom=.3, left=.5, hspace=0.)
        if ax is None:
            fig, ax = ge.figure(figsize=(3,2), bottom=.3, left=.5)
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
        ax.annotate((' %is' % Tbar if Tbar>1 else  '%.1fs' % Tbar) ,
                    [self.shifted_start(tlim), 1.02], color='k', fontsize=9)
        
        ax.axis('off')
        ax.set_xlim([self.shifted_start(tlim)-0.01*(tlim[1]-tlim[0]),tlim[1]+0.01*(tlim[1]-tlim[0])])
        ax.set_ylim([-0.05,1.05])
        
        return fig, ax

    
    ###------------------------------------------
    ### ----- Trial Average plot components -----
    ###------------------------------------------

    def plot_trial_average(self,
                           # episodes props
                           EPISODES=None,
                           protocol_id=0, quantity='Photodiode-Signal', subquantity='dF/F', roiIndex=0,
                           dt_sampling=1, # ms
                           interpolation='linear',
                           baseline_substraction=False,
                           prestim_duration=None,
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
                           with_stat_test=False, stat_test_props=dict(interval_pre=[-1,0], interval_post=[1,2], test='wilcoxon', positive=True),
                           with_annotation=False,
                           color='k',
                           label='',
                           ylim=None, xlim=None,
                           fig=None, AX=None, verbose=False):

        # ----- protocol cond ------
        Pcond = self.get_protocol_cond(protocol_id)

        if EPISODES is None:
            # ----- building episodes of cell response ------
            EPISODES = process_NWB.EpisodeResponse(self,
                                                   protocol_id=protocol_id,
                                                   quantity=quantity,
                                                   subquantity=subquantity,
                                                   roiIndex=roiIndex,
                                                   dt_sampling=dt_sampling,
                                                   verbose=verbose)

        if with_screen_inset and (self.visual_stim is None):
            print('initializing stim [...]')
            self.init_visual_stim()
        
        if condition is None:
            condition = np.ones(np.sum(Pcond), dtype=bool)
        elif len(condition)==len(Pcond):
            condition = condition[Pcond]
            
        # ----- building conditions ------

        # columns
        if column_key!='':
            COL_CONDS = self.get_stimulus_conditions([np.sort(np.unique(self.nwbfile.stimulus[column_key].data[Pcond]))], [column_key], protocol_id)
        elif len(column_keys)>0:
            COL_CONDS = self.get_stimulus_conditions([np.sort(np.unique(self.nwbfile.stimulus[key].data[Pcond])) for key in column_keys],
                                                       column_keys, protocol_id)
        elif (COL_CONDS is None):
            COL_CONDS = [np.ones(np.sum(Pcond), dtype=bool)]

        # rows
        if row_key!='':
            ROW_CONDS = self.get_stimulus_conditions([np.sort(np.unique(self.nwbfile.stimulus[row_key].data[Pcond]))], [row_key], protocol_id)
        elif row_keys!='':
            ROW_CONDS = self.get_stimulus_conditions([np.sort(np.unique(self.nwbfile.stimulus[key].data[Pcond])) for key in row_keys],
                                                       row_keys, protocol_id)
        elif (ROW_CONDS is None):
            ROW_CONDS = [np.ones(np.sum(Pcond), dtype=bool)]
            
        # colors
        if color_key!='':
            COLOR_CONDS = self.get_stimulus_conditions([np.sort(np.unique(self.nwbfile.stimulus[color_key].data[Pcond]))], [color_key], protocol_id)
        elif color_keys!='':
            COLOR_CONDS = self.get_stimulus_conditions([np.sort(np.unique(self.nwbfile.stimulus[key].data[Pcond])) for key in color_keys],
                                                       color_keys, protocol_id)
        elif COLOR_CONDS is None:
            COLOR_CONDS = [np.ones(np.sum(Pcond), dtype=bool)]

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
            no_set=True

        self.ylim = [np.inf, -np.inf]
        for irow, row_cond in enumerate(ROW_CONDS):
            for icol, col_cond in enumerate(COL_CONDS):
                for icolor, color_cond in enumerate(COLOR_CONDS):
                    
                    cond = np.array(condition & col_cond & row_cond & color_cond)[:EPISODES.resp.shape[0]]
                    
                    if EPISODES.resp[cond,:].shape[0]>0:
                        my = EPISODES.resp[cond,:].mean(axis=0)
                        if with_std:
                            sy = EPISODES.resp[cond,:].std(axis=0)
                            ge.plot(EPISODES.t, my, sy=sy,
                                    ax=AX[irow][icol], color=COLORS[icolor], lw=1)
                            self.ylim = [min([self.ylim[0], np.min(my-sy)]),
                                         max([self.ylim[1], np.max(my+sy)])]
                        else:
                            AX[irow][icol].plot(EPISODES.t, my,
                                                color=COLORS[icolor], lw=1)
                            self.ylim = [min([self.ylim[0], np.min(my)]),
                                         max([self.ylim[1], np.max(my)])]

                            
                    if with_screen_inset:
                        inset = ge.inset(AX[irow][icol], [.83, .9, .3, .25])
                        self.visual_stim.plot_stim_picture(EPISODES.index_from_start[cond][0],
                                                           ax=inset)
                        
                    if with_annotation:
                        
                        # column label
                        if (len(COL_CONDS)>1) and (irow==0) and (icolor==0):
                            s = ''
                            for i, key in enumerate(EPISODES.varied_parameters.keys()):
                                if (key==column_key) or (key in column_keys):
                                    s+=format_key_value(key, getattr(EPISODES, key)[cond][0])+',' # should have a unique value
                            # ge.annotate(AX[irow][icol], s, (1, 1), ha='right', va='bottom', size='small')
                            ge.annotate(AX[irow][icol], s[:-1], (0.5, 1), ha='center', va='bottom', size='small')
                        # row label
                        if (len(ROW_CONDS)>1) and (icol==0) and (icolor==0):
                            s = ''
                            for i, key in enumerate(EPISODES.varied_parameters.keys()):
                                if (key==row_key) or (key in row_keys):
                                    try:
                                        s+=format_key_value(key, getattr(EPISODES, key)[cond][0])+', ' # should have a unique value
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
                            for i, key in enumerate(EPISODES.varied_parameters.keys()):
                                if (key==color_key) or (key in color_keys):
                                    s+=20*' '+icolor*18*' '+format_key_value(key, getattr(EPISODES, key)[cond][0])
                                    ge.annotate(fig, s, (0,0), color=COLORS[icolor], ha='left', va='bottom', size='small')
                    
        if with_stat_test:
            for irow, row_cond in enumerate(ROW_CONDS):
                for icol, col_cond in enumerate(COL_CONDS):
                    for icolor, color_cond in enumerate(COLOR_CONDS):
                        
                        cond = np.array(condition & col_cond & row_cond & color_cond)[:EPISODES.resp.shape[0]]
                        results = EPISODES.stat_test_for_evoked_responses(episode_cond=cond, **stat_test_props)
                        ps, size = results.pval_annot()
                        AX[irow][icol].annotate(icolor*'\n'+ps, ((stat_test_props['interval_post'][0]+stat_test_props['interval_pre'][1])/2.,
                                                                 self.ylim[0]), va='top', ha='center', size=size-1, xycoords='data', color=COLORS[icolor])
                        AX[irow][icol].plot(stat_test_props['interval_pre'], self.ylim[0]*np.ones(2), 'k-', lw=1)
                        AX[irow][icol].plot(stat_test_props['interval_post'], self.ylim[0]*np.ones(2), 'k-', lw=1)
                            
        if xlim is None:
            self.xlim = [EPISODES.t[0], EPISODES.t[-1]]
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
                    AX[irow][icol].fill_between([0, np.mean(EPISODES.time_duration)],
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
            if quantity=='CaImaging':
                S+='roi #%i' % (roiIndex+1)
            # for i, key in enumerate(EPISODES.varied_parameters.keys()):
            #     if 'single-value' in getattr(self, '%s_plot' % key).currentText():
            #         S += ', %s=%.2f' % (key, getattr(self, '%s_values' % key).currentText())
            ge.annotate(fig, S, (0,0), color='k', ha='left', va='bottom')
            
        return fig, AX
    
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

        # ----- protocol cond ------
        Pcond = self.get_protocol_cond(protocol_id)

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
        

        # build column conditions
        if column_key is not None:
            column_keys = [column_key]
        if column_keys is None:
            column_keys = [k for k in ALL_ROIS[0].varied_parameters.keys() if k!='repeat']
        COL_CONDS = self.get_stimulus_conditions([np.sort(np.unique(self.nwbfile.stimulus[key].data[Pcond])) for key in column_keys],
                                                 column_keys, protocol_id)

        # build row conditions
        ROW_CONDS = self.get_stimulus_conditions([np.sort(np.unique(self.nwbfile.stimulus[row_key].data[Pcond]))],
                                                 [row_key], protocol_id)

        if with_screen_inset and (self.visual_stim is None):
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
                            if 'center-time' in self.nwbfile.stimulus:
                                t0 = self.nwbfile.stimulus['center-time'].data[np.argwhere(cond)[0][0]]
                            else:
                                t0 = 0
                            self.visual_stim.show_frame(ALL_ROIS[0].index_from_start[np.argwhere(cond)[0][0]],
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
                                (self.pixel_masks_index[roiIndex] if roiIndex<len(self.validROI_indices) else len(self.pixel_masks_index)))
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
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")

    args = parser.parse_args()
    
    data = MultimodalData(args.datafile)

    if args.ops=='raw':
        data.plot_raw_data(args.tlim, 
                  settings={'CaImagingRaster':dict(fig_fraction=4, subsampling=1,
                                                   roiIndices='all',
                                                   normalization='per-line',
                                                   quantity='CaImaging', subquantity='Fluorescence'),
                            'CaImaging':dict(fig_fraction=3, subsampling=1, 
                                             # quantity='CaImaging', subquantity='dF/F', color=ge.green,
                                             quantity='CaImaging', subquantity='Fluorescence', color=ge.green,
                                             roiIndices=np.sort(np.random.choice(np.arange(np.sum(data.iscell)), args.Nmax, replace=False))),
                            'Locomotion':dict(fig_fraction=1, subsampling=1, color=ge.blue),
                            # 'Pupil':dict(fig_fraction=2, subsampling=1, color=ge.red),
                            # 'GazeMovement':dict(fig_fraction=1, subsampling=1, color=ge.orange),
                            'Photodiode':dict(fig_fraction=.5, subsampling=1, color='grey'),
                            'VisualStim':dict(fig_fraction=.5, color='black')},
                            Tbar=5)
        
    elif args.ops=='trial-average':
        fig, AX = data.plot_trial_average(roiIndex=args.roiIndex,
                                          protocol_id=0,
                                          quantity='CaImaging', subquantity='dF/F', column_key='angle', with_screen_inset=True,
                                          xbar=1, xbarlabel='1s', ybar=1, ybarlabel='1dF/F',
                                          with_stat_test=True,
                                          with_annotation=True,
                                          fig_preset='raw-traces-preset', color=ge.blue, label='test\n')
        
    elif args.ops=='visual-stim':
        fig, AX = data.show_VisualStim(args.tlim, Npanels=args.Npanels)
        fig2 = data.visual_stim.plot_stim_picture(args.episode, enhance=True)
        print('interval [%.1f, %.1f] ' % (data.nwbfile.stimulus['time_start_realigned'].data[args.episode],
                                          data.nwbfile.stimulus['time_stop_realigned'].data[args.episode]))
        
    elif args.ops=='FOV':
        fig, ax = data.show_CaImaging_FOV('meanImg', NL=3, cmap=ge.get_linear_colormap('k', 'lightgreen'))
    else:
        print(' option not recognized !')
        
    ge.show()




