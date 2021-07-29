# general modules
import pynwb, os, sys, pathlib
import numpy as np
import matplotlib.pylab as plt

# custom modules
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from dataviz import tools
from analysis.read_NWB import Data
from analysis import stat_tools
from Ca_imaging.tools import compute_CaImaging_trace
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
    
    def add_Photodiode(self, tlim, ax,
                       fig_fraction_start=0., fig_fraction=1., subsampling=10, color=ge.grey, name='photodiode'):
        i1, i2 = tools.convert_times_to_indices(*tlim, self.nwbfile.acquisition['Photodiode-Signal'])
        x = tools.convert_index_to_time(range(i1,i2), self.nwbfile.acquisition['Photodiode-Signal'])[::subsampling]
        y = self.nwbfile.acquisition['Photodiode-Signal'].data[i1:i2][::subsampling]
        ax.plot(x, (y-y.min())/(y.max()-y.min())*fig_fraction+fig_fraction_start, color=color)
        ge.annotate(ax, ' '+name, (tlim[1], fig_fraction/2.+fig_fraction_start), xycoords='data', color=color, va='center')


    def add_Electrophy(self, tlim, ax,
                       fig_fraction_start=0., fig_fraction=1., subsampling=2, color='k',
                       name='LFP'):
        i1, i2 = tools.convert_times_to_indices(*tlim, self.nwbfile.acquisition['Electrophysiological-Signal'])
        x = tools.convert_index_to_time(range(i1,i2), self.nwbfile.acquisition['Electrophysiological-Signal'])[::subsampling]
        y = self.nwbfile.acquisition['Electrophysiological-Signal'].data[i1:i2][::subsampling]
        ax.plot(x, (y-y.min())/(y.max()-y.min())*fig_fraction+fig_fraction_start, color=color)
        ge.annotate(ax, ' '+name, (tlim[1], fig_fraction/2.+fig_fraction_start), xycoords='data', color=color, va='center')
        
    def add_Locomotion(self, tlim, ax,
                       fig_fraction_start=0., fig_fraction=1., subsampling=2,
                       Sscale=1, # cm/s
                       color=ge.blue, name='run. speed'):
        i1, i2 = tools.convert_times_to_indices(*tlim, self.nwbfile.acquisition['Running-Speed'])
        x = tools.convert_index_to_time(range(i1,i2), self.nwbfile.acquisition['Running-Speed'])[::subsampling]
        y = self.nwbfile.acquisition['Running-Speed'].data[i1:i2][::subsampling]
        scale_range = (y.max()-y.min())
        if scale_range>0:
            ax.plot(self.shifted_start(tlim)*np.ones(2), fig_fraction_start+np.arange(2)*fig_fraction*0.3/scale_range, color=color, lw=1)
            ge.annotate(ax, '%.0fcm/s' % Sscale, (self.shifted_start(tlim), fig_fraction_start), ha='right', color=color, va='center', xycoords='data')
            ax.plot(x, (y-y.min())/(y.max()-y.min())*fig_fraction+fig_fraction_start, color=color)
        else:
            ax.plot(x, 0*x+fig_fraction_start, color=color)
        ge.annotate(ax, ' '+name, (tlim[1], fig_fraction/2.+fig_fraction_start), xycoords='data', color=color, va='center')
        
    def add_FaceMotion(self, tlim, ax,
                       fig_fraction_start=0., fig_fraction=1., subsampling=2, color=ge.purple, name='facemotion'):
        i1, i2 = tools.convert_times_to_indices(*tlim, self.nwbfile.processing['FaceMotion'].data_interfaces['face-motion'])
        t = self.nwbfile.processing['FaceMotion'].data_interfaces['face-motion'].timestamps[i1:i2]
        motion = self.nwbfile.processing['FaceMotion'].data_interfaces['face-motion'].data[i1:i2]
        x, y = t[::subsampling], motion[::subsampling]
        scale_range = (y.max()-y.min())
        ax.plot(x, (y-y.min())/scale_range*fig_fraction+fig_fraction_start, color=color)
        ax.plot(self.shifted_start(tlim)*np.ones(2), fig_fraction_start+np.arange(2)*fig_fraction*0.3/scale_range, color=color, lw=1)
        ge.annotate(ax, 'a.u. ', (self.shifted_start(tlim), fig_fraction_start), ha='right', color=color, xycoords='data')
        ge.annotate(ax, ' '+name, (tlim[1], fig_fraction/2.+fig_fraction_start), color=color, xycoords='data', va='center')

    def add_Pupil(self, tlim, ax,
                  fig_fraction_start=0., fig_fraction=1., subsampling=2,
                  Pbar = 0.5, # scale bar in mm
                  color='red', name='pupil diam.'):
        i1, i2 = tools.convert_times_to_indices(*tlim, self.nwbfile.processing['Pupil'].data_interfaces['cx'])
        t = self.nwbfile.processing['Pupil'].data_interfaces['sx'].timestamps[i1:i2]
        diameter = 2*np.max([self.nwbfile.processing['Pupil'].data_interfaces['sx'].data[i1:i2],
                           self.nwbfile.processing['Pupil'].data_interfaces['sy'].data[i1:i2]], axis=0)
        x, y = t[::subsampling], diameter[::subsampling]
        scale_range = (y.max()-y.min())
        ax.plot(x, (y-y.min())/scale_range*fig_fraction+fig_fraction_start, color=color)
        ax.plot(self.shifted_start(tlim)*np.ones(2), fig_fraction_start+np.arange(2)*fig_fraction*Pbar/scale_range, color=color, lw=1)
        ge.annotate(ax, '%.1fmm ' % Pbar, (self.shifted_start(tlim), fig_fraction_start), ha='right', va='center', xycoords='data', color=color) # twice for diam.
        ge.annotate(ax, ' '+name, (tlim[1], fig_fraction/2.+fig_fraction_start), color=color, va='center', xycoords='data')
        
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
        i1 = tools.convert_time_to_index(tlim[0], self.Neuropil, axis=1)
        i2 = tools.convert_time_to_index(tlim[1], self.Neuropil, axis=1)
        tt = np.array(self.Neuropil.timestamps[:])[np.arange(i1,i2)][::subsampling]
        if vicinity_factor>1:
            ymax_factor = fig_fraction*(1-1./vicinity_factor)
        else:
            ymax_factor = fig_fraction/len(roiIndices)
        for n, ir in zip(range(len(roiIndices))[::-1], roiIndices[::-1]):
            y = dF[n, np.arange(i1,i2)][::subsampling]
            ypos = n*fig_fraction/len(roiIndices)/vicinity_factor+fig_fraction_start
            if subquantity in ['dF/F', 'dFoF']:
                ax.plot(tt, y/2.*ymax_factor+ypos, color=COLORS[n], lw=1)
                ax.plot(self.shifted_start(tlim)*np.ones(2), np.arange(2)/2.*ymax_factor+ypos, color=COLORS[n], lw=1)
            elif y.max()>y.min():
                rescaled_y = (y-y.min())/(y.max()-y.min())
                ax.plot(tt, rescaled_y*ymax_factor+ypos, color=COLORS[n], lw=1)

            ax.annotate('ROI#%i'%(ir+1), (tlim[1], ypos), color=COLORS[n], fontsize=8)
        if subquantity in ['dF/F', 'dFoF']:
            ax.annotate('1$\Delta$F/F', (self.shifted_start(tlim), fig_fraction_start), ha='right',
                        rotation=90, color=color, fontsize=9)
        ge.annotate(ax, name+'\n ', (self.shifted_start(tlim), fig_fraction/2.+fig_fraction_start), color=color,
                    xycoords='data', ha='right', va='center', rotation=90)
            

    def add_CaImagingSum(self, tlim, ax,
                         fig_fraction_start=0., fig_fraction=1., color='green',
                         quantity='CaImaging', subquantity='Fluorescence', subsampling=1,
                         name='Sum [Ca]'):
        i1 = tools.convert_time_to_index(tlim[0], self.Neuropil, axis=1)
        i2 = tools.convert_time_to_index(tlim[1], self.Neuropil, axis=1)
        tt = np.array(self.Neuropil.timestamps[:])[np.arange(i1,i2)][::subsampling]
        y = compute_CaImaging_trace(self, subquantity, np.arange(np.sum(self.iscell))).sum(axis=0)[np.arange(i1,i2)][::subsampling]
        ax.plot(tt, (y-y.min())/(y.max()-y.min())*fig_fraction+fig_fraction_start, color=color)
        ax.annotate(name, (tlim[0], fig_fraction/2.+fig_fraction_start), color=color,
                    fontsize=8, ha='right')
            
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
            ax.fill_between([tstart, tstop], [0,0], np.zeros(2)+ylevel, lw=0, alpha=0.05, color=color)
            axi = ax.inset_axes([tstart, 1.01, (tstop-tstart), size], transform=ax.transData)
            axi.axis('equal')
            # add arrow if drifting stim
            if (self.nwbfile.stimulus['frame_run_type'].data[i]=='drifting'):
                arrow={'direction':self.nwbfile.stimulus['angle'].data[i],
                       'center':(0,0), 'length':25, 'width_factor':0.1, 'color':'red'}
            else:
                arrow = None
            # add VSE if in stim (TO BE DONE)
            if True:
                vse = None
            self.visual_stim.show_frame(i, ax=axi, label=None, arrow=arrow, enhance=True)
        ax.annotate(name, (self.shifted_start(tlim), fig_fraction/2.+fig_fraction_start), color=color,
                    fontsize=8, ha='right')
    
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
                      figsize=(15,6), Tbar=20,
                      ax=None):
        
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
            
        ax.axis('off')
        ax.plot([self.shifted_start(tlim), self.shifted_start(tlim)+Tbar], [1.,1.], lw=2, color='k')
        ax.set_xlim([self.shifted_start(tlim)-0.01*(tlim[1]-tlim[0]),tlim[1]+0.01*(tlim[1]-tlim[0])])
        ax.set_ylim([-0.05,1.05])
        ax.annotate(' %is' % Tbar, [self.shifted_start(tlim), 1.02], color='k', fontsize=9)
        
        return fig, ax
        
    ###------------------------------------------
    ### ----- Trial Average plot components -----
    ###------------------------------------------

    def plot_trial_average(self,
                           # episodes props
                           protocol_id=0,
                           quantity='Photodiode-Signal', subquantity='dF/F', roiIndex=0,
                           dt_sampling=1, # ms
                           interpolation='linear',
                           baseline_substraction=False,
                           prestim_duration=None,
                           condition=None,
                           COL_CONDS=None, column_keys=[], column_key='',
                           ROW_CONDS=None, row_keys=[], row_key='',
                           COLOR_CONDS = None, color_keys=[], color_key='',
                           fig_preset='',
                           xbar=0., xbarlabel='',
                           ybar=0., ybarlabel='',
                           with_std=True,
                           with_screen_inset=False,
                           with_stim=True,
                           with_axis=False,
                           with_stat_test=False, stat_test_props=dict(interval_pre=[-1,0], interval_post=[1,2], test='wilcoxon'),
                           color='k',
                           label='',
                           ylim=None, xlim=None,
                           fig=None, AX=None, verbose=False):
        
        # ----- protocol cond ------
        Pcond = self.get_protocol_cond(protocol_id)
        
        # ----- building episodes of cell response ------
        cellR = stat_tools.CellResponse(self,
                                        protocol_id=protocol_id,
                                        quantity=quantity,
                                        subquantity=subquantity,
                                        roiIndex=roiIndex)

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
        if COLOR_CONDS is None:
            COLOR_CONDS = [np.ones(np.sum(Pcond), dtype=bool)]

        # single-value
        # condition = [...]
            
        # if (len(COLOR_CONDS)>1) and (self.color.text()!=''):
        #     COLORS = [getattr(ge, self.color.text())((c%10)/10.) for c in np.arange(len(COLOR_CONDS))]
        # elif (len(COLOR_CONDS)>1):
        #     COLORS = [ge.tab10((c%10)/10.) for c in np.arange(len(COLOR_CONDS))]
        # elif self.color.text()!='':
        #     COLORS = [getattr(ge, self.color.text())]
        # else:
        COLORS = [color]
                
        if (fig is None) and (AX is None) and (fig_preset=='raw-traces-preset'):
            fig, AX = ge.figure(axes=(len(COL_CONDS), len(ROW_CONDS)), reshape_axes=False,
                                top=0.4, bottom=0.4, left=0.7, right=0.7,
                                wspace=0.5, hspace=0.5)
        elif (fig is None) and (AX is None):
            fig, AX = ge.figure(axes=(len(COL_CONDS), len(ROW_CONDS)), reshape_axes=False)

        self.ylim = [np.inf, -np.inf]
        for irow, row_cond in enumerate(ROW_CONDS):
            for icol, col_cond in enumerate(COL_CONDS):
                for icolor, color_cond in enumerate(COLOR_CONDS):
                    cond = np.array(condition & col_cond & row_cond & color_cond)[:cellR.EPISODES['resp'].shape[0]]

                    if cellR.EPISODES['resp'][cond,:].shape[0]>0:
                        my = cellR.EPISODES['resp'][cond,:].mean(axis=0)
                        if with_std:
                            sy = cellR.EPISODES['resp'][cond,:].std(axis=0)
                            ge.plot(cellR.EPISODES['t'], my, sy=sy,
                                    ax=AX[irow][icol], color=COLORS[icolor], lw=1)
                            self.ylim = [min([self.ylim[0], np.min(my-sy)]),
                                         max([self.ylim[1], np.max(my+sy)])]
                        else:
                            AX[irow][icol].plot(cellR.EPISODES['t'], my,
                                                color=COLORS[icolor], lw=1)
                            self.ylim = [min([self.ylim[0], np.min(my)]),
                                         max([self.ylim[1], np.max(my)])]

                            
                    if with_screen_inset:
                        inset = ge.inset(AX[irow][icol], [.8, .9, .3, .25])
                        self.visual_stim.show_frame(\
                                    cellR.EPISODES['index_from_start'][cond][0],
                                    ax=inset, enhance=True, label=None)
                        
          
        if with_stat_test:
            for irow, row_cond in enumerate(ROW_CONDS):
                for icol, col_cond in enumerate(COL_CONDS):
                    for icolor, color_cond in enumerate(COLOR_CONDS):
                        
                        cond = np.array(condition & col_cond & row_cond & color_cond)[:cellR.EPISODES['resp'].shape[0]]
                        test = stat_tools.stat_test_for_evoked_responses(cellR.EPISODES, cond, **stat_test_props)
                        AX[irow][icol].plot(stat_test_props['interval_pre'], self.ylim[0]*np.ones(2), 'k-', lw=1)
                        AX[irow][icol].plot(stat_test_props['interval_post'], self.ylim[0]*np.ones(2), 'k-', lw=1)
                        ps, size = stat_tools.pval_to_star(test)
                        AX[irow][icol].annotate(ps, ((stat_test_props['interval_post'][0]+stat_test_props['interval_pre'][1])/2.,
                                                     self.ylim[0]), va='top', ha='center', size=size, xycoords='data')
                            
        if xlim is None:
            self.xlim = [cellR.EPISODES['t'][0], cellR.EPISODES['t'][-1]]
        else:
            self.xlim = xlim
            
        if ylim is not None:
            self.ylim = ylim

            
        for irow, row_cond in enumerate(ROW_CONDS):
            for icol, col_cond in enumerate(COL_CONDS):
                ge.set_plot(AX[irow][icol],
                            spines=(['left', 'bottom'] if with_axis else []),
                            # xlabel=(self.xbarlabel.text() if with_axis else ''),
                            # ylabel=(self.ybarlabel.text() if with_axis else ''),
                            ylim=self.ylim, xlim=self.xlim)

                if with_stim:
                    AX[irow][icol].fill_between([0, np.mean(cellR.EPISODES['time_duration'])],
                                        self.ylim[0]*np.ones(2), self.ylim[1]*np.ones(2),
                                        color='grey', alpha=.2, lw=0)

        if not with_axis:
            ge.draw_bar_scales(AX[0][0],
                               Xbar=xbar, Xbar_label=xbarlabel,
                               Ybar=ybar,  Ybar_label=ybarlabel,
                               Xbar_fraction=0.1, Xbar_label_format='%.1f',
                               Ybar_fraction=0.2, Ybar_label_format='%.1f',
                               loc='top-left')

        if label!='':
            ge.annotate(fig, label, (0,0), color=color, ha='left', va='bottom')


        # if self.annot.isChecked():
        #     S=''
        #     if hasattr(self, 'roiPick'):
        #         S+='roi #%s' % self.roiPick.text()
        #     for i, key in enumerate(self.varied_parameters.keys()):
        #         if 'single-value' in getattr(self, '%s_plot' % key).currentText():
        #             S += ', %s=%.2f' % (key, getattr(self, '%s_values' % key).currentText())
        #     ge.annotate(fig, S, (0,0), color='k', ha='left', va='bottom')
            
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
    if key=='angle':
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
    elif key=='light-level':
        if value==0:
            return 'grey'
        elif value==1:
            return 'white'
        else:
            return 'lum.=%.1f' % value
    else:
        return '%.2f' % value

    
     
if __name__=='__main__':
    
    # filename = os.path.join(os.path.expanduser('~'), 'DATA', '2021_03_11-17-32-34.nwb')
    filename = sys.argv[-1]
    data = MultimodalData(filename)

    # TRIAL AVERAGING EXAMPLE
    fig, AX = data.plot_trial_average(roiIndex=3,
                                      protocol_id=0,
                                      quantity='CaImaging', subquantity='dF/F', column_key='angle', with_screen_inset=True,
                                      xbar=1, xbarlabel='1s', ybar=1, ybarlabel='1dF/F',
                                      with_stat_test=True,
                                      with_annot=True,
                                      fig_preset='raw-traces-preset', color=ge.blue, label='test\n')
    # data.plot_trial_average(roiIndex=3,
    #                         protocol_id=0,
    #                         quantity='CaImaging', subquantity='dF/F', column_key='angle', with_screen_inset=True,
    #                         xbar=1, xbarlabel='1s', ybar=1, ybarlabel='1dF/F',
    #                         ylim=AX[0][0].get_ylim(),
    #                         fig_preset='raw-traces-preset', color=ge.red, label='test 2\n\n', fig=fig, AX=AX)
    ge.show()
    
    # RAW DATA EXAMPLE
    # data.plot_raw_data([1040, 1060], 
    #           # settings={'Photodiode':dict(fig_fraction=.1, subsampling=1, color='grey'),
    #           settings={'Locomotion':dict(fig_fraction=1, subsampling=5, color=ge.blue),
    #                     'Pupil':dict(fig_fraction=1, subsampling=1, color=ge.red),
    #                     'CaImaging':dict(fig_fraction=7, subsampling=2, 
    #                                      # quantity='CaImaging', subquantity='dF/F', color=ge.green,
    #                                      quantity='CaImaging', subquantity='Fluorescence', color=ge.green,
    #                                      # roiIndices=np.arange(np.sum(data.iscell))),
    #                                      roiIndices=np.sort([5, 0, 18, 1, 3, 2, 11, 6, 8, 10, 9])),
    #                     'VisualStim':dict(fig_fraction=2, color='black')},                    
    #           Tbar=10)


    # from datavyz import ge
    # fig, ax = data.show_CaImaging_FOV('meanImg', NL=3, cmap=ge.get_linear_colormap('k', 'lightgreen'))
    # ge.save_on_desktop(fig, 'fig.png')
    # plt.show()







