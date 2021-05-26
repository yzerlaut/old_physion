import sys, time, tempfile, os, pathlib, json, datetime, string
from PyQt5 import QtGui, QtWidgets, QtCore
import numpy as np
import pyqtgraph as pg
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import day_folder
from dataviz.guiparts import NewWindow, smallfont
from Ca_imaging.tools import compute_CaImaging_trace
from scipy.interpolate import interp1d
from analysis.stat_tools import stat_test_for_evoked_responses, pval_to_star
from analysis.trial_averaging import build_episodes
from datavyz.stack_plots import add_plot_to_svg, export_drawing_as_png

from datavyz import graph_env_manuscript as ge
NMAX_PARAMS = 5

class FiguresWindow(NewWindow):

    def __init__(self, 
                 parent=None,
                 dt_sampling=10, # ms
                 fig_name=os.path.join(os.path.expanduser('~'), 'Desktop', 'fig.svg'),
                 title='Figures'):

        super(FiguresWindow, self).__init__(parent=parent.app,
                                            title=title,
                                            i=-10, size=(800,800))

        self.parent = parent
        self.get_varied_parameters()
        self.fig_name = fig_name
        
        self.EPISODES = None
        self.xlim, self.ylim = [-10, 10], [-10, 10]
        
        mainLayout, Layouts = QtWidgets.QVBoxLayout(self.cwidget), []

        # -- protocol
        Layouts.append(QtWidgets.QHBoxLayout())
        # Layouts[-1].addWidget(QtWidgets.QLabel('-- Protocol:   ', self))
        self.pbox = QtWidgets.QComboBox(self)
        self.pbox.setFixedWidth(400)
        self.pbox.addItem(' [PROTOCOL] ')
        self.pbox.addItems(self.parent.protocols)
        Layouts[-1].addWidget(self.pbox)
        
        # -- quantity and subquantity
        Layouts.append(QtWidgets.QHBoxLayout())
        Layouts[-1].addWidget(QtWidgets.QLabel('-- Quantity:   ', self))
        self.qbox = QtWidgets.QComboBox(self)
        self.qbox.addItem('')
        if 'ophys' in self.parent.nwbfile.processing:
            self.qbox.addItem('CaImaging')
        for key in parent.nwbfile.acquisition:
            if len(parent.nwbfile.acquisition[key].data.shape)==1:
                self.qbox.addItem(key) # only for scalar variables
        self.qbox.setFixedWidth(400)
        Layouts[-1].addWidget(self.qbox)
        Layouts[-1].addWidget(QtWidgets.QLabel('       -- Sub-quantity:   ', self))
        self.sqbox = QtWidgets.QLineEdit(self)
        self.sqbox.setText(35*' '+'e.g. "dF/F", "Neuropil", "pLFP", ...')
        self.sqbox.setMinimumWidth(400)
        Layouts[-1].addWidget(self.sqbox)
        
        # -- roi
        if 'ophys' in self.parent.nwbfile.processing:
            Layouts.append(QtWidgets.QHBoxLayout())
            Layouts[-1].addWidget(QtWidgets.QLabel('-- ROI:   ', self))
            self.roiPick = QtGui.QLineEdit()
            self.roiPick.setText('    [select ROI]  e.g.: "1",  "10-20", "3, 45, 7", ... ')
            self.roiPick.setMinimumWidth(400)
            self.roiPick.returnPressed.connect(self.select_ROI)
            Layouts[-1].addWidget(self.roiPick)
            Layouts[-1].addWidget(QtWidgets.QLabel(150*'-', self))

        Layouts.append(QtWidgets.QHBoxLayout())
        Layouts[-1].addWidget(QtWidgets.QLabel(50*'<->', self)) # SEPARATOR

        # varied keys
        Layouts.append(QtWidgets.QHBoxLayout())
        Layouts[-1].addWidget(QtWidgets.QLabel('  -* STIMULI PARAMETERS *-', self))
        for key in self.varied_parameters:
            Layouts.append(QtWidgets.QHBoxLayout())
            Layouts[-1].addWidget(QtWidgets.QLabel('--- %s ' % key, self))
            setattr(self, '%s_plot' % key, QtWidgets.QComboBox(self))
            getattr(self, '%s_plot' % key).addItems(['merged', 'single-value', 'column-panels', 'row-panels', 'color-coded', 'x-axis', 'N/A'])
            Layouts[-1].addWidget(getattr(self, '%s_plot' % key))
            setattr(self, '%s_values' % key, QtWidgets.QComboBox(self))
            getattr(self, '%s_values' % key).addItems(['full', 'custom']+[str(s) for s in self.varied_parameters[key]])
            Layouts[-1].addWidget(getattr(self, '%s_values' % key))
            Layouts[-1].addWidget(QtWidgets.QLabel(10*' ', self))
            Layouts[-1].addWidget(QtWidgets.QLabel(' custom values : ', self))
            setattr(self, '%s_customvalues' % key, QtWidgets.QLineEdit(self))
            getattr(self, '%s_customvalues' % key).setMaximumWidth(150)
            Layouts[-1].addWidget(getattr(self, '%s_customvalues' % key))
            Layouts[-1].addWidget(QtWidgets.QLabel(30*' ', self))

        Layouts.append(QtWidgets.QHBoxLayout())
        Layouts[-1].addWidget(QtWidgets.QLabel(50*'<->', self)) # SEPARATOR
        
        # figure props type
        Layouts.append(QtWidgets.QHBoxLayout())
        Layouts[-1].addWidget(QtWidgets.QLabel('       -* RESPONSES *-        ', self))
        self.responseType = QtWidgets.QComboBox(self)
        self.responseType.addItems(['stim-evoked-traces', 'mean-stim-evoked', 'integral-stim-evoked'])
        Layouts[-1].addWidget(self.responseType)
        Layouts[-1].addWidget(QtWidgets.QLabel('       color:  ', self))
        self.color = QtGui.QLineEdit()
        Layouts[-1].addWidget(self.color)
        Layouts[-1].addWidget(QtWidgets.QLabel('       label:  ', self))
        self.label = QtGui.QLineEdit()
        Layouts[-1].addWidget(self.label)
        Layouts[-1].addWidget(QtWidgets.QLabel('    n-label:  ', self))
        self.nlabel = QtGui.QLineEdit()
        self.nlabel.setText('1')
        Layouts[-1].addWidget(self.nlabel)
        Layouts.append(QtWidgets.QHBoxLayout())
        Layouts[-1].addWidget(QtWidgets.QLabel(' pre-stimulus window (s):', self))
        self.preWindow = QtGui.QLineEdit()
        self.preWindow.setText('[-1,0]')
        Layouts[-1].addWidget(self.preWindow)
        Layouts[-1].addWidget(QtWidgets.QLabel(' post-stimulus window (s):', self))
        self.postWindow = QtGui.QLineEdit()
        self.postWindow.setText('[1,4]')
        Layouts[-1].addWidget(self.postWindow)
        Layouts.append(QtWidgets.QHBoxLayout())
        self.baseline = QtGui.QCheckBox("baseline substraction")
        Layouts[-1].addWidget(self.baseline)
        Layouts[-1].addWidget(QtWidgets.QLabel(10*'  ', self))
        self.withStatTest = QtGui.QCheckBox("with stat. test")
        Layouts[-1].addWidget(self.withStatTest)
        
        Layouts.append(QtWidgets.QHBoxLayout())
        Layouts[-1].addWidget(QtWidgets.QLabel(50*'<->', self)) # SEPARATOR
        
        Layouts.append(QtWidgets.QHBoxLayout())
        Layouts[-1].addWidget(QtWidgets.QLabel('       -* Figure Properties *-        ', self))
        self.fig_presets = QtWidgets.QComboBox(self)
        self.fig_presets.setFixedWidth(400)
        self.fig_presets.addItems(['', 'raw-traces-preset', 'IO-curves-preset'])
        Layouts[-1].addWidget(self.fig_presets)
        
        Layouts[-1].addWidget(QtWidgets.QLabel('Panel size:  ', self))
        self.panelsize = QtGui.QLineEdit()
        self.panelsize.setText('(1,1)')
        Layouts[-1].addWidget(self.panelsize)

        #
        self.samplingBox = QtWidgets.QDoubleSpinBox(self)
        self.samplingBox.setValue(dt_sampling)
        self.samplingBox.setMaximum(500)
        self.samplingBox.setMinimum(0.1)
        self.samplingBox.setSuffix(' (ms) sampling')
        Layouts[-1].addWidget(self.samplingBox)
        
        # plot type
        Layouts.append(QtWidgets.QHBoxLayout())
        Layouts[-1].addWidget(QtWidgets.QLabel('Plot type: ', self))
        self.plotbox = QtWidgets.QComboBox(self)
        self.plotbox.addItems(['2d-plot', 'polar-plot', 'bar-plot'])
        Layouts[-1].addWidget(self.plotbox)
        Layouts[-1].addWidget(QtWidgets.QLabel(10*'  ', self))
        self.plotbox2 = QtWidgets.QComboBox(self)
        self.plotbox2.addItems(['line', 'dot'])
        Layouts[-1].addWidget(self.plotbox2)
        Layouts[-1].addWidget(QtWidgets.QLabel(10*'  ', self))
        self.withSTDbox = QtGui.QCheckBox("with s.d.")
        Layouts[-1].addWidget(self.withSTDbox)
        Layouts[-1].addWidget(QtWidgets.QLabel(10*'  ', self))
        self.stim = QtGui.QCheckBox("with stim. ")
        Layouts[-1].addWidget(self.stim)
        Layouts[-1].addWidget(QtWidgets.QLabel(10*'  ', self))
        self.screen = QtGui.QCheckBox("with screen inset ")
        Layouts[-1].addWidget(self.screen)
        Layouts[-1].addWidget(QtWidgets.QLabel(10*'  ', self))
        self.annot = QtGui.QCheckBox("with annot. ")
        Layouts[-1].addWidget(self.annot)
        Layouts[-1].addWidget(QtWidgets.QLabel(10*'  ', self))
        self.axis = QtGui.QCheckBox("with axis")
        Layouts[-1].addWidget(self.axis)
        Layouts[-1].addWidget(QtWidgets.QLabel(10*'  ', self))
        self.grid = QtGui.QCheckBox("with grid")
        Layouts[-1].addWidget(self.grid)

        # X-scales
        Layouts.append(QtWidgets.QHBoxLayout())
        Layouts[-1].addWidget(QtWidgets.QLabel('X-SCALE      ---     ', self))
        Layouts[-1].addWidget(QtWidgets.QLabel('x-min:', self))
        self.xmin = QtGui.QLineEdit()
        self.xmin.setText('')
        Layouts[-1].addWidget(self.xmin)
        Layouts[-1].addWidget(QtWidgets.QLabel('x-max:', self))
        self.xmax = QtGui.QLineEdit()
        self.xmax.setText('')
        Layouts[-1].addWidget(self.xmax)
        Layouts[-1].addWidget(QtWidgets.QLabel('x-bar:', self))
        self.xbar = QtGui.QLineEdit()
        self.xbar.setText('')
        Layouts[-1].addWidget(self.xbar)
        Layouts[-1].addWidget(QtWidgets.QLabel('x-barlabel:', self))
        self.xbarlabel = QtGui.QLineEdit()
        self.xbarlabel.setText('')
        Layouts[-1].addWidget(self.xbarlabel)

        # Y-scales
        Layouts.append(QtWidgets.QHBoxLayout())
        Layouts[-1].addWidget(QtWidgets.QLabel('Y-SCALE      ---     ', self))
        Layouts[-1].addWidget(QtWidgets.QLabel('y-min:', self))
        self.ymin = QtGui.QLineEdit()
        self.ymin.setText('')
        Layouts[-1].addWidget(self.ymin)
        Layouts[-1].addWidget(QtWidgets.QLabel('y-max:', self))
        self.ymax = QtGui.QLineEdit()
        self.ymax.setText('')
        Layouts[-1].addWidget(self.ymax)
        Layouts[-1].addWidget(QtWidgets.QLabel('y-bar:', self))
        self.ybar = QtGui.QLineEdit()
        self.ybar.setText('')
        Layouts[-1].addWidget(self.ybar)
        Layouts[-1].addWidget(QtWidgets.QLabel('y-barlabel:', self))
        self.ybarlabel = QtGui.QLineEdit()
        self.ybarlabel.setText('')
        Layouts[-1].addWidget(self.ybarlabel)

        # curve
        Layouts.append(QtWidgets.QHBoxLayout())
        # self.Layout12.addWidget(self.baselineCB)
        
        Layouts.append(QtWidgets.QHBoxLayout())
        Layouts[-1].addWidget(QtWidgets.QLabel(50*'<->', self)) # SEPARATOR
        
        # BUTTONS
        Layouts.append(QtWidgets.QHBoxLayout())
        self.compute = QtWidgets.QPushButton('compute episodes', self)
        self.compute.setFixedWidth(200)
        self.compute.clicked.connect(self.compute_episodes)
        Layouts[-1].addWidget(self.compute)
        
        Layouts.append(QtWidgets.QHBoxLayout())
        self.plotBtn = QtWidgets.QPushButton(' -* PLOT *- ', self)
        self.plotBtn.setFixedWidth(200)
        self.plotBtn.clicked.connect(self.plot)
        Layouts[-1].addWidget(self.plotBtn)
        
        Layouts.append(QtWidgets.QHBoxLayout())
        self.setBtn = QtWidgets.QPushButton('extract settings', self)
        self.setBtn.setFixedWidth(200)
        self.setBtn.clicked.connect(self.set_settings)
        Layouts[-1].addWidget(self.setBtn)
        self.rstBtn = QtWidgets.QPushButton('reset settings', self)
        self.rstBtn.setFixedWidth(200)
        self.rstBtn.clicked.connect(self.reset_settings)
        Layouts[-1].addWidget(self.rstBtn)

        Layouts.append(QtWidgets.QHBoxLayout())
        self.newFig = QtWidgets.QPushButton('save as new figure', self)
        self.newFig.setFixedWidth(200)
        self.newFig.clicked.connect(self.new_fig)
        Layouts[-1].addWidget(self.newFig)

        self.append2Fig = QtWidgets.QPushButton('append to figure', self)
        self.append2Fig.setFixedWidth(200)
        self.append2Fig.clicked.connect(self.append)
        Layouts[-1].addWidget(self.append2Fig)

        self.exportBtn = QtWidgets.QPushButton('export to png', self)
        self.exportBtn.setFixedWidth(200)
        self.exportBtn.clicked.connect(self.export)
        Layouts[-1].addWidget(self.exportBtn)

        self.locBtn = QtWidgets.QComboBox(self)
        self.locBtn.addItems(['Desktop', 'summary'])
        self.locBtn.setFixedWidth(200)
        Layouts[-1].addWidget(self.locBtn)

        self.nameBtn = QtWidgets.QLineEdit(self)
        self.nameBtn.setText('fig')
        self.nameBtn.setFixedWidth(200)
        Layouts[-1].addWidget(self.nameBtn)

        self.dpi = QtWidgets.QSpinBox(self)
        self.dpi.setValue(100)
        self.dpi.setRange(10, 500)
        self.dpi.setSuffix(' (dpi)')
        self.dpi.setFixedWidth(80)
        Layouts[-1].addWidget(self.dpi)
        
        for l in Layouts:
            mainLayout.addLayout(l)

    def set_fig_name(self):
        if self.locBtn.currentText()=='Desktop':
            self.fig_name = os.path.join(os.path.expanduser('~'), 'Desktop', self.nameBtn.text()+'.svg')
        elif self.locBtn.currentText()=='summary':
            summary_dir = os.path.join(os.path.dirname(self.parent.datafile), 'summary', os.path.basename(self.parent.datafile).replace('.nwb', ''))
            pathlib.Path(summary_dir).mkdir(parents=True, exist_ok=True)
            self.fig_name = os.path.join(summary_dir, self.nameBtn.text()+'.svg')

    def export(self):
        export_drawing_as_png(self.fig_name, dpi=self.dpi.value(), background='white')

    def append(self):
        add_plot_to_svg(self.fig, self.fig_name)
        
    def new_fig(self):
        self.set_fig_name()
        self.fig.savefig(self.fig_name, transparent=True)
    
    def get_varied_parameters(self):
        self.varied_parameters = {}
        for key in self.parent.nwbfile.stimulus.keys():
            if key not in ['frame_run_type', 'index', 'protocol_id', 'time_duration', 'time_start',
                           'time_start_realigned', 'time_stop', 'time_stop_realigned', 'interstim', 'interstim-screen']:
                unique = np.unique(self.parent.nwbfile.stimulus[key].data[:])
                if len(unique)>1:
                    self.varied_parameters[key] = unique
        self.show()

    def set_settings(self):
        self.ymin.setText(str(self.ylim[0]))
        self.ymax.setText(str(self.ylim[1]))
        self.xmin.setText(str(self.xlim[0]))
        self.xmax.setText(str(self.xlim[1]))
        dx, dy = 0.1*(self.xlim[1]-self.xlim[0]), 0.2*(self.xlim[1]-self.xlim[0])
        self.xbar.setText('%.1f' % dx)
        self.xbarlabel.setText('%.1f' % dx)
        self.ybar.setText('%.1f' % dy)
        self.ybarlabel.setText('%.1f' % dy)

    def reset_settings(self):
        for x in [self.xmin, self.xmax, self.ymin, self.ymax,
                  self.xbar, self.ybar, self.xbarlabel, self.ybarlabel]:
            x.setText('')

            
    def plot_row_column_of_quantity(self):

        single_cond = self.build_single_conditions()
        COL_CONDS = self.build_column_conditions()
        ROW_CONDS = self.build_row_conditions()
        COLOR_CONDS = self.build_color_conditions()

        if (len(COLOR_CONDS)>1) and (self.color.text()!=''):
            COLORS = [getattr(ge, self.color.text())((c%10)/10.) for c in np.arange(len(COLOR_CONDS))]
        elif (len(COLOR_CONDS)>1):
            COLORS = [ge.tab10((c%10)/10.) for c in np.arange(len(COLOR_CONDS))]
        elif self.color.text()!='':
            COLORS = [getattr(ge, self.color.text())]
        else:
            COLORS = ['k']
                
        if self.fig_presets.currentText()=='raw-traces-preset':
            fig, AX = ge.figure(axes=(len(COL_CONDS), len(ROW_CONDS)), reshape_axes=False,
                                top=0.4, bottom=0.4, left=0.7, right=0.7,
                                wspace=0.5, hspace=0.5)
        else:
            fig, AX = ge.figure(axes=(len(COL_CONDS), len(ROW_CONDS)),
                                reshape_axes=False)

        self.ylim = [np.inf, -np.inf]
        for irow, row_cond in enumerate(ROW_CONDS):
            for icol, col_cond in enumerate(COL_CONDS):
                for icolor, color_cond in enumerate(COLOR_CONDS):
                    cond = np.array(single_cond & col_cond & row_cond & color_cond)[:self.EPISODES['resp'].shape[0]]
                    if self.EPISODES['resp'][cond,:].shape[0]>0:
                        my = self.EPISODES['resp'][cond,:].mean(axis=0)
                        if self.withSTDbox.isChecked():
                            sy = self.EPISODES['resp'][cond,:].std(axis=0)
                            ge.plot(self.EPISODES['t'], my, sy=sy,
                                    ax=AX[irow][icol], color=COLORS[icolor], lw=1)
                            self.ylim = [min([self.ylim[0], np.min(my-sy)]),
                                         max([self.ylim[1], np.max(my+sy)])]
                        else:
                            AX[irow][icol].plot(self.EPISODES['t'], my,
                                                color=COLORS[icolor], lw=1)
                            self.ylim = [min([self.ylim[0], np.min(my)]),
                                         max([self.ylim[1], np.max(my)])]

                    if self.screen.isChecked():
                        inset = ge.inset(AX[irow][icol], [.8, .9, .3, .25])
                        self.parent.visual_stim.show_frame(\
                                    self.EPISODES['index_from_start'][cond][0],
                                    ax=inset, parent=self.parent, enhance=True, label=None)
          
        if self.withStatTest.isChecked():
            for irow, row_cond in enumerate(ROW_CONDS):
                for icol, col_cond in enumerate(COL_CONDS):
                    for icolor, color_cond in enumerate(COLOR_CONDS):
                        
                        cond = np.array(single_cond & col_cond & row_cond & color_cond)[:self.EPISODES['resp'].shape[0]]
                        test = stat_test_for_evoked_responses(self.EPISODES, cond,
                                                              interval_pre=[self.t0pre, self.t1pre],
                                                              interval_post=[self.t0post, self.t1post],
                                                              test='wilcoxon')
                        
                        AX[irow][icol].plot([self.t0pre, self.t1pre], self.ylim[0]*np.ones(2), 'k-', lw=2)
                        AX[irow][icol].plot([self.t0post, self.t1post], self.ylim[0]*np.ones(2), 'k-', lw=2)
                        ps, size = pval_to_star(test)
                        AX[irow][icol].annotate(ps, ((self.t1pre+self.t0post)/2., self.ylim[0]), va='top', ha='center', size=size, xycoords='data')
                            
                            
        if (self.ymin.text()!='') and (self.ymax.text()!=''):
            self.ylim = [float(self.ymin.text()), float(self.ymax.text())]
        if (self.xmin.text()!='') and (self.xmax.text()!=''):
            self.xlim = [float(self.xmin.text()), float(self.xmax.text())]
        else:
            self.xlim = [self.EPISODES['t'][0], self.EPISODES['t'][-1]]
                            
        for irow, row_cond in enumerate(ROW_CONDS):
            for icol, col_cond in enumerate(COL_CONDS):
                ge.set_plot(AX[irow][icol],
                            spines=(['left', 'bottom'] if self.axis.isChecked() else []),
                            ylim=self.ylim, xlim=self.xlim,
                            xlabel=(self.xbarlabel.text() if self.axis.isChecked() else ''),
                            ylabel=(self.ybarlabel.text() if self.axis.isChecked() else ''))

                if self.stim.isChecked():
                    AX[irow][icol].fill_between([0, np.mean(self.EPISODES['time_duration'])],
                                        self.ylim[0]*np.ones(2), self.ylim[1]*np.ones(2),
                                        color='grey', alpha=.2, lw=0)

        if not self.axis.isChecked():
            ge.draw_bar_scales(AX[0][0],
                               Xbar=(0. if self.xbar.text()=='' else float(self.xbar.text())),
                               Xbar_label=self.xbarlabel.text(),
                               Ybar=(0. if self.ybar.text()=='' else float(self.ybar.text())),
                               Ybar_label=self.ybarlabel.text(),
                               Xbar_fraction=0.1, Xbar_label_format='%.1f',
                               Ybar_fraction=0.2, Ybar_label_format='%.1f',
                               loc='top-left')

        if self.label.text()!='':
            ge.annotate(fig, ' '+self.label.text()+\
                        (1+int(self.nlabel.text()))*'\n', (0,0), color=COLORS[0],
                        ha='left', va='bottom')


        if self.annot.isChecked():
            S=''
            if hasattr(self, 'roiPick'):
                S+='roi #%s' % self.roiPick.text()
            for i, key in enumerate(self.varied_parameters.keys()):
                if 'single-value' in getattr(self, '%s_plot' % key).currentText():
                    S += ', %s=%.2f' % (key, getattr(self, '%s_values' % key).currentText())
            ge.annotate(fig, S, (0,0), color='k', ha='left', va='bottom')
            
        return fig, AX
            
    def plot(self):

        if self.responseType.currentText()=='stim-evoked-traces':
            self.fig, AX = self.plot_row_column_of_quantity()
        else:
            pass
        
        ge.show()


    def select_ROI(self):
        """ see dataviz/gui.py """
        roiIndices = self.parent.select_ROI_from_pick(cls=self)
        if len(roiIndices)>0:
            self.parent.roiIndices = roiIndices
            self.parent.roiPick.setText(self.roiPick.text())
        self.statusBar.showMessage('ROIs set to %s ' % self.parent.roiIndices)
        
    def compute_episodes(self):
        
        if self.sqbox.text() in ['dF/F', 'Fluorescence', 'Neuropil', 'Deconvolved']:
            self.parent.CaImaging_key = self.sqbox.text()

        self.t0pre = float(self.preWindow.text().replace('[', '').replace(']', '').split(',')[0])
        self.t1pre = float(self.preWindow.text().replace('[', '').replace(']', '').split(',')[1])
        self.t0post = float(self.postWindow.text().replace('[', '').replace(']', '').split(',')[0])
        self.t1post = float(self.postWindow.text().replace('[', '').replace(']', '').split(',')[1])
            
        if (self.qbox.currentIndex()>0) and (self.pbox.currentIndex()>0):
            self.EPISODES = build_episodes(self,
                                           parent=self.parent,
                                           protocol_id=self.pbox.currentIndex()-1,
                                           quantity=self.qbox.currentText(),
                                           dt_sampling=self.samplingBox.value(), # ms
                                           interpolation='linear',
                                           baseline_substraction=self.baseline.isChecked(),
                                           prestim_duration=(self.t1pre-self.t0pre),
                                           verbose=True)
        else:
            print(' /!\ Pick a protocol and a quantity')
            
        
    def build_conditions(self, X, K):
        if len(K)>0:
            CONDS = []
            XK = np.meshgrid(*X)
            for i in range(len(XK[0].flatten())): # looping over joint conditions
                cond = np.ones(np.sum(self.Pcond), dtype=bool)
                for k, xk in zip(K, XK):
                    cond = cond & (self.parent.nwbfile.stimulus[k].data[self.Pcond]==xk.flatten()[i])
                CONDS.append(cond)
            return CONDS
        else:
            return [np.ones(np.sum(self.Pcond), dtype=bool)]
            
    def build_single_conditions(self):
        
        full_cond = np.ones(np.sum(self.Pcond), dtype=bool)
        
        for i, key in enumerate(self.varied_parameters.keys()):
            if 'single-value' in getattr(self, '%s_plot' % key).currentText():
                cond=(np.array(self.parent.nwbfile.stimulus[key].data[self.Pcond],
                               dtype=str)!=getattr(self, '%s_values' % key).currentText())
                full_cond[cond] = False
                
        return full_cond

    def build_column_conditions(self):
        X, K = [], []
        for i, key in enumerate(self.varied_parameters.keys()):
            if 'column' in getattr(self, '%s_plot' % key).currentText():
                X.append(np.sort(np.unique(self.parent.nwbfile.stimulus[key].data[self.Pcond])))
                K.append(key)
        return self.build_conditions(X, K)
    
    def build_row_conditions(self):
        X, K = [], []
        for i, key in enumerate(self.varied_parameters.keys()):
            if 'row' in getattr(self, '%s_plot' % key).currentText():
                X.append(np.sort(np.unique(self.parent.nwbfile.stimulus[key].data[self.Pcond])))
                K.append(key)
        return self.build_conditions(X, K)

    def build_color_conditions(self):
        X, K = [], []
        for i, key in enumerate(self.varied_parameters.keys()):
            if 'color' in getattr(self, '%s_plot' % key).currentText():
                X.append(np.sort(np.unique(self.parent.nwbfile.stimulus[key].data[self.Pcond])))
                K.append(key)
        return self.build_conditions(X, K)

        
if __name__=='__main__':

    import sys
    sys.path.append('/home/yann/work/physion')
    from physion.analysis.read_NWB import read as read_NWB
    from physion.visual_stim.psychopy_code.stimuli import build_stim
    
    class Parent:
        def __init__(self, filename=''):
            read_NWB(self, filename, verbose=False)
            self.datafile=filename
            self.app = QtWidgets.QApplication(sys.argv)
            self.description=''
            self.roiIndices = [0]
            self.CaImaging_key = 'Fluorescence'
            self.metadata['load_from_protocol_data'] = True
            self.metadata['no-window'] = True
            self.visual_stim = build_stim(self.metadata, no_psychopy=True)

    filename = sys.argv[-1]

    if '.nwb' in sys.argv[-1]:
        cls = Parent(filename)
        window = FiguresWindow(cls)
        sys.exit(cls.app.exec_())
    else:
        print('/!\ Need to provide a NWB datafile as argument ')
    
