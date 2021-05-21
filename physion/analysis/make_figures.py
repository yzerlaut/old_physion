import sys, time, tempfile, os, pathlib, json, datetime, string
from PyQt5 import QtGui, QtWidgets, QtCore
import numpy as np
import pyqtgraph as pg
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import day_folder
from dataviz.guiparts import NewWindow, smallfont
from Ca_imaging.tools import compute_CaImaging_trace
from scipy.interpolate import interp1d
from misc.colors import build_colors_from_array
from analysis.trial_averaging import build_episodes

from datavyz import graph_env_screen as ge
NMAX_PARAMS = 5

class FiguresWindow(NewWindow):

    def __init__(self, 
                 parent=None,
                 dt_sampling=10, # ms
                 title='Figures'):

        super(FiguresWindow, self).__init__(parent=parent.app,
                                            title=title)

        self.parent = parent
        self.get_varied_parameters()
        
        self.EPISODES, self.AX, self.l = None, None, None

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
            self.sqbox.setMinimumWidth(400)
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
            getattr(self, '%s_plot' % key).addItems(['merged', 'single-value', 'column-panels', 'raw-panels', 'color-coded', 'N/A'])
            Layouts[-1].addWidget(getattr(self, '%s_plot' % key))
            setattr(self, '%s_values' % key, QtWidgets.QComboBox(self))
            getattr(self, '%s_values' % key).addItems(['full', 'custom']+[str(s) for s in self.varied_parameters[key]])
            Layouts[-1].addWidget(getattr(self, '%s_values' % key))
            Layouts[-1].addWidget(QtWidgets.QLabel(' custom values : ', self))
            setattr(self, '%s_customvalues' % key, QtWidgets.QLineEdit(self))
            getattr(self, '%s_customvalues' % key).setMaximumWidth(300)
            Layouts[-1].addWidget(getattr(self, '%s_customvalues' % key))

        Layouts.append(QtWidgets.QHBoxLayout())
        Layouts[-1].addWidget(QtWidgets.QLabel(50*'<->', self)) # SEPARATOR
        
        # figure props type
        Layouts.append(QtWidgets.QHBoxLayout())
        Layouts[-1].addWidget(QtWidgets.QLabel('Figure properties   ----     ', self))
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
        self.baseline = QtGui.QCheckBox("baseline substraction")
        Layouts[-1].addWidget(self.baseline)
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
        self.plotBtn = QtWidgets.QPushButton(' -* PLOT *- ', self)
        self.plotBtn.setFixedWidth(200)
        self.plotBtn.clicked.connect(self.plot)
        Layouts[-1].addWidget(self.plotBtn)
        
        Layouts.append(QtWidgets.QHBoxLayout())
        self.compute = QtWidgets.QPushButton('compute episodes', self)
        self.compute.setFixedWidth(200)
        self.compute.clicked.connect(self.compute_episodes)
        Layouts[-1].addWidget(self.compute)
        
        Layouts.append(QtWidgets.QHBoxLayout())
        self.defaultsBtn = QtWidgets.QPushButton('find default settings', self)
        self.defaultsBtn.setFixedWidth(200)
        # self.defaultsBtn.clicked.connect(self.find_defaults)
        Layouts[-1].addWidget(self.defaultsBtn)

        Layouts.append(QtWidgets.QHBoxLayout())
        self.newFig = QtWidgets.QPushButton('generate new figure', self)
        self.newFig.setFixedWidth(200)
        # self.newFig.clicked.connect(self.new_fig)
        Layouts[-1].addWidget(self.newFig)

        Layouts.append(QtWidgets.QHBoxLayout())
        self.append2Fig = QtWidgets.QPushButton('append to figure', self)
        self.append2Fig.setFixedWidth(200)
        # self.append2Fig.clicked.connect(self.new_fig)
        Layouts[-1].addWidget(self.append2Fig)
        
        for l in Layouts:
            mainLayout.addLayout(l)

    def get_varied_parameters(self):
        self.varied_parameters = {}
        for key in self.parent.nwbfile.stimulus.keys():
            if key not in ['frame_run_type', 'index', 'protocol_id', 'time_duration', 'time_start',
                           'time_start_realigned', 'time_stop', 'time_stop_realigned', 'interstim', 'interstim-screen']:
                unique = np.unique(self.parent.nwbfile.stimulus[key].data[:])
                if len(unique)>1:
                    self.varied_parameters[key] = unique
        self.show()

    def plot(self):

        fig, ax = ge.figure()
        ge.show()

        
    def compute_episodes(self):
        if (self.qbox.currentIndex()>0) and (self.pbox.currentIndex()>0):
            self.EPISODES = build_episodes(self,
                                       parent=self.parent,
                                       protocol_id=self.pbox.currentIndex()-1,
                                       quantity=self.qbox.currentText(),
                                       dt_sampling=self.samplingBox.value(), # ms
                                       interpolation='linear',
                                       baseline_substraction=self.baseline.isChecked(),
                                       verbose=True)
        else:
            print(' /!\ Pick a protocol an a quantity')
            
    # def update_selection(self):
    #     for i in range(NMAX_PARAMS):
    #         getattr(self, "box%i"%i).clear()
    #     for i, key in enumerate(self.varied_parameters.keys()):
    #         for k in ['(merge)', '(color-code)', '(row)', '(column)']:
    #             getattr(self, "box%i"%i).addItem(key+((30-len(k)-len(key))*' ')+k)
        
    # def select_ROI(self):
    #     """ see dataviz/gui.py """
    #     roiIndices = self.parent.select_ROI_from_pick(cls=self)
    #     if len(roiIndices)>0:
    #         self.parent.roiIndices = roiIndices
    #         self.parent.roiPick.setText(self.roiPick.text())
    #     self.statusBar.showMessage('ROIs set to %s' % self.parent.roiIndices)


    # def keyword_update2(self):
    #     self.keyword_update(string=self.guiKeywords.text(), parent=self.parent)

    def plot_row_column_of_quantity(self):

        COL_CONDS = self.build_column_conditions()
        ROW_CONDS = self.build_row_conditions()
        COLOR_CONDS = self.build_color_conditions()

        if len(COLOR_CONDS)>1:
            COLORS = build_colors_from_array(np.arange(len(COLOR_CONDS)))
        else:
            COLORS = [(255,255,255,255)]

        self.l = self.plots.addLayout(rowspan=len(ROW_CONDS),
                                      colspan=len(COL_CONDS),
                                      border=(0,0,0))
        self.l.setContentsMargins(4, 4, 4, 4)
        self.l.layout.setSpacing(2.)            

        # re-adding stuff
        self.AX, ylim = [], [np.inf, -np.inf]
        for irow, row_cond in enumerate(ROW_CONDS):
            self.AX.append([])
            for icol, col_cond in enumerate(COL_CONDS):
                self.AX[irow].append(self.l.addPlot())
                for icolor, color_cond in enumerate(COLOR_CONDS):
                    cond = np.array(col_cond & row_cond & color_cond)[:self.EPISODES['resp'].shape[0]]
                    pen = pg.mkPen(color=COLORS[icolor], width=2)
                    if self.EPISODES['resp'][cond,:].shape[0]>0:
                        my = self.EPISODES['resp'][cond,:].mean(axis=0)
                        if np.sum(cond)>1:
                            spen = pg.mkPen(color=(0,0,0,0), width=0)
                            spenbrush = pg.mkBrush(color=(*COLORS[icolor][:3], 100))
                            sy = self.EPISODES['resp'][cond,:].std(axis=0)
                            phigh = pg.PlotCurveItem(self.EPISODES['t'], my+sy, pen = spen)
                            plow = pg.PlotCurveItem(self.EPISODES['t'], my-sy, pen = spen)
                            pfill = pg.FillBetweenItem(phigh, plow, brush=spenbrush)
                            self.AX[irow][icol].addItem(phigh)
                            self.AX[irow][icol].addItem(plow)
                            self.AX[irow][icol].addItem(pfill)
                            ylim[0] = np.min([np.min(my-sy), ylim[0]])
                            ylim[1] = np.max([np.max(my+sy), ylim[1]])
                        self.AX[irow][icol].plot(self.EPISODES['t'], my, pen = pen)
                    else:
                        print(' /!\ Problem with episode (%i, %i, %i)' % (irow, icol, icolor))
                if icol>0:
                    self.AX[irow][icol].hideAxis('left')
                if irow<(len(ROW_CONDS)-1):
                    self.AX[irow][icol].hideAxis('bottom')
                self.AX[irow][icol].setYLink(self.AX[0][0]) # locking axis together
                self.AX[irow][icol].setXLink(self.AX[0][0])
            self.l.nextRow()
        self.AX[0][0].setRange(xRange=[self.EPISODES['t'][0], self.EPISODES['t'][-1]], yRange=ylim, padding=0.0)
            
        
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
            
    
    def build_column_conditions(self):
        X, K = [], []
        for i, key in enumerate(self.varied_parameters.keys()):
            if len(getattr(self, 'box%i'%i).currentText().split('column'))>1:
                X.append(np.sort(np.unique(self.parent.nwbfile.stimulus[key].data[self.Pcond])))
                K.append(key)
        return self.build_conditions(X, K)

    
    def build_row_conditions(self):
        X, K = [], []
        for i, key in enumerate(self.varied_parameters.keys()):
            if len(getattr(self, 'box%i'%i).currentText().split('row'))>1:
                X.append(np.sort(np.unique(self.parent.nwbfile.stimulus[key].data[self.Pcond])))
                K.append(key)
        return self.build_conditions(X, K)

    def build_color_conditions(self):
        X, K = [], []
        for i, key in enumerate(self.varied_parameters.keys()):
            if len(getattr(self, 'box%i'%i).currentText().split('color'))>1:
                X.append(np.sort(np.unique(self.parent.nwbfile.stimulus[key].data[self.Pcond])))
                K.append(key)
        return self.build_conditions(X, K)

        
if __name__=='__main__':

    import sys
    sys.path.append('/home/yann/work/physion')
    from physion.analysis.read_NWB import read as read_NWB
    
    class Parent:
        def __init__(self, filename=''):
            read_NWB(self, filename, verbose=False)
            self.app = QtWidgets.QApplication(sys.argv)
            self.description=''
            self.roiIndices = [0]
            self.CaImaging_key = 'Fluorescence'

    filename = sys.argv[-1]

    if '.nwb' in sys.argv[-1]:
        cls = Parent(filename)
        window = FiguresWindow(cls)
        sys.exit(cls.app.exec_())
    else:
        print('/!\ Need to provide a NWB datafile as argument ')
    
