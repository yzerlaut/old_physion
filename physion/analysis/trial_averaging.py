import sys, time, tempfile, os, pathlib, json, datetime, string
from PyQt5 import QtGui, QtWidgets, QtCore
import numpy as np
import pyqtgraph as pg

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import day_folder
from misc.guiparts import NewWindow, smallfont
from Ca_imaging.tools import compute_CaImaging_trace
from scipy.interpolate import interp1d
from misc.colors import build_colors_from_array
from analysis.process_NWB import EpisodeResponse

NMAX_PARAMS = 8

class TrialAverageWindow(NewWindow):

    def __init__(self, 
                 parent=None,
                 dt_sampling=10, # ms
                 title='Trial-Averaging'):

        super(TrialAverageWindow, self).__init__(parent=parent.app,
                                                 title=title)

        self.data = parent.data
        self.roiIndices, self.CaImaging_key = [0], parent.CaImaging_key
        self.l = None
        
        self.computeSc = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+C'), self)
        self.computeSc.activated.connect(self.compute_episodes)
        self.nextSc = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+N'), self)
        self.nextSc.activated.connect(self.next_roi)
        
        mainLayout = QtWidgets.QHBoxLayout(self.cwidget)
        Layout1 = QtWidgets.QVBoxLayout()
        mainLayout.addLayout(Layout1)
        Layout2 = QtWidgets.QVBoxLayout()
        mainLayout.addLayout(Layout2)
        
        self.Layout11 = QtWidgets.QHBoxLayout()
        Layout1.addLayout(self.Layout11)

        # description
        self.notes = QtWidgets.QLabel(self.data.description, self)
        noteBoxsize = (200, 100)
        self.notes.setMinimumHeight(noteBoxsize[1])
        self.notes.setMaximumHeight(noteBoxsize[1])
        self.notes.setMinimumWidth(noteBoxsize[0])
        self.notes.setMaximumWidth(noteBoxsize[0])
        self.Layout11.addWidget(self.notes)
        

        self.Layout12 = QtWidgets.QVBoxLayout()
        Layout1.addLayout(self.Layout12)
        # -- protocol
        self.Layout12.addWidget(QtWidgets.QLabel('Protocol', self))
        self.pbox = QtWidgets.QComboBox(self)
        self.pbox.addItem('')
        self.pbox.addItems(self.data.protocols)
        self.pbox.activated.connect(self.update_protocol)
        self.Layout12.addWidget(self.pbox)

        # -- quantity
        self.Layout12.addWidget(QtWidgets.QLabel('Quantity', self))
        self.qbox = QtWidgets.QComboBox(self)
        self.qbox.addItem('')
        if 'ophys' in self.data.nwbfile.processing:
            self.qbox.addItem('CaImaging')
        if 'Pupil' in self.data.nwbfile.processing:
            self.qbox.addItem('pupil-size')
            self.qbox.addItem('gaze-movement')
        if 'FaceMotion' in self.data.nwbfile.processing:
            self.qbox.addItem('facemotion')
        for key in self.data.nwbfile.acquisition:
            if len(self.data.nwbfile.acquisition[key].data.shape)==1:
                self.qbox.addItem(key) # only for scalar variables
        self.qbox.activated.connect(self.update_quantity)
        self.Layout12.addWidget(self.qbox)

        self.Layout12.addWidget(QtWidgets.QLabel('sub-quantity', self))
        self.sqbox = QtWidgets.QComboBox(self)
        self.sqbox.addItem('')
        self.Layout12.addWidget(self.sqbox)
        
        self.guiKeywords = QtGui.QLineEdit()
        self.guiKeywords.setText('  [GUI keywords]  ')
        self.guiKeywords.setFixedWidth(250)
        self.guiKeywords.returnPressed.connect(self.keyword_update2)
        self.guiKeywords.setFont(smallfont)
        self.Layout12.addWidget(self.guiKeywords)
        
        self.roiPick = QtGui.QLineEdit()
        self.roiPick.setText('  [select ROI]  ')
        self.roiPick.setFixedWidth(250)
        self.roiPick.returnPressed.connect(self.select_ROI)
        self.roiPick.setFont(smallfont)
        self.Layout12.addWidget(self.roiPick)
        self.baselineCB = QtGui.QCheckBox("baseline substraction")
        self.Layout12.addWidget(self.baselineCB)

        self.Layout12.addWidget(QtWidgets.QLabel('', self))
        self.computeBtn = QtWidgets.QPushButton('[Ctrl+C]ompute episodes', self)
        self.computeBtn.clicked.connect(self.compute_episodes_wsl)
        self.Layout12.addWidget(self.computeBtn)
        self.Layout12.addWidget(QtWidgets.QLabel('', self))

        self.Layout12.addWidget(QtWidgets.QLabel('', self))
        self.nextBtn = QtWidgets.QPushButton('[Ctrl+N]ext roi', self)
        # self.nextBtn.clicked.connect(self.compute_episodes_wsl)
        self.nextBtn.clicked.connect(self.next_roi)
        self.Layout12.addWidget(self.nextBtn)
        self.Layout12.addWidget(QtWidgets.QLabel('', self))
        
        # then parameters
        self.Layout13 = QtWidgets.QVBoxLayout()
        Layout1.addLayout(self.Layout13)
        self.Layout13.addWidget(QtWidgets.QLabel('', self))
        self.Layout13.addWidget(QtWidgets.QLabel(9*'-'+' Display options '+9*'-', self))

        for i in range(NMAX_PARAMS): # controls the max number of parameters varied
            setattr(self, "box%i"%i, QtWidgets.QComboBox(self))
            self.Layout13.addWidget(getattr(self, "box%i"%i))
        self.Layout13.addWidget(QtWidgets.QLabel(' ', self))

        self.refreshBtn = QtWidgets.QPushButton('[Ctrl+R]efresh plots', self)
        self.refreshBtn.clicked.connect(self.refresh)
        self.Layout13.addWidget(self.refreshBtn)
        self.Layout13.addWidget(QtWidgets.QLabel('', self))
        
        self.samplingBox = QtWidgets.QDoubleSpinBox(self)
        self.samplingBox.setValue(dt_sampling)
        self.samplingBox.setMaximum(500)
        self.samplingBox.setMinimum(0.1)
        self.samplingBox.setSuffix(' (ms) sampling')
        self.Layout13.addWidget(self.samplingBox)

        self.plots = pg.GraphicsLayoutWidget()
        Layout2.addWidget(self.plots)
        
        self.show()

    def update_params_choice(self):
        pass

    def update_protocol(self):
        self.EPISODES = None
        # self.qbox.setCurrentIndex(0)

    def hitting_space(self):
        self.compute_episodes()
        self.refresh()

    def next_roi(self):
        if len(self.roiIndices)==1:
            self.roiIndices = [np.min([np.sum(self.data.iscell)-1, self.roiIndices[0]+1])]
        else:
            self.roiIndices = [0]
            self.statusBar.showMessage('ROIs set to %s' % self.roiIndices)
        self.roiPick.setText('%i' % self.roiIndices[0])
        self.compute_episodes()
        self.refresh()
        
    def refresh(self):
        self.plots.clear()
        if self.l is not None:
            self.l.setParent(None) # this is how you remove a layout
        self.plot_row_column_of_quantity()
        
    def update_quantity(self):
        self.sqbox.clear()
        self.sqbox.addItems(self.data.list_subquantities(self.qbox.currentText()))
        self.sqbox.setCurrentIndex(0)

    def compute_episodes(self):
        self.select_ROI()
        if (self.qbox.currentIndex()>0) and (self.pbox.currentIndex()>0):
            self.EPISODES = EpisodeResponse(self.data,
                                            protocol_id=self.pbox.currentIndex()-1,
                                            quantity=self.qbox.currentText(),
                                            subquantity=self.sqbox.currentText(),
                                            dt_sampling=self.samplingBox.value(), # ms
                                            roiIndices=self.roiIndices,
                                            interpolation='linear',
                                            baseline_substraction=self.baselineCB.isChecked(),
                                            verbose=True)
        else:
            print(' /!\ Pick a protocol an a quantity')

    def compute_episodes_wsl(self):
        self.compute_episodes()
        self.update_selection() # with update selection
            
    def update_selection(self):
        for i in range(NMAX_PARAMS):
            getattr(self, "box%i"%i).clear()
        for i, key in enumerate(self.EPISODES.varied_parameters.keys()):
            for k in ['(merge)', '(color-code)', '(row)', '(column)']:
                getattr(self, "box%i"%i).addItem(key+((30-len(k)-len(key))*' ')+k)
        
    def select_ROI(self):
        """ see dataviz/gui.py """
        try:
            self.roiIndices = [int(self.roiPick.text())]
            self.statusBar.showMessage('ROIs set to %s' % self.roiIndices)
        except BaseException:
            self.roiIndices = [0]
            self.roiPick.setText('0')
            self.statusBar.showMessage('/!\ ROI string not recognized /!\ --> ROI set to [0]')

            
    def keyword_update2(self):
        self.keyword_update(string=self.guiKeywords.text(), parent=self.parent)

        
    def plot_row_column_of_quantity(self):

        self.Pcond = self.data.get_protocol_cond(self.pbox.currentIndex()-1)
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
                    cond = np.array(col_cond & row_cond & color_cond)[:self.EPISODES.resp.shape[0]]
                    pen = pg.mkPen(color=COLORS[icolor], width=2)
                    if self.EPISODES.resp[cond,:].shape[0]>0:
                        my = self.EPISODES.resp[cond,:].mean(axis=0)
                        if np.sum(cond)>1:
                            spen = pg.mkPen(color=(0,0,0,0), width=0)
                            spenbrush = pg.mkBrush(color=(*COLORS[icolor][:3], 100))
                            sy = self.EPISODES.resp[cond,:].std(axis=0)
                            phigh = pg.PlotCurveItem(self.EPISODES.t, my+sy, pen = spen)
                            plow = pg.PlotCurveItem(self.EPISODES.t, my-sy, pen = spen)
                            pfill = pg.FillBetweenItem(phigh, plow, brush=spenbrush)
                            self.AX[irow][icol].addItem(phigh)
                            self.AX[irow][icol].addItem(plow)
                            self.AX[irow][icol].addItem(pfill)
                            ylim[0] = np.min([np.min(my-sy), ylim[0]])
                            ylim[1] = np.max([np.max(my+sy), ylim[1]])
                        else:
                            ylim[0] = np.min([np.min(my), ylim[0]])
                            ylim[1] = np.max([np.max(my), ylim[1]])
                        self.AX[irow][icol].plot(self.EPISODES.t, my, pen = pen)
                    else:
                        print(' /!\ Problem with episode (%i, %i, %i)' % (irow, icol, icolor))
                if icol>0:
                    self.AX[irow][icol].hideAxis('left')
                if irow<(len(ROW_CONDS)-1):
                    self.AX[irow][icol].hideAxis('bottom')
                self.AX[irow][icol].setYLink(self.AX[0][0]) # locking axis together
                self.AX[irow][icol].setXLink(self.AX[0][0])
            self.l.nextRow()
        self.AX[0][0].setRange(xRange=[self.EPISODES.t[0], self.EPISODES.t[-1]], yRange=ylim, padding=0.0)
            
        
    def build_column_conditions(self):
        X, K = [], []
        for i, key in enumerate(self.EPISODES.varied_parameters.keys()):
            if len(getattr(self, 'box%i'%i).currentText().split('column'))>1:
                X.append(np.sort(np.unique(self.data.nwbfile.stimulus[key].data[self.Pcond])))
                K.append(key)
        return self.data.get_stimulus_conditions(X, K, self.pbox.currentIndex()-1)
    
    def build_row_conditions(self):
        X, K = [], []
        for i, key in enumerate(self.EPISODES.varied_parameters.keys()):
            if len(getattr(self, 'box%i'%i).currentText().split('row'))>1:
                X.append(np.sort(np.unique(self.data.nwbfile.stimulus[key].data[self.Pcond])))
                K.append(key)
        return self.data.get_stimulus_conditions(X, K, self.pbox.currentIndex()-1)


    def build_color_conditions(self):
        X, K = [], []
        for i, key in enumerate(self.EPISODES.varied_parameters.keys()):
            if len(getattr(self, 'box%i'%i).currentText().split('color-code'))>1:
                X.append(np.sort(np.unique(self.data.nwbfile.stimulus[key].data[self.Pcond])))
                K.append(key)
        return self.data.get_stimulus_conditions(X, K, self.pbox.currentIndex()-1)
    
    
if __name__=='__main__':

    from analysis.read_NWB import Data
    
    class Parent:
        def __init__(self, filename=''):
            super().__init__()
            self.app = QtWidgets.QApplication(sys.argv)
            self.data = Data(filename)
            self.CaImaging_key = 'Fluorescence'
            # self.description=''
            # self.roiIndices = [0]
                
    filename = sys.argv[-1]

    if '.nwb' in sys.argv[-1]:
        cls = Parent(filename)
        window = TrialAverageWindow(cls)
        sys.exit(cls.app.exec_())
    else:
        print('/!\ Need to provide a NWB datafile as argument ')
    

    # class Data:
    #     def __init__(self, filename):
    #         read_NWB(self, filename, verbose=False)
    #         self.CaImaging_key = 'Fluorescence'
    #         self.roiIndices = np.arange(100)

    # data = Data(filename)
    # EPISODES = build_episodes(data, quantity='CaImaging', protocol_id=0, verbose=True)

    # import matplotlib.pylab as plt
    # plt.plot(EPISODES['t'], EPISODES['resp'].mean(axis=0))
    # plt.show()

    
