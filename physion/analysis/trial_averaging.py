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

NMAX_PARAMS = 5

class TrialAverageWindow(NewWindow):

    def __init__(self, 
                 parent=None,
                 dt_sampling=10, # ms
                 title='Trial-Averaging'):

        super(TrialAverageWindow, self).__init__(parent=parent.app,
                                                 title=title)

        self.parent = parent
        self.EPISODES, self.AX, self.l = None, None, None

        self.computeSc = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+C'), self)
        self.computeSc.activated.connect(self.compute_episodes)
        
        mainLayout = QtWidgets.QHBoxLayout(self.cwidget)
        Layout1 = QtWidgets.QVBoxLayout()
        mainLayout.addLayout(Layout1)
        Layout2 = QtWidgets.QVBoxLayout()
        mainLayout.addLayout(Layout2)
        
        self.Layout11 = QtWidgets.QHBoxLayout()
        Layout1.addLayout(self.Layout11)

        # description
        self.notes = QtWidgets.QLabel(parent.description, self)
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
        self.pbox.addItems(self.parent.protocols)
        self.pbox.activated.connect(self.update_protocol)
        self.Layout12.addWidget(self.pbox)

        # -- quantity
        self.Layout12.addWidget(QtWidgets.QLabel('Quantity', self))
        self.qbox = QtWidgets.QComboBox(self)
        self.qbox.addItem('')
        if 'ophys' in self.parent.nwbfile.processing:
            self.qbox.addItem('CaImaging')
        for key in parent.nwbfile.acquisition:
            if len(parent.nwbfile.acquisition[key].data.shape)==1:
                self.qbox.addItem(key) # only for scalar variables
        self.qbox.activated.connect(self.update_quantity)
        self.Layout12.addWidget(self.qbox)

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
        
    def refresh(self):
        self.plots.clear()
        if self.l is not None:
            self.l.setParent(None) # this is how you remove a layout
        self.plot_row_column_of_quantity()
        
    def update_quantity(self):
        pass

    def compute_episodes(self):
        if (self.qbox.currentIndex()>0) and (self.pbox.currentIndex()>0):
            self.EPISODES = build_episodes(self,
                                       parent=self.parent,
                                       protocol_id=self.pbox.currentIndex()-1,
                                       quantity=self.qbox.currentText(),
                                       dt_sampling=self.samplingBox.value(), # ms
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
        for i, key in enumerate(self.varied_parameters.keys()):
            for k in ['(merge)', '(color-code)', '(row)', '(column)']:
                getattr(self, "box%i"%i).addItem(key+((30-len(k)-len(key))*' ')+k)
        
    def select_ROI(self):
        """ see dataviz/gui.py """
        roiIndices = self.parent.select_ROI_from_pick(cls=self)
        if len(roiIndices)>0:
            self.parent.roiIndices = roiIndices
            self.parent.roiPick.setText(self.roiPick.text())
        self.statusBar.showMessage('ROIs set to %s' % self.parent.roiIndices)


    def keyword_update2(self):
        self.keyword_update(string=self.guiKeywords.text(), parent=self.parent)

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

        
def build_episodes(self,
                   parent=None,
                   protocol_id=0,
                   quantity='Photodiode-Signal',
                   prestim_duration=None, # to force the prestim window otherwise, half the value in between episodes
                   dt_sampling=1, # ms
                   interpolation='linear',
                   baseline_substraction=False,
                   verbose=True):

    EPISODES = {'dt_sampling':dt_sampling,
                'quantity':quantity,
                'resp':[]}

    parent = (parent if parent is not None else self)

    # choosing protocol (if multiprotocol)
    if ('protocol_id' in parent.nwbfile.stimulus) and (len(np.unique(parent.nwbfile.stimulus['protocol_id'].data[:]))>1):
        Pcond = (parent.nwbfile.stimulus['protocol_id'].data[:]==protocol_id)
    else:
        Pcond = np.ones(parent.nwbfile.stimulus['time_start'].data.shape[0], dtype=bool)
    # limiting to available episodes
    Pcond[np.arange(len(Pcond))>=parent.nwbfile.stimulus['time_start_realigned'].num_samples] = False
    
    if verbose:
        print('Number of episodes over the whole recording: %i/%i (with protocol condition)' % (np.sum(Pcond), len(Pcond)))

    # find the parameter(s) varied within that specific protocol
    EPISODES['varied_parameters'] =  {}
    for key in parent.nwbfile.stimulus.keys():
        if key not in ['frame_run_type', 'index', 'protocol_id', 'time_duration', 'time_start',
                       'time_start_realigned', 'time_stop', 'time_stop_realigned']:
            unique = np.unique(parent.nwbfile.stimulus[key].data[Pcond])
            if len(unique)>1:
                EPISODES['varied_parameters'][key] = unique
    # for the parent class
    self.varied_parameters = EPISODES['varied_parameters'] # adding this as a shortcut
    self.Pcond = Pcond # protocol condition
    
    # new sampling
    if (prestim_duration is None) and ('interstim' in parent.nwbfile.stimulus):
        prestim_duration = np.min(parent.nwbfile.stimulus['interstim'].data[:])/2. # half the stim duration
    elif prestim_duration is None:
        prestim_duration = 1
    ipre = int(prestim_duration/dt_sampling*1e3)
        
    duration = parent.nwbfile.stimulus['time_stop'].data[Pcond][0]-parent.nwbfile.stimulus['time_start'].data[Pcond][0]
    idur = int(duration/dt_sampling/1e-3)
    EPISODES['t'] = np.arange(-ipre+1, idur+ipre-1)*dt_sampling*1e-3
    
    if quantity=='CaImaging':
        tfull = parent.Neuropil.timestamps[:]
        valfull = compute_CaImaging_trace(parent, parent.CaImaging_key, parent.roiIndices).sum(axis=0) # valid ROI indices inside
    else:
        try:
            tfull = np.arange(parent.nwbfile.acquisition[quantity].data.shape[0])/parent.nwbfile.acquisition[quantity].rate
            valfull = parent.nwbfile.acquisition[quantity].data[:]
        except BaseException as be:
            print(be)
            print(30*'-')
            print(quantity, 'not recognized')
            print(30*'-')
        
    # adding the parameters
    for key in parent.nwbfile.stimulus.keys():
        EPISODES[key] = []

    for iEp in np.arange(parent.nwbfile.stimulus['time_start'].num_samples)[Pcond]:
        tstart = parent.nwbfile.stimulus['time_start_realigned'].data[iEp]
        tstop = parent.nwbfile.stimulus['time_stop_realigned'].data[iEp]

        # compute time and interpolate
        cond = (tfull>=(tstart-1.5*prestim_duration)) & (tfull<(tstop+1.5*prestim_duration)) # higher range of interpolation to avoid boundary problems
        func = interp1d(tfull[cond]-tstart, valfull[cond],
                        kind=interpolation)
        try:
            if baseline_substraction:
                y = func(EPISODES['t'])
                EPISODES['resp'].append(y-np.mean(y[EPISODES['t']<0]))
            else:
                EPISODES['resp'].append(func(EPISODES['t']))
            for key in parent.nwbfile.stimulus.keys():
                EPISODES[key].append(parent.nwbfile.stimulus[key].data[iEp])
        except BaseException as be:
            print('----')
            print(be)
            print('Problem with episode %i between (%.2f, %.2f)s' % (iEp, tstart, tstop))

    EPISODES['resp'] = np.array(EPISODES['resp'])
    for key in parent.nwbfile.stimulus.keys():
        EPISODES[key] = np.array(EPISODES[key])
    
    if verbose:
        print('[ok] episodes ready !')
        
    return EPISODES
    
if __name__=='__main__':

    # folder = os.path.join(os.path.expanduser('~'),\
    #                       'DATA', '2020_11_04', '01-02-03')
    
    # dataset = Dataset(folder,
    #                   with_CaImaging_stat=False,
    #                   modalities=['Screen', 'Locomotion', 'CaImaging'])
    import sys
    sys.path.append('/home/yann/work/physion')
    from physion.analysis.read_NWB import read as read_NWB
    
    # filename = os.path.join(os.path.expanduser('~'), 'DATA', 'data.nwb')
    filename = sys.argv[-1]
    
    class Parent:
        def __init__(self, filename=''):
            read_NWB(self, filename, verbose=False)
            self.app = QtWidgets.QApplication(sys.argv)
            self.description=''
            self.roiIndices = [0]
            self.CaImaging_key = 'Fluorescence'
            
    cls = Parent(filename)
    window = TrialAverageWindow(cls)
    sys.exit(cls.app.exec_())
    

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

    
