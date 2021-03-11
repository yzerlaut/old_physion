import sys, time, tempfile, os, pathlib, json, datetime, string
from PyQt5 import QtGui, QtWidgets, QtCore
import numpy as np
import pyqtgraph as pg
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import day_folder
from dataviz.guiparts import NewWindow, smallfont
from scipy.interpolate import interp1d
from misc.colors import build_colors_from_array

class TrialAverageWindow(NewWindow):

    def __init__(self, 
                 parent=None,
                 dt_sampling=10, # ms
                 title='Trial-Averaging'):

        super(TrialAverageWindow, self).__init__(parent=parent.app,
                                                 title=title)

        self.parent = parent
        self.EPISODES, self.AX, self.l = None, None, None
        
        mainLayout = QtWidgets.QHBoxLayout(self.cwidget)
        Layout1 = QtWidgets.QVBoxLayout()
        mainLayout.addLayout(Layout1)
        Layout2 = QtWidgets.QVBoxLayout()
        mainLayout.addLayout(Layout2)
        
        self.Layout11 = QtWidgets.QHBoxLayout()
        Layout1.addLayout(self.Layout11)

        # description
        self.notes = QtWidgets.QLabel(parent.description, self)
        noteBoxsize = (200, 200)
        self.notes.setMinimumHeight(noteBoxsize[1])
        self.notes.setMaximumHeight(noteBoxsize[1])
        self.notes.setMinimumWidth(noteBoxsize[0])
        self.notes.setMaximumWidth(noteBoxsize[0])
        self.Layout11.addWidget(self.notes)
        
        # controls
        self.Layout12 = QtWidgets.QVBoxLayout()
        Layout1.addLayout(self.Layout12)
        self.Layout12.addWidget(QtWidgets.QLabel('Quantity', self))
        self.qbox = QtWidgets.QComboBox(self)
        if 'Photodiode-Signal' in self.parent.nwbfile.acquisition:
            self.qbox.addItem('Photodiode')
        if 'Electrophysiological-Signal' in self.parent.nwbfile.acquisition:
            self.qbox.addItem('Electrophy')
        if 'Running-Speed' in self.parent.nwbfile.acquisition:
            self.qbox.addItem('Running-Speed')
        if 'Pupil' in self.parent.nwbfile.processing:
            self.qbox.addItem('Pupil')
        if 'ophys' in self.parent.nwbfile.processing:
            self.qbox.addItem('CaImaging')
        self.qbox.activated.connect(self.update_quantity)
        self.Layout12.addWidget(self.qbox)
        

        
        self.Layout12.addWidget(QtWidgets.QLabel('', self))
        self.guiKeywords = QtGui.QLineEdit()
        self.guiKeywords.setText('  [GUI keywords]  ')
        self.guiKeywords.setFixedWidth(250)
        self.guiKeywords.returnPressed.connect(self.keyword_update2)
        self.guiKeywords.setFont(smallfont)
        self.Layout12.addWidget(self.guiKeywords)
        
        self.Layout12.addWidget(QtWidgets.QLabel('', self))
        self.roiPick = QtGui.QLineEdit()
        self.roiPick.setText('  [select ROI]  ')
        self.roiPick.setFixedWidth(250)
        self.roiPick.returnPressed.connect(self.select_ROI)
        self.roiPick.setFont(smallfont)
        self.Layout12.addWidget(self.roiPick)

        self.Layout12.addWidget(QtWidgets.QLabel('', self))
        self.Layout12.addWidget(QtWidgets.QLabel(9*'-'+\
                                                 ' Display options '+\
                                                 9*'-', self))
        for key in self.parent.keys:
            setattr(self, "c"+key, QtWidgets.QComboBox(self))
            self.Layout12.addWidget(getattr(self, "c"+key))
            for k in ['(merge)', '(color-code)', '(row)', '(column)']:
                getattr(self, "c"+key).addItem(key+\
                            ((30-len(k)-len(key))*' ')+k)
        for i in range(4-len(self.parent.keys)):
            self.Layout12.addWidget(QtWidgets.QLabel('', self))

        self.refreshBtn = QtWidgets.QPushButton('[R]efresh plots', self)
        self.Layout12.addWidget(self.refreshBtn)
        self.Layout12.addWidget(QtWidgets.QLabel('', self))
        self.samplingBox = QtWidgets.QDoubleSpinBox(self)
        self.samplingBox.setValue(dt_sampling)
        self.samplingBox.setMaximum(500)
        self.samplingBox.setMinimum(0.1)
        self.samplingBox.setSuffix(' (ms) sampling')
        self.Layout12.addWidget(self.samplingBox)

        self.plots = pg.GraphicsLayoutWidget()
        Layout2.addWidget(self.plots)
        
        self.show()

    def select_ROI(self):
        """ see dataviz/gui.py """
        roiIndices = self.parent.select_ROI_from_pick(cls=self)
        if len(roiIndices)>0:
            self.parent.roiIndices = roiIndices
            self.parent.roiPick.setText(self.roiPick.text())
        self.statusBar.showMessage('ROIs set to %s' % self.parent.roiIndices)


    def keyword_update2(self):
        self.keyword_update(string=self.guiKeywords.text(), parent=self.parent)

        # if self.guiKeywords.text() in ['F', 'meanImgE', 'Vcorr', 'max_proj']:
        #     self.CaImaging_bg_key = self.guiKeywords.text()
        # else:
        #     self.statusBar.showMessage('  /!\ keyword not recognized /!\ ')
        # plots.raw_data_plot(self, self.tzoom, with_roi=True)
        

    def plot_row_column_of_quantity(self, quantity):

        
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
        self.AX = []
        for irow, row_cond in enumerate(ROW_CONDS):
            self.AX.append([])
            for icol, col_cond in enumerate(COL_CONDS):
                self.AX[irow].append(self.l.addPlot())
                for icolor, color_cond in enumerate(COLOR_CONDS):
                    cond = np.array(col_cond & row_cond & color_cond)[:len(self.EPISODES[quantity])]
                    pen = pg.mkPen(color=COLORS[icolor], width=2)
                    my = np.array(self.EPISODES[quantity])[cond,:].mean(axis=0)
                    if np.sum(cond)>1:
                        spen = pg.mkPen(color=(0,0,0,0), width=0)
                        spenbrush = pg.mkBrush(color=(*COLORS[icolor][:3], 100))
                        sy = np.array(self.EPISODES[quantity])[cond,:].std(axis=0)
                        phigh = pg.PlotCurveItem(self.EPISODES['t'], my+sy, pen = spen)
                        plow = pg.PlotCurveItem(self.EPISODES['t'], my-sy, pen = spen)
                        pfill = pg.FillBetweenItem(phigh, plow, brush=spenbrush)
                        self.AX[irow][icol].addItem(phigh)
                        self.AX[irow][icol].addItem(plow)
                        self.AX[irow][icol].addItem(pfill)                    
                    self.AX[irow][icol].plot(self.EPISODES['t'], my, pen = pen)
                if icol>0:
                    self.AX[irow][icol].hideAxis('left')
                    self.AX[irow][icol].setYLink(self.AX[irow][0])
            self.l.nextRow()
        for irow, row_cond in enumerate(ROW_CONDS):
            for icol, col_cond in enumerate(COL_CONDS):
                if irow<(len(ROW_CONDS)-1):
                    self.AX[irow][icol].hideAxis('bottom')
                    self.AX[irow][icol].setXLink(self.AX[-1][icol])
        
    def refresh(self):

        self.plots.clear()
        if self.l is not None:
            self.l.setParent(None) # this is how you remove a layout
        
        self.quantity = self.qbox.currentText()
        if self.quantity=='CaImaging':
            self.statusBar.showMessage('rebuilding episodes for "%s" and ROI "%s" [...]' % (self.quantity,\
                                                                              self.parent.roiIndices))
        else:
            self.statusBar.showMessage('rebuilding episodes for "%s" [...]' % self.quantity)
        self.EPISODES = build_episodes(self, parent=self.parent,
                                       quantity=self.quantity,
                                       dt_sampling=self.samplingBox.value())
        self.statusBar.showMessage('-> done !')
        self.plot_row_column_of_quantity(self.quantity)
        
    def build_conditions(self, X, K):
        if len(K)>0:
            CONDS = []
            XK = np.meshgrid(*X)
            for i in range(len(XK[0].flatten())): # looping over joint conditions
                cond = np.ones(self.parent.nwbfile.stimulus['time_start'].data.shape[0], dtype=bool)
                for k, xk in zip(K, XK):
                    cond = cond & (self.parent.nwbfile.stimulus[k].data[:]==xk.flatten()[i])
                CONDS.append(cond)
            return CONDS
        else:
            return [np.ones(self.parent.nwbfile.stimulus['time_start'].data.shape[0], dtype=bool)]
            
    
    def build_column_conditions(self):
        X, K = [], []
        for key in self.parent.keys:
            if len(getattr(self, "c"+key).currentText().split('column'))>1:
                X.append(np.sort(np.unique(self.parent.nwbfile.stimulus[key].data[:])))
                K.append(key)
        return self.build_conditions(X, K)

    
    def build_row_conditions(self):
        X, K = [], []
        for key in self.parent.keys:
            if len(getattr(self, "c"+key).currentText().split('row'))>1:
                X.append(np.sort(np.unique(self.parent.nwbfile.stimulus[key].data[:])))
                K.append(key)
        return self.build_conditions(X, K)

    def build_color_conditions(self):
        X, K = [], []
        for key in self.parent.keys:
            if len(getattr(self, "c"+key).currentText().split('color'))>1:
                X.append(np.sort(np.unique(self.parent.nwbfile.stimulus[key].data[:])))
                K.append(key)
        return self.build_conditions(X, K)

    
    def update_quantity(self):
        pass
        # if self.qbox.currentText()=='CaImaging':
        #     self.sqbox.addItem('[sum]')
        #     self.sqbox.addItem('[all] (row)')
        #     self.sqbox.addItem('[all] (color-code)')
        #     for i in range(np.sum(self.parent.iscell)):
        #         self.sqbox.addItem('ROI-%i' % (i+1))
        #     for k in ['Fluorescence', 'Neuropil', 'Deconvolved', 'dF (F-0.7*Fneu)']:
        #         self.pbox.addItem(k)

        
def build_episodes(self,
                   parent=None,
                   protocol_id=0,
                   quantity='Photodiode-Signal',
                   subquantity='',
                   dt_sampling=1, # ms
                   interpolation='linear',
                   verbose=True):

    EPISODES = {'dt_sampling':dt_sampling,
                'quantity':quantity}

    if parent is None:
        parent = self

    # choosing protocol (if multiprotocol)
    if len(np.unique(parent.nwbfile.stimulus['protocol_id'].data[:]))>1:
        Pcond = (parent.nwbfile.stimulus['protocol_id'].data[:]==protocol_id)
    else:
        Pcond = np.ones(parent.nwbfile.stimulus['time_start'].data.shape[0], dtype=bool)
    if verbose:
        print('Number of episodse over the whole recording: %i/%i (with protocol condition)' % (np.sum(Pcond), len(Pcond)))
        
    # new sampling
    interstim = parent.metadata['presentation-interstim-period']
    ipre = int(interstim/dt_sampling/1e-3*3./4.) # 3/4 of prestim
    duration = parent.nwbfile.stimulus['time_stop'].data[Pcond][0]-parent.nwbfile.stimulus['time_start'].data[Pcond][0]
    idur = int(duration/dt_sampling/1e-3)
    EPISODES['t'] = np.arange(-ipre+1, idur+ipre-1)*dt_sampling*1e-3
    EPISODES[quantity] = []
    for key in parent.nwbfile.stimulus.keys():
        EPISODES[key] = parent.nwbfile.stimulus[key].data[Pcond]
        
    if quantity=='CaImaging':
        if 'CaImaging-TimeSeries' in parent.nwbfile.acquisition and len(parent.nwbfile.acquisition['CaImaging-TimeSeries'].timestamps)>2:
            tfull = parent.nwbfile.acquisition['CaImaging-TimeSeries'].timestamps
        else:
            dt = parent.nwbfile.processing['ophys']['Neuropil'].roi_response_series['Neuropil'].timestamps[1]-parent.nwbfile.processing['ophys']['Neuropil'].roi_response_series['Neuropil'].timestamps[0]
            tfull = np.arange(parent.nwbfile.processing['ophys']['Neuropil'].roi_response_series['Neuropil'].data.shape[1])*dt
        if len(parent.roiIndices)>1:
            valfull = getattr(parent, parent.CaImaging_key).data[parent.validROI_indices[np.array(parent.roiIndices)], :].mean(axis=0)
        elif len(parent.roiIndices)==1:
            valfull = getattr(parent, parent.CaImaging_key).data[parent.validROI_indices[parent.roiIndices[0]], :]
        else:
            valfull = getattr(parent, parent.CaImaging_key).data[parent.validROI_indices[parent.roiIndices], :].sum(axis=0)
    else:
        try:
            tfull = np.arange(parent.nwbfile.acquisition[quantity].data.shape[0])/parent.nwbfile.acquisition[quantity].rate
            valfull = parent.nwbfile.acquisition[quantity].data[:]
        except BaseException as be:
            print(be)
            print(30*'-')
            print(quantity, 'not recognized')
            print(30*'-')
        
    for tstart, tstop in zip(parent.nwbfile.stimulus['time_start_realigned'].data[Pcond],
                             parent.nwbfile.stimulus['time_stop_realigned'].data[Pcond]):

        cond = (tfull>=(tstart-interstim)) & (tfull<(tstop+interstim))
        func = interp1d(tfull[cond]-tstart, valfull[cond],
                        kind=interpolation)
        
        try:
            EPISODES[quantity].append(func(EPISODES['t']))
        except ValueError:
            print(tstart, tstop)
            pass

    if verbose:
        print('[ok] episodes ready !')
    return EPISODES
    
if __name__=='__main__':

    # folder = os.path.join(os.path.expanduser('~'),\
    #                       'DATA', '2020_11_04', '01-02-03')
    
    # dataset = Dataset(folder,
    #                   with_CaImaging_stat=False,
    #                   modalities=['Screen', 'Locomotion', 'CaImaging'])
    
    # app = QtWidgets.QApplication(sys.argv)
    # from misc.colors import build_dark_palette
    # build_dark_palette(app)
    # window = TrialAverageWindow(app, dataset=dataset)
    # sys.exit(app.exec_())
    import sys
    sys.path.append('/home/yann/work/physion')
    from physion.analysis.read_NWB import read as read_NWB

    class Data:
        def __init__(self, filename):
            read_NWB(self, filename, verbose=False)
            self.CaImaging_key = 'Fluorescence'
            self.roiIndices = [0]

    key = 'CaImaging'
    filename = os.path.join(os.path.expanduser('~'), 'DATA', 'data.nwb')
    data = Data(filename)
    EPISODES = build_episodes(data, quantity=key, protocol_id=1, verbose=True)

    import matplotlib.pylab as plt
    for i in range(10):
        plt.plot(EPISODES['t'], EPISODES[key][i])
    plt.show()
