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
        self.guiKeywords.returnPressed.connect(self.keyword_update)
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


    def keyword_update(self):

        print(self.guiKeywords.text())
        # if self.guiKeywords.text() in ['F', 'meanImgE', 'Vcorr', 'max_proj']:
        #     self.CaImaging_bg_key = self.guiKeywords.text()
        # else:
        #     self.statusBar.setText('  /!\ keyword not recognized /!\ ')
        # plots.raw_data_plot(self, self.tzoom, with_roi=True)
        

    def plot_row_column_of_quantity(self, quantity):

        
        COL_CONDS = self.build_column_conditions()
        ROW_CONDS = self.build_row_conditions()
        COLOR_CONDS = self.build_color_conditions()

        print(COL_CONDS, ROW_CONDS, COLOR_CONDS)
        
        if len(COLOR_CONDS)>1:
            COLORS = build_colors_from_array(np.arange(len(COLOR_CONDS)))
        else:
            COLORS = [(255,255,255,255)]

        self.plots.clear()
        if self.l is not None:
            self.l.setParent(None)
            # self.l.deleteLater()
        self.l = self.plots.addLayout(rowspan=len(ROW_CONDS),
                                      colspan=len(COL_CONDS),
                                      border=(0,0,0))
        self.l.setContentsMargins(2, 2, 2, 2)
        self.l.layout.setSpacing(1.)            

        # re-adding stuff
        self.AX = []
        for irow, row_cond in enumerate(ROW_CONDS):
            self.AX.append([])
            for icol, col_cond in enumerate(COL_CONDS):
                self.AX[irow].append(self.l.addPlot())
                for icolor, color_cond in enumerate(COLOR_CONDS):
                    cond = col_cond & row_cond & color_cond
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
                    self.AX[irow][icol].setYLink(AX[irow][0])
            self.l.nextRow()
        for irow, row_cond in enumerate(ROW_CONDS):
            for icol, col_cond in enumerate(COL_CONDS):
                if irow<(len(ROW_CONDS)-1):
                    self.AX[irow][icol].hideAxis('bottom')
                    self.AX[irow][icol].setXLink(AX[-1][icol])
        
    def refresh(self):

        self.plots.clear()
        self.quantity = self.qbox.currentText()
        # if (self.EPISODES is None) or\
        #    (self.quantity!=self.EPISODES['quantity'])or\
        #    (self.samplingBox.value()!=self.EPISODES['dt_sampling']):
        self.EPISODES = build_episodes(self,
                            quantity=self.quantity,
                            dt_sampling=self.samplingBox.value())
        print(self.EPISODES)
        self.statusBar.showMessage('-> done !')
        self.plot_row_column_of_quantity(self.quantity)
        
    def build_conditions(self, X, K):
        if len(K)>0:
            CONDS = []
            XK = np.meshgrid(*X)
            print(XK)
            for i in range(len(XK[0].flatten())): # looping over joint conditions
                cond = np.ones(self.parent.nwbfile.stimulus['time_start'].data.shape[0], dtype=bool)
                for k, xk in zip(K, XK):
                    cond = cond & (self.parent.nwbfile.stimulus[k]==xk.flatten()[i])
                CONDS.append(cond)
            return CONDS
        else:
            return [np.ones(self.parent.nwbfile.stimulus['time_start'].data.shape[0], dtype=bool)]
            
    
    def build_column_conditions(self):
        X, K = [], []
        for key in self.parent.keys:
            if len(getattr(self, "c"+key).currentText().split('column'))>1:
                X.append(np.unique(self.parent.nwbfile.stimulus[key]))
                K.append(key)
        return self.build_conditions(X, K)

    
    def build_row_conditions(self):
        X, K = [], []
        for key in self.parent.keys:
            if len(getattr(self, "c"+key).currentText().split('row'))>1:
                X.append(np.sort(np.unique(self.parent.nwbfile.stimulus[key])))
                K.append(key)
        return self.build_conditions(X, K)

    def build_color_conditions(self):
        X, K = [], []
        for key in self.parent.keys:
            if len(getattr(self, "c"+key).currentText().split('color'))>1:
                X.append(np.sort(np.unique(self.parent.nwbfile.stimulus[key])))
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
    
    def update_subquantity(self):
        pass

def dataset_description(dataset):
    S = '%s\n\n%s -- %s\n\n' % (dataset.metadata['Stimulus'],
                                dataset.metadata['day'].replace('_', '/'),
                                dataset.metadata['time'].replace('-', ':'))
    KEYS = []
    for key in dataset.VisualStim:
        if key not in ['time_start', 'time_stop', 'index']:
            # print('then this is a parameter of the stimulus')
            if len(np.unique(dataset.VisualStim[key]))>1:
                S+='%s=[%s,%s]\n      ---> N-%s = %i' % (key,
                                                  np.min(dataset.VisualStim[key]),
                                                  np.max(dataset.VisualStim[key]),
                                                  key,
                                                  len(np.unique(dataset.VisualStim[key])))
                KEYS.append(key)
            else:
                S+='%s = %.2f' % (key, dataset.VisualStim[key][0])
            S+='\n'
    if 'time_start_realigned' in dataset.metadata:
        S += 'completed N=%i/%i episodes' %(\
                       len(dataset.metadata['time_start_realigned']), len(dataset.metadata['time_start']))
    else:
        print('"time_start_realigned" not available')
        print('--> Need to realign data with respect to Visual-Stimulation !!')

    quantities = {}
    if dataset.Screen is not None:
        if dataset.Screen.photodiode is not None:
            quantities['Photodiode'] = {}
    if dataset.CaImaging is not None:
        quantities['CaImaging'] = {}
    if dataset.Locomotion is not None:
        quantities['Locomotion'] = {}
    if dataset.Electrophy is not None:
        quantities['Electrophy'] = {}

    return S, KEYS, quantities

def build_episodes(self,
                   quantity='Locomotion',
                   subquantity='',
                   dt_sampling=1, # ms
                   interpolation='linear'):

    EPISODES = {'dt_sampling':dt_sampling,
                'quantity':quantity}
    
    self.statusBar.showMessage('building episodes [...]')
    # new sampling
    interstim = self.parent.metadata['presentation-interstim-period']
    ipre = int(interstim/dt_sampling/1e-3*3./4.) # 3/4 of prestim
    idur = int(self.parent.metadata['presentation-duration']/dt_sampling/1e-3)
    EPISODES['t'] = np.arange(-ipre+1, idur+ipre-1)*dt_sampling*1e-3
    EPISODES[quantity] = []
    if quantity=='Photodiode':
        tfull = np.arange(self.parent.nwbfile.acquisition['Photodiode-Signal'].data.shape[0])/self.parent.nwbfile.acquisition['Photodiode-Signal'].rate
        valfull = self.parent.nwbfile.acquisition['Photodiode-Signal'].data
    elif quantity=='CaImaging':
        if 'CaImaging-TimeSeries' in self.parent.nwbfile.acquisition:
            tfull = self.parent.nwbfile.acquisition['CaImaging-TimeSeries'].timestamps
        else:
            dt = 1./np.self.parent.nwbfile.processing['ophys'].rate
            tfull = np.arange(self.nwbfile.processing['ophys']['Neuropil'].roi_response_series['Neuropil'].data.shape[1])*dt
        if len(self.parent.roiIndices)>1:
            valfull = getattr(self.parent, self.parent.CaImaging_key).data[self.parent.iscell[self.parent.roiIndices], :].mean(axis=0)
        elif len(self.parent.roiIndices)==1:
            valfull = getattr(self.parent, self.parent.CaImaging_key).data[self.parent.iscell[self.parent.roiIndices[0]], :]
        else:
            valfull = getattr(self.parent, self.parent.CaImaging_key).data[self.parent.iscell[self.parent.roiIndices], :].sum(axis=0)
    else:
        print(quantity, 'not recognized')
    
    for tstart, tstop in zip(self.parent.nwbfile.stimulus['time_start_realigned'].data[:],\
                             self.parent.nwbfile.stimulus['time_stop_realigned'].data[:]):

        cond = (tfull>=(tstart-interstim)) & (tfull<(tstop+interstim))
        func = interp1d(tfull[cond]-tstart, valfull[cond],
                        kind=interpolation)
        EPISODES[quantity].append(func(EPISODES['t']))
            
    print('[ok] episodes ready !')
    return EPISODES
    
if __name__=='__main__':

    folder = os.path.join(os.path.expanduser('~'),\
                          'DATA', '2020_11_04', '01-02-03')
    
    dataset = Dataset(folder,
                      with_CaImaging_stat=False,
                      modalities=['Screen', 'Locomotion', 'CaImaging'])
    
    app = QtWidgets.QApplication(sys.argv)
    from misc.colors import build_dark_palette
    build_dark_palette(app)
    window = TrialAverageWindow(app, dataset=dataset)
    sys.exit(app.exec_())
