import sys, time, tempfile, os, pathlib, json, datetime, string
import numpy as np
import pyqtgraph as pg
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import day_folder
from assembling.dataset import Dataset, MODALITIES
from guiparts import NewWindow, build_dark_palette
from scipy.interpolate import interp1d

class TrialAverageWindow(NewWindow):

    def __init__(self,
                 parent=None,
                 dataset=None,
                 dt_sampling=10, # ms
                 title='Trial-Averaging'):

        super(TrialAverageWindow, self).__init__(parent=parent,
                                                 title=title)

        self.dataset = dataset
        description, self.keys,\
            self.quantities = dataset_description(self.dataset)
        
        self.EPISODES = None
        
        mainLayout = QtWidgets.QHBoxLayout(self.cwidget)
        Layout1 = QtWidgets.QVBoxLayout()
        mainLayout.addLayout(Layout1)
        Layout2 = QtWidgets.QVBoxLayout()
        mainLayout.addLayout(Layout2)
        
        self.Layout11 = QtWidgets.QHBoxLayout()
        Layout1.addLayout(self.Layout11)

        # description
        self.notes = QtWidgets.QLabel(description, self)
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
        for q in self.quantities:
            self.qbox.addItem(q)
        self.qbox.activated.connect(self.update_quantity)
        self.Layout12.addWidget(self.qbox)
        self.Layout12.addWidget(QtWidgets.QLabel('Sub-Quantity', self))
        self.sqbox = QtWidgets.QComboBox(self)
        self.sqbox.activated.connect(self.update_subquantity)
        self.Layout12.addWidget(self.sqbox)
        self.Layout12.addWidget(QtWidgets.QLabel('Property', self))
        self.pbox = QtWidgets.QComboBox(self)
        self.Layout12.addWidget(self.pbox)

        self.Layout12.addWidget(QtWidgets.QLabel('', self))
        self.Layout12.addWidget(QtWidgets.QLabel(9*'-'+\
                                                 ' Display options '+\
                                                 9*'-', self))
        for key in self.keys:
            setattr(self, "c"+key, QtWidgets.QComboBox(self))
            self.Layout12.addWidget(getattr(self, "c"+key))
            for k in ['(merge)', '(color-code)', '(row)', '(column)']:
                getattr(self, "c"+key).addItem(key+\
                                    ((30-len(k)-len(key))*' ')+k)
        for i in range(4-len(self.keys)):
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

    def refresh(self):

        self.plots.clear()
        self.quantity = self.qbox.currentText()
        if (self.EPISODES is None) or\
           (self.quantity!=self.EPISODES['quantity'])or\
           (self.samplingBox.value()!=self.EPISODES['dt_sampling']):
            self.EPISODES = build_episodes(self.dataset,
                           quantity=self.quantity,
                           dt_sampling=self.samplingBox.value())

            
        COL_CONDS = self.build_column_conditions()
        ROW_CONDS = self.build_row_conditions()

        l = self.plots.addLayout(rowspan=len(ROW_CONDS),
                                 colspan=len(COL_CONDS),
                                 border=(0,0,0))
        l.setContentsMargins(2, 2, 2, 2)
        l.layout.setSpacing(1.)            

        AX = []
        for irow, row_cond in enumerate(ROW_CONDS):
            AX.append([])
            for icol, col_cond in enumerate(COL_CONDS):
                AX[irow].append(l.addPlot())
                cond = col_cond & row_cond
                AX[irow][icol].plot(self.EPISODES['t'],
                    np.array(self.EPISODES[self.quantity])[cond,:].mean(axis=0))
                if icol>0:
                    AX[irow][icol].hideAxis('left')
                    AX[irow][icol].setYLink(AX[irow][0])
            l.nextRow()
        for irow, row_cond in enumerate(ROW_CONDS):
            for icol, col_cond in enumerate(COL_CONDS):
                if irow<(len(ROW_CONDS)-1):
                    AX[irow][icol].hideAxis('bottom')
                    AX[irow][icol].setXLink(AX[-1][icol])


        
    def build_conditions(self, X, K):
        if len(K)>0:
            CONDS = []
            XK = np.meshgrid(*X)
            print(XK)
            for i in range(len(XK[0].flatten())): # looping over joint conditions
                cond = np.ones(len(self.dataset.metadata['time_start']), dtype=bool)
                for k, xk in zip(K, XK):
                    cond = cond & (self.dataset.metadata[k]==xk.flatten()[i])
                CONDS.append(cond)
            return CONDS
        else:
            return [np.ones(len(self.dataset.metadata['time_start']), dtype=bool)]
            
    
    def build_column_conditions(self):
        X, K = [], []
        for key in self.keys:
            if len(getattr(self, "c"+key).currentText().split('column'))>1:
                X.append(np.unique(self.dataset.metadata[key]))
                K.append(key)
        return self.build_conditions(X, K)

    
    def build_row_conditions(self):
        X, K = [], []
        for key in self.keys:
            if len(getattr(self, "c"+key).currentText().split('row'))>1:
                X.append(np.sort(np.unique(self.dataset.metadata[key])))
                K.append(key)
        return self.build_conditions(X, K)

    
    def update_quantity(self):
        self.sqbox.clear()
        self.pbox.clear()
        if self.qbox.currentText()=='CaImaging':
            self.sqbox.addItem('[sum]')
            for i in range(self.dataset.CaImaging.F.shape[0]):
                self.sqbox.addItem('ROI-%i' % (i+1))
            for k in ['Firing', 'F', 'Fneu', 'dF (F-0.7*Fneu)']:
                self.pbox.addItem(k)
    
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

def build_episodes(dataset,
                   quantity='Locomotion',
                   dt_sampling=1, # ms
                   interpolation='linear'):

    print('building episodes [...]')
    EPISODES = {'dt_sampling':dt_sampling,
                'quantity':quantity}
    

    # new sampling
    interstim = dataset.metadata['presentation-interstim-period']
    ipre = int(interstim/dt_sampling/1e-3/2.)
    idur = int(dataset.metadata['presentation-duration']/dt_sampling/1e-3)
    EPISODES['t'] = np.arange(-ipre+1, idur+2*ipre-1)*dt_sampling*1e-3
    EPISODES[quantity] = []
    if quantity=='Photodiode':
        tfull = dataset.Screen.photodiode.t
        valfull = dataset.Screen.photodiode.val
    elif quantity=='CaImaging':
        tfull = dataset.CaImaging.t
        valfull =dataset.CaImaging.dF.mean(axis=0)
        for n in range(dataset.CaImaging.Firing.shape[0]):
            for k in ['Firing', 'F', 'Fneu', 'dF']:
                EPISODES['%s-%i' % (k, n+1)] = []
    else:
        tfull = getattr(dataset, quantity).t
        valfull = getattr(dataset, quantity).val
    
    for tstart, tstop in zip(dataset.metadata['time_start_realigned'],\
                             dataset.metadata['time_stop_realigned']):

        cond = (tfull>=(tstart-interstim/2.)) & (tfull<(tstop+interstim))
        func = interp1d(tfull[cond]-tstart, valfull[cond],
                        kind=interpolation)
        EPISODES[quantity].append(func(EPISODES['t']))
        
    if quantity=='CaImaging':
        for tstart, tstop in zip(dataset.metadata['time_start_realigned'],\
                                 dataset.metadata['time_stop_realigned']):
            cond = (tfull>=(tstart-interstim/2.)) & (tfull<(tstop+interstim))
            for n in range(dataset.CaImaging.Firing.shape[0]):
                for k in ['Firing', 'F', 'Fneu', 'dF']:
                    func = interp1d(tfull[cond]-tstart, getattr(dataset.CaImaging, k)[n,cond],
                                    kind=interpolation)
                    EPISODES['%s-%i' % (k, n+1)].append(func(EPISODES['t']))
        
            
    print('[ok] episodes ready !')
    return EPISODES
    
if __name__=='__main__':

    folder = os.path.join(os.path.expanduser('~'),\
                          'DATA', '2020_11_04', '01-02-03')
    
    dataset = Dataset(folder,
                      with_CaImaging_stat=False,
                      modalities=['Screen', 'Locomotion', 'CaImaging'])
    
    from PyQt5 import QtGui, QtWidgets, QtCore
    app = QtWidgets.QApplication(sys.argv)
    build_dark_palette(app)
    window = TrialAverageWindow(app, dataset=dataset)
    sys.exit(app.exec_())
    
    # app = QtWidgets.QApplication(sys.argv)
    # main = run(app,raw_data_visualization=raw_data_visualization)
    # sys.exit(app.exec_())
    
    # print(print_summary(dataset))
    # build_episodes(dataset)
