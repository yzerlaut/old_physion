import sys, time, tempfile, os, pathlib, json, subprocess
import numpy as np
from PyQt5 import QtGui, QtWidgets, QtCore

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import day_folder, generate_filename_path, save_dict, load_dict
from assembling.analysis import quick_data_view, analyze_data, last_datafile

from visual_stim.psychopy_code.stimuli import build_stim
from visual_stim.default_params import SETUP

from hardware_control.NIdaq.main import Acquisition

class MasterWindow(QtWidgets.QMainWindow):
    
    def __init__(self, app,
                 parent=None,
                 button_length = 100):
        
        super(MasterWindow, self).__init__(parent)
        
        self.protocol, self.protocol_folder = None, os.path.join('master', 'protocols')
        self.config, self.config_folder = None, os.path.join('master', 'configs')
        self.get_protocol_list()
        self.get_config_list()
        self.experiment = {} # storing the specifics of an experiment
        
        self.stim, self.init, self.setup, self.stop_flag = None, False, SETUP[0], False
        self.params_window = None
        self.data_folder = tempfile.gettempdir()
        
        self.setWindowTitle('Master Program -- Physiology of Visual Circuits')
        self.setGeometry(50, 50, 500, 300)

        # buttons and functions
        LABELS = ["i) Initialize", "r) Run", "s) Stop", "q) Quit"]
        FUNCTIONS = [self.initialize, self.run, self.stop, self.quit]
        
        mainMenu = self.menuBar()
        self.fileMenu = mainMenu.addMenu('')

        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage('ready for initialization/analysis')
        
        for func, label, shift in zip(FUNCTIONS, LABELS,\
                                      button_length*np.arange(len(LABELS))):
            btn = QtWidgets.QPushButton(label, self)
            btn.clicked.connect(func)
            btn.setMinimumWidth(button_length)
            btn.move(shift, 20)
            action = QtWidgets.QAction(label, self)
            action.setShortcut(label.split(')')[0])
            action.triggered.connect(func)
            self.fileMenu.addAction(action)
            
        # protocol choice
        QtWidgets.QLabel(" /|=>  Protocol <=|\\", self).move(30, 80)
        self.cbp = QtWidgets.QComboBox(self)
        self.cbp.addItems([f.replace('.json', '') for f in self.protocol_list])
        self.cbp.setMinimumWidth(200)
        self.cbp.move(150, 80)
        self.pbtn = QtWidgets.QPushButton('Set folder', self)
        self.pbtn.clicked.connect(self.set_protocol_folder)
        self.pbtn.move(370, 80)

        # config choice
        QtWidgets.QLabel("   /|=>  Config <=|\\", self).move(30, 120)
        self.cbc = QtWidgets.QComboBox(self)
        self.cbc.addItems([f.replace('.json', '') for f in self.config_list])
        self.cbc.setMinimumWidth(200)
        self.cbc.move(150, 120)
        self.dbtn = QtWidgets.QPushButton('Set folder', self)
        self.dbtn.clicked.connect(self.set_config_folder)
        self.dbtn.move(370, 120)

        LABELS = ["v) View Data", " a) Analyze Data"]
        FUNCTIONS = [self.view_data, self.analyze_data]
        for func, label, shift, size in zip(FUNCTIONS, LABELS,\
                                            160*np.arange(len(LABELS)), [130, 130]):
            btn = QtWidgets.QPushButton(label, self)
            btn.clicked.connect(func)
            btn.setMinimumWidth(size)
            btn.move(shift, 180)
            action = QtWidgets.QAction(label, self)
            if len(label.split(')'))>0:
                action.setShortcut(label.split(')')[0])
                action.triggered.connect(func)
                self.fileMenu.addAction(action)

        self.show()

    def analyze_data(self):
        analyze_data(last_datafile(self.data_folder))
    
    def view_data(self):
        quick_data_view(last_datafile(self.data_folder))
    
    def initialize(self):
        try:
            filename = os.path.join(self.protocol_folder, self.cbp.currentText()+'.json')
            with open(filename, 'r') as fp:
                self.protocol = json.load(fp)
            filename = os.path.join(self.config_folder, self.cbc.currentText()+'.json')
            with open(filename, 'r') as fp:
                self.config = json.load(fp)
            self.statusBar.showMessage('[...] preparing stimulation')
            self.stim = build_stim(self.protocol)
            self.statusBar.showMessage('stimulation ready !')
            self.filename = generate_filename_path(self.config['data-folder'],
                                                   filename='visual-stim', extension='.npz')
            self.acq = Acquisition(dt=1./self.config['NIdaq-acquisition-frequency'],
                                   Nchannel_in=self.config['NIdaq-input-channels'],
                                   max_time=self.stim.experiment['time_stop'][-1]+20,
                                   filename= self.filename.replace('visual-stim.npz', 'NIdaq.npy'))
            self.init = True
        except FileNotFoundError:
            self.statusBar.showMessage('protocol file "%s" not found !' % filename)

    def run(self):
        self.stop_flag=False
        if (self.stim is None) or not self.init:
            self.statusBar.showMessage('Need to initialize the stimulation !')
        else:
            self.save_experiment()
            self.acq.launch()
            self.statusBar.showMessage('stimulation & recording running [...]')
            self.stim.run(self)
            self.stim.close()
            self.acq.close()
            self.init = False
    
    def stop(self):
        self.stop_flag=True
        self.acq.close()
        self.statusBar.showMessage('stimulation stopped !')
        if self.stim is not None:
            self.stim.close()
            self.init = False
    
    def quit(self):
        if self.stim is not None:
            self.acq.close()
            self.stim.quit()
        sys.exit()

    def save_experiment(self):
        full_exp = dict(**self.protocol, **self.stim.experiment)
        save_dict(self.filename, full_exp)
        print('Stimulation data saved as: %s ' % self.filename)
        self.statusBar.showMessage('Stimulation data saved as: %s ' % self.filename)

    def get_protocol_list(self):
        files = os.listdir(self.protocol_folder)
        self.protocol_list = [f for f in files if f.endswith('.json')]
        
    def get_config_list(self):
        files = os.listdir(self.config_folder)
        self.config_list = [f for f in files if f.endswith('.json')]
        
    def set_protocol_folder(self):
        fd = str(QtWidgets.QFileDialog.getExistingDirectory(self,
                                                            "Select Protocol Folder", self.protocol_folder))
        if fd!='':
            fd = self.protocol_folder
            self.get_protocol_list()
            self.cbp.addItems([f.replace('.json', '') for f in self.protocol_list])

    def set_config_folder(self):
        fd = str(QtWidgets.QFileDialog.getExistingDirectory(self,
                                                            "Select Config Folder", self.config_folder))
        if fd!='':
            fd = self.config_folder
            self.get_config_list()
            self.cbp.addItems([f.replace('.json', '') for f in self.config_list])
        
if __name__ == '__main__':
    
    app = QtWidgets.QApplication(sys.argv)
    main = MasterWindow(app)
    sys.exit(app.exec_())
