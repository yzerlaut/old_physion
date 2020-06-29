import sys, time, tempfile, os, pathlib, json, subprocess
import numpy as np
from PyQt5 import QtGui, QtWidgets, QtCore

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import day_folder, generate_filename_path, save_dict, load_dict
from assembling.analysis import quick_data_view, last_datafile

from visual_stim.psychopy_code.stimuli import build_stim
from visual_stim.default_params import SETUP

class MasterWindow(QtWidgets.QMainWindow):
    
    def __init__(self, app,
                 parent=None,
                 button_length = 100):
        
        super(MasterWindow, self).__init__(parent)
        
        self.protocol, self.protocol_folder = None, os.path.join('master', 'protocols')
        self.get_protocol_list()
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
        self.statusBar.showMessage('initialize an experiment // analyze data')
        
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

        # setup pick
        QtWidgets.QLabel("  /|=>  Setup <=|\\", self).move(30, 120)
        self.cbs = QtWidgets.QComboBox(self)
        self.cbs.addItems(SETUP)
        self.cbs.move(150, 120)

        # data pick
        QtWidgets.QLabel("   /|=>  Data <=|\\", self).move(30, 160)
        self.cbd = QtWidgets.QComboBox(self)
        self.cbd.addItems(['VisualStim+NIdaq', 'VisualStim+NIdaq+FaceCamera', 'VisualStim+NIdaq+FaceCamera+Imaging'])
        self.cbd.setMinimumWidth(200)
        self.cbd.move(150, 160)
        self.dbtn = QtWidgets.QPushButton('Set folder', self)
        self.dbtn.clicked.connect(self.set_data_folder)
        self.dbtn.move(370, 160)

        LABELS = ["v) View Data", " a) Analyze Data"]
        FUNCTIONS = [self.view_data, self.analyze_data, self.set_analysis_folder]
        for func, label, shift, size in zip(FUNCTIONS, LABELS,\
                                            150*np.arange(len(LABELS)), [180, 180]):
            btn = QtWidgets.QPushButton(label, self)
            btn.clicked.connect(func)
            btn.setMinimumWidth(size)
            btn.move(shift, 250)
            action = QtWidgets.QAction(label, self)
            if len(label.split(')'))>0:
                action.setShortcut(label.split(')')[0])
                action.triggered.connect(func)
                self.fileMenu.addAction(action)

        self.show()

    def set_data_folder(self):
        pass
    
    def set_analysis_folder(self):
        pass
    
    def analyze_data(self):
        pass
    
    def view_data(self):
        quick_data_view(last_datafile(self))
    
    def initialize(self):
        try:
            filename = os.path.join(self.protocol_folder, self.cbp.currentText()+'.json')
            with open(filename, 'r') as fp:
                self.protocol = json.load(fp)
            self.protocol['NIdaq-rec'] = True
            self.statusBar.showMessage('[...] preparing stimulation')
            self.stim = build_stim(self.protocol)
            self.statusBar.showMessage('stimulation ready !')
            self.init = True
        except FileNotFoundError:
            self.statusBar.showMessage('protocol file "%s" not found !' % filename)

    def run(self):
        self.stop_flag=False
        if (self.stim is None) or not self.init:
            self.statusBar.showMessage('Need to initialize the stimulation !')
        else:
            self.save_experiment()
            self.statusBar.showMessage('stimulation & recording running [...]')
            self.stim.run(self, with_NIdaq=True)
            self.stim.close()
            self.init = False
    
    def stop(self):
        self.stop_flag=True
        self.statusBar.showMessage('stimulation stopped !')
        if self.stim is not None:
            self.stim.close()
            self.init = False
    
    def quit(self):
        if self.stim is not None:
            self.stim.quit()
        sys.exit()

    def save_experiment(self):
        full_exp = dict(**self.protocol, **self.stim.experiment)
        self.filename = generate_filename_path(self.data_folder, filename='visual-stim', extension='.npz')
        save_dict(self.filename, full_exp)
        print('Stimulation data saved as: %s ' % self.filename)
        self.statusBar.showMessage('Stimulation data saved as: %s ' % self.filename)

    def get_protocol_list(self):
        files = os.listdir(self.protocol_folder)
        self.protocol_list = [f for f in files if f.endswith('.json')]
        
    def set_protocol_folder(self):
        self.protocol_folder = str(QtWidgets.QFileDialog.getExistingDirectory(self,
                                                                              "Select Protocol Folder"))
        self.get_protocol_list()
        self.cbp.addItems([f.replace('.json', '') for f in self.protocol_list])
        
    def save_protocol(self):
        if self.params_window is not None:
            self.protocol = extract_params_from_window(self)
            self.protocol['data-folder'] = self.data_folder
            self.protocol['protocol-folder'] = self.protocol_folder
            self.protocol['Setup'] = self.setup
            with open('protocol.json', 'w') as fp:
                json.dump(self.protocol, fp, indent=2)
                self.statusBar.showMessage('protocol saved as "protocol.json"')
        else:
            self.statusBar.showMessage('No protocol data available')
            
    def load_protocol(self):
        filename = 'protocol.json'
        try:
            with open(filename, 'r') as fp:
                self.protocol = json.load(fp)
            self.data_folder = self.protocol['data-folder']
            self.protocol_folder = self.protocol['protocol-folder']
            self.setup = self.protocol['Setup']
            # update main window
            s1, s2, s3 = self.protocol['Presentation'], self.protocol['Stimulus'], self.protocol['Setup']
            self.cbp.setCurrentIndex(np.argwhere(s1==np.array(list(['']+PRESENTATIONS)))[0][0])
            self.cbs.setCurrentIndex(np.argwhere(s2==np.array(list(['']+list(STIMULI.keys()))))[0][0])
            self.cbst.setCurrentIndex(np.argwhere(s3==np.array(SETUP))[0][0])
            self.statusBar.showMessage('successfully loaded "%s"' % filename)
            # draw params window
            self.params_window = draw_window(self, self.protocol)
            # self.params_window = draw_window(self, self.protocol)
            self.params_window.show()
        except FileNotFoundError:
            self.statusBar.showMessage('protocol file "%s" not found !' % filename)

    def set_folders(self):
        self.protocol_folder = str(QtWidgets.QFileDialog.getExistingDirectory(self,
                                                                              "Select Protocol Folder"))
        self.data_folder = str(QtWidgets.QFileDialog.getExistingDirectory(self,
                                                                          "Select Data Folder"))
        self.statusBar.showMessage('Protocol folder: "%s", Data folder "%s"' %\
                                   (self.protocol_folder, self.data_folder))

    def change_protocol(self):
        self.params_window = draw_window(self, None)
        self.params_window.show()
        
    def change_stimulus(self):
        self.params_window = draw_window(self, None)
        self.params_window.show()

    def change_setup(self):
        self.setup = self.cbst.currentText()
        
if __name__ == '__main__':
    
    app = QtWidgets.QApplication(sys.argv)
    main = MasterWindow(app)
    sys.exit(app.exec_())
