import sys, time, tempfile, os, pathlib, json, subprocess
import multiprocessing # for the camera streams !!
import numpy as np
from PyQt5 import QtGui, QtWidgets, QtCore

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import day_folder, generate_filename_path, save_dict, load_dict
from assembling.analysis import quick_data_view, analyze_data, last_datafile

from visual_stim.psychopy_code.stimuli import build_stim
from visual_stim.default_params import SETUP

from hardware_control.NIdaq.main import Acquisition
from hardware_control.FLIRcamera.recording import launch_FaceCamera
from hardware_control.LogitechWebcam.preview import launch_RigView

# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
## NASTY workaround to the error:
# ** OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized. **

DFFN = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'master', 'data-folder.json') # DATA-FOLDER-FILENAME

class MasterWindow(QtWidgets.QMainWindow):
    
    def __init__(self, app,
                 parent=None,
                 button_length = 100):
        
        super(MasterWindow, self).__init__(parent)
        
        self.protocol, self.protocol_folder = None, os.path.join('master', 'protocols')
        self.config, self.config_folder = None, os.path.join('master', 'configs')

        # data folder
        if not os.path.isfile(DFFN):
            with open(DFFN, 'w') as fp:
                json.dump({"folder":str(tempfile.gettempdir())}, fp)
        with open(DFFN, 'r') as fp:
            self.data_folder = json.load(fp)['folder']
            if not os.path.isdir(self.data_folder): # then temp folder
                with open(DFFN, 'w') as fp:
                    json.dump({"folder":str(tempfile.gettempdir())}, fp)
                self.data_folder = tempfile.gettempdir()
            
        self.get_protocol_list()
        self.get_config_list()
        self.experiment = {} # storing the specifics of an experiment
        self.quit_event = multiprocessing.Event() # to control the RigView !
        self.run_event = multiprocessing.Event() # to turn on and off recordings execute through multiprocessing.Process
        # self.camready_event = multiprocessing.Event() # to turn on and off recordings execute through multiprocessing.Process

        self.stim, self.init, self.setup, self.stop_flag = None, False, SETUP[0], False
        self.FaceCamera_process = None
        self.RigView_process = None
        self.params_window = None
        
        self.setWindowTitle('Master Program -- Physiology of Visual Circuits')
        self.setGeometry(50, 50, 500, 260)

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

        self.dfl = QtWidgets.QLabel('Data-Folder (root): "%s"' % str(self.data_folder), self)
        self.dfl.setMinimumWidth(300)
        self.dfl.move(30, 160)
        dfb = QtWidgets.QPushButton('Set folder', self)
        dfb.clicked.connect(self.choose_data_folder)
        dfb.move(350, 160)
        
        LABELS = ["Launch RigView", "v) View Data"]
        FUNCTIONS = [self.rigview, self.view_data]
        for func, label, shift, size in zip(FUNCTIONS, LABELS,\
                                            160*np.arange(len(LABELS)), [130, 130]):
            btn = QtWidgets.QPushButton(label, self)
            btn.clicked.connect(func)
            btn.setMinimumWidth(size)
            btn.move(shift+30, 200)
            action = QtWidgets.QAction(label, self)
            if len(label.split(')'))>0:
                action.setShortcut(label.split(')')[0])
                action.triggered.connect(func)
                self.fileMenu.addAction(action)

        self.show()
        self.facecamera_init()
        
    def facecamera_init(self):
        if self.FaceCamera_process is not None:
            self.FaceCamera_process.terminate()
        self.FaceCamera_process = multiprocessing.Process(target=launch_FaceCamera,
                                                          args=(self.run_event , self.quit_event, self.data_folder))
        self.FaceCamera_process.start()
        self.statusBar.showMessage('Initializing Camera streams [...]')
        time.sleep(6)
        self.statusBar.showMessage('Setup ready !')
            
    def choose_data_folder(self):
        fd = str(QtWidgets.QFileDialog.getExistingDirectory(self,
                                                            "Select Root Data Folder", self.data_folder))
        if os.path.isdir(fd):
            self.data_folder = fd
            with open(DFFN, 'w') as fp:
                json.dump({"folder":self.data_folder}, fp)
            self.dfl.setText('Data-Folder (root): "%s"' % str(self.data_folder))
        else:
            self.statusBar.showMessage('Invalid folder -> folder unchanged')

            
    def analyze_data(self):
        self.statusBar.showMessage('Analyzing last recording [...]')
        data, fig1 = quick_data_view(last_datafile(self.data_folder), realign=True)
        _, fig2 = analyze_data(data=data)
        fig1.show()
        fig2.show()
    
    def view_data(self):
        _, fig = quick_data_view(last_datafile(self.data_folder))
        fig.show()

    def rigview(self):
        if self.RigView_process is not None:
            self.RigView_process.terminate()
        self.statusBar.showMessage('Initializing RigView stream [...]')
        self.RigView_process = multiprocessing.Process(target=launch_RigView,
                                                       args=(self.run_event , self.quit_event, self.data_folder))
        self.RigView_process.start()
        time.sleep(5)
        self.statusBar.showMessage('Setup ready')
        
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
            self.filename = generate_filename_path(self.data_folder,
                                                   filename='visual-stim', extension='.npz')
            output_steps, istep = [], 1
            while 'NIdaq-output-step-%i'%istep in self.config:
                print(self.config['NIdaq-output-step-%i'%istep])
                output_steps.append(self.config['NIdaq-output-step-%i'%istep])
                istep+=1
            self.acq = Acquisition(dt=1./self.config['NIdaq-acquisition-frequency'],
                                   Nchannel_in=self.config['NIdaq-input-channels'],
                                   max_time=self.stim.experiment['time_stop'][-1]+20,
                                   output_steps=output_steps,
                                   filename= self.filename.replace('visual-stim.npz', 'NIdaq.npy'))
            self.init = True
        except FileNotFoundError:
            self.statusBar.showMessage('protocol file "%s" not found !' % filename)

    def run(self):
        self.stop_flag=False
        self.run_event.set() # start the run flag for the facecamera
        
        if (self.stim is None) or not self.init:
            self.statusBar.showMessage('Need to initialize the stimulation !')
        else:
            self.save_experiment()
            # Ni-Daq
            self.acq.launch()
            self.statusBar.showMessage('stimulation & recording running [...]')
            # run
            self.stim.run(self)
            # stop and clean up things
            self.run_event.clear() # this will close the camera process
            self.stim.close()
            self.acq.close()
            self.init = False
    
    def stop(self):
        self.run_event.clear() # this will close the camera process
        self.stop_flag=True
        self.acq.close()
        self.statusBar.showMessage('stimulation stopped !')
        if self.stim is not None:
            self.stim.close()
            self.init = False
    
    def quit(self):
        self.quit_event.set()
        if self.stim is not None:
            self.acq.close()
            self.stim.quit()
        sys.exit()

    def save_experiment(self):
        for key in self.config:
            self.protocol[key] = self.config[key] # "config" overrides "protocol"
        full_exp = dict(**self.protocol, **self.stim.experiment)
        save_dict(self.filename, full_exp)
        print('Stimulation & acquisition data saved as: %s ' % self.filename)
        self.statusBar.showMessage('Stimulation & acquisition data saved as: %s ' % self.filename)

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
