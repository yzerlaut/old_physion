import sys, time, tempfile, os, pathlib, json, subprocess
import multiprocessing # for the camera streams !!
import numpy as np
from PyQt5 import QtGui, QtWidgets, QtCore

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import *

from visual_stim.psychopy_code.stimuli import build_stim
from visual_stim.default_params import SETUP

from analysis.guiparts import set_app_icon, build_dark_palette
try:
    from hardware_control.NIdaq.main import Acquisition
    from hardware_control.FLIRcamera.recording import launch_FaceCamera
    from hardware_control.LogitechWebcam.preview import launch_RigView
except ModuleNotFoundError:
    # just to be able to work on the UI without the modules
    pass

# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
## NASTY workaround to the error:
# ** OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized. **

CONFIG_LIST = ['                                (choose)',
               'NIdaq',
               'VisualStim',
               'VisualStim+NIdaq',
               'VisualStim+NIdaq+FaceCamera',
               'VisualStim+NIdaq+CaImaging',
               'VisualStim+NIdaq+FaceCamera+CaImaging',
               'FaceCamera',
               'FaceCamera+NIdaq',
               'FaceCamera+NIdaq+CaImaging',
               'CaImaging']

STEP_FOR_CA_IMAGING = {"channel":0, "onset": 0.1, "duration": .3, "value":5.0}

default_settings = {'NIdaq-acquisition-frequency':10000.,
                    'NIdaq-input-channels': 4,
                    'protocol_folder':os.path.join('exp', 'protocols'),
                    'root_datafolder':os.path.join(os.path.expanduser('~'), 'DATA'),
                    'config' : CONFIG_LIST[0],
                    'FaceCamera-frame-rate': 20}


class MainWindow(QtWidgets.QMainWindow):
    
    def __init__(self, parent=None,
                 button_length = 135):
        
        super(MainWindow, self).__init__()
        
        self.setWindowTitle('Experimental module -- Physiology of Visual Circuits')
        self.setGeometry(50, 50, 550, 500)

        self.metadata = default_settings # set a load/save interface
        self.protocol, self.protocol_folder = None, self.metadata['protocol_folder']
        self.config = None

        self.root_datafolder = self.metadata['root_datafolder']
        self.datafolder = None
            
        self.get_protocol_list()
        self.experiment = {} # storing the specifics of an experiment
        self.quit_event = multiprocessing.Event() # to control the RigView !
        self.run_event = multiprocessing.Event() # to turn on and off recordings execute through multiprocessing.Process
        # self.camready_event = multiprocessing.Event() # to turn on and off recordings execute through multiprocessing.Process

        self.stim, self.acq, self.init, self.setup, self.stop_flag = None, None, False, SETUP[0], False
        self.FaceCamera_process = None
        self.RigView_process = None
        self.params_window = None
        
        # buttons and functions
        LABELS = ["i) Initialize", "r) Run", "s) Stop", "q) Quit"]
        FUNCTIONS = [self.initialize, self.run, self.stop, self.quit]
        
        mainMenu = self.menuBar()
        self.fileMenu = mainMenu.addMenu('')

        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage('ready for select a protocol/')
        
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
            
        # config choice
        QtWidgets.QLabel("  Setup Config. :", self).move(30, 80)
        self.cbc = QtWidgets.QComboBox(self)
        self.cbc.addItems(CONFIG_LIST)
        self.cbc.setMinimumWidth(350)
        self.cbc.move(150, 80)

        # protocol choice
        QtWidgets.QLabel("Visual Protocol :", self).move(30, 120)
        self.cbp = QtWidgets.QComboBox(self)
        self.cbp.addItems(['None']+\
                          [f.replace('.json', '') for f in self.protocol_list])
        self.cbp.setMinimumWidth(350)
        self.cbp.move(150, 120)
        self.pbtn = QtWidgets.QPushButton('Set folder', self)
        self.pbtn.clicked.connect(self.set_protocol_folder)
        self.pbtn.move(470, 120)

        self.dfl = QtWidgets.QLabel('Data-Folder (root): "%s"' % str(self.root_datafolder), self)
        self.dfl.setMinimumWidth(300)
        self.dfl.move(30, 160)
        dfb = QtWidgets.QPushButton('Set folder', self)
        dfb.clicked.connect(self.choose_data_folder)
        dfb.move(370, 160)

        naf = QtWidgets.QLabel("NI-daq Acquisition Freq. (kHz): ", self)
        naf.setMinimumWidth(300)
        naf.move(50, 210)
        self.NIdaqFreq = QtWidgets.QDoubleSpinBox(self)
        self.NIdaqFreq.move(250, 210)
        self.NIdaqFreq.setValue(self.metadata['NIdaq-acquisition-frequency']/1e3)
        
        nrc = QtWidgets.QLabel("NI-daq recording channels (#): ", self)
        nrc.setMinimumWidth(300)
        nrc.move(50, 250)
        self.NIdaqNchannel = QtWidgets.QSpinBox(self)
        self.NIdaqNchannel.move(250, 250)
        self.NIdaqNchannel.setValue(self.metadata['NIdaq-input-channels'])
        
        ffr = QtWidgets.QLabel("FaceCamera frame rate (Hz): ", self)
        ffr.setMinimumWidth(300)
        ffr.move(50, 290)
        self.FaceCameraFreq = QtWidgets.QDoubleSpinBox(self)
        self.FaceCameraFreq.move(250, 290)
        self.FaceCameraFreq.setValue(self.metadata['FaceCamera-frame-rate'])
        
        LABELS = ["Launch RigView"]#, "v) View Data"]
        FUNCTIONS = [self.rigview]#, self.view_data]
        for func, label, shift, size in zip(FUNCTIONS, LABELS,\
                                            160*np.arange(len(LABELS)), [130, 130]):
            btn = QtWidgets.QPushButton(label, self)
            btn.clicked.connect(func)
            btn.setMinimumWidth(size)
            btn.move(shift+30, 350)
            action = QtWidgets.QAction(label, self)
            if len(label.split(')'))>0:
                action.setShortcut(label.split(')')[0])
                action.triggered.connect(func)
                self.fileMenu.addAction(action)

        self.show()
        
    def facecamera_init(self):
        if self.FaceCamera_process is None:
            self.FaceCamera_process = multiprocessing.Process(target=launch_FaceCamera,
                                        args=(self.run_event , self.quit_event,
                                          self.root_datafolder))
            self.FaceCamera_process.start()
            print('  starting FaceCamera stream [...] ')
            time.sleep(6)
            print('[ok] FaceCamera ready ! ')
        else:
            print('[ok] FaceCamera already initialized ')
            
        return True
            
    def choose_data_folder(self):
        fd = str(QtWidgets.QFileDialog.getExistingDirectory(self,
                                                            "Select Root Data Folder", self.root_datafolder))
        if os.path.isdir(fd):
            self.root_datafolder = fd
            set_data_folder(fd)
            self.dfl.setText('Data-Folder (root): "%s"' % str(self.datafolder))
        else:
            self.statusBar.showMessage('Invalid folder -> folder unchanged')

            
    def analyze_data(self):
        pass
    
    def view_data(self):
        pass

    def rigview(self):
        if self.RigView_process is not None:
            self.RigView_process.terminate()
        self.statusBar.showMessage('Initializing RigView stream [...]')
        self.RigView_process = multiprocessing.Process(target=launch_RigView,
                                                       args=(self.run_event , self.quit_event, self.datafolder))
        self.RigView_process.start()
        time.sleep(5)
        self.statusBar.showMessage('Setup ready')
        
    def initialize(self):

        i = self.cbc.currentIndex()
        if self.cbc.currentIndex()==0:
            self.statusBar.showMessage('/!\ Need to choose a configuration !')
        elif self.cbp.currentIndex()==0:
            self.statusBar.showMessage('/!\ Need to choose a protocol !')
        else:
            self.config = self.cbc.currentText()
            self.metadata['protocol'] = self.cbp.currentText()
            
            self.statusBar.showMessage('[...] preparing stimulation')

            self.filename = generate_filename_path(self.root_datafolder,
                                    filename='metadata', extension='.npy',
                                    with_FaceCamera_frames_folder=('FaceCamera' in self.config),
                                    with_screen_frames_folder=('VisualStim' in self.config))
            self.datafolder = os.path.dirname(self.filename)
            
            if self.metadata['protocol']!='None':
                with open(os.path.join(self.protocol_folder,
                                       self.metadata['protocol']+'.json'), 'r') as fp:
                    self.protocol = json.load(fp)
            else:
                    self.protocol = {}
                    
            # init facecamera
            if 'FaceCamera' in self.config:
                self.statusBar.showMessage('Initializing Camera stream [...]')
                self.facecamera_init()
                
            # init visual stimulation
            if 'VisualStim' in self.config:
                self.stim = build_stim(self.protocol)
                np.save(os.path.join(self.datafolder, 'visual-stim.npy'), self.stim.experiment)
                print('[ok] Visual-stimulation data saved as "%s"' % os.path.join(self.datafolder, 'visual-stim.npy'))
                max_time = self.stim.experiment['time_stop'][-1]+20
            else:
                max_time = 2*60*60 # 2 hours, should be stopped manually
                self.stim = None

            output_steps = []
            if 'CaImaging' in self.config:
                output_steps.append(STEP_FOR_CA_IMAGING)


            if 'NIdaq' in self.config:
                self.acq = Acquisition(dt=1./self.metadata['NIdaq-acquisition-frequency'],
                                       Nchannel_in=self.metadata['NIdaq-input-channels'],
                                       max_time=max_time,
                                       output_steps=output_steps,
                                       filename= self.filename.replace('metadata', 'NIdaq'))
            else:
                self.acq = None
            
            self.init = True
            
            self.save_experiment() # saving all metadata after full initialization

            self.statusBar.showMessage('stimulation ready !')
            
    def run(self):
        self.stop_flag=False
        self.run_event.set() # start the run flag for the facecamera
        
        if ((self.acq is None) and (self.stim is None)) or not self.init:
            self.statusBar.showMessage('Need to initialize the stimulation !')
        elif self.stim is None and self.acq is not None:
            self.acq.launch()
            self.statusBar.showMessage('NIdaq recording running [...]')
            self.init = False
        else:
            # Ni-Daq
            if self.acq is not None:
                self.acq.launch()
            self.statusBar.showMessage('stimulation & recording running [...]')
            # run visual stim
            if 'VisualStim' in self.config:
                self.stim.run(self)
            # stop and clean up things
            if 'FaceCamera' in self.config:
                self.run_event.clear() # this will close the camera process
            if 'VisualStim' in self.config:
                self.stim.close() # close the visual stim
            if self.acq is not None:
                self.acq.close()
            self.init = False
        if 'CaImaging' in self.config and not self.stop_flag:
            self.send_CaImaging_Stop_signal()
        print(100*'-', '\n', 50*'=')
    
    def stop(self):
        self.run_event.clear() # this will close the camera process
        self.stop_flag=True
        if self.acq is not None:
            self.acq.close()
        if self.stim is not None:
            self.stim.close()
            self.init = False
        if 'CaImaging' in self.config:
            self.send_CaImaging_Stop_signal()
        self.statusBar.showMessage('stimulation stopped !')
        print(100*'-', '\n', 50*'=')
        
    def send_CaImaging_Stop_signal(self):
        acq = Acquisition(dt=1e-3, # 1kHz
                          Nchannel_in=2, max_time=1.1,
                          buffer_time=0.1,
                          output_steps= [STEP_FOR_CA_IMAGING],
                          filename=None)
        acq.launch()
        time.sleep(1.1)
        acq.close()
    
    def quit(self):
        self.quit_event.set()
        if self.acq is not None:
            self.acq.close()
        if self.stim is not None:
            self.stim.quit()
        QtWidgets.QApplication.quit()

    def save_experiment(self):
        # SAVING THE METADATA FILES
        self.metadata = {**self.metadata, **self.protocol} # joining dictionaries
        np.save(os.path.join(self.datafolder, 'metadata.npy'), self.metadata)
        print('[ok] Metadata data saved as: %s ' % os.path.join(self.datafolder, 'metadata.npy'))
        self.statusBar.showMessage('Metadata saved as: "%s" ' % os.path.join(self.datafolder, 'metadata.npy'))

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
    build_dark_palette(app)
    set_app_icon(app)
    main = MainWindow(app)
    sys.exit(app.exec_())
