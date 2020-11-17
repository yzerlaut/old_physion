from PyQt5 import QtGui, QtWidgets, QtCore
import sys, time, tempfile, os, pathlib, json, subprocess
import multiprocessing # for the camera streams !!
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import *

if not sys.argv[-1]=='no-stim':
    from visual_stim.psychopy_code.stimuli import build_stim
    from visual_stim.default_params import SETUP
else:
    SETUP = [None]

from misc.style import set_app_icon, set_dark_style
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

STEP_FOR_CA_IMAGING = {"channel":0, "onset": 0.1, "duration": .3, "value":5.0}
DT_frame, max_recording_time = 0.0333, 500 # seconds
STEPS_FOR_CAMERA_FRAME_TRIGGER = [{"channel":1, "onset": t+0.005, "duration": .01, "value":3.3}\
                                  for t in np.arange(int(max_recording_time/DT_frame))*DT_frame]

default_settings = {'NIdaq-acquisition-frequency':2000.,
                    'NIdaq-analog-input-channels': 1,
                    'NIdaq-digital-input-channels': 3,
                    'protocol_folder':os.path.join('exp', 'protocols'),
                    'root_datafolder':os.path.join(os.path.expanduser('~'), 'DATA'),
                    # 'config' : CONFIG_LIST[0],
                    'FaceCamera-frame-rate': 30}

class MainWindow(QtWidgets.QMainWindow):
    
    def __init__(self, app, args=None):
        """
        """
        super(MainWindow, self).__init__()
        
        self.setWindowTitle('Experimental module -- Physiology of Visual Circuits')
        self.setGeometry(50, 50, 550, 370)

        self.metadata = default_settings # set a load/save interface
        self.protocol, self.protocol_folder = None,\
            self.metadata['protocol_folder']

        if args is not None:
            self.root_datafolder = args.root_datafolder
            self.metadata['root_datafolder'] = args.root_datafolder
<<<<<<< HEAD
        
=======
        else:
            self.root_datafolder = os.path.join(os.path.expanduser('~'), 'DATA')
>>>>>>> 787f057010ac86ddda7dab16551dd3faf3c5a86f
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
        
        rml = QtWidgets.QLabel('   '+'-'*40+" Recording modalities "+'-'*40, self)
        rml.move(30, 5)
        rml.setMinimumWidth(500)
        self.VisualStimButton = QtWidgets.QPushButton("Visual-Stim", self)
        self.VisualStimButton.move(30, 40)
        self.LocomotionButton = QtWidgets.QPushButton("Locomotion", self)
        self.LocomotionButton.move(130, 40)
        self.ElectrophyButton = QtWidgets.QPushButton("Electrophy", self)
        self.ElectrophyButton.move(230, 40)
        self.FaceCameraButton = QtWidgets.QPushButton("FaceCamera", self)
        self.FaceCameraButton.move(330, 40)
        self.CaImagingButton = QtWidgets.QPushButton("CaImaging", self)
        self.CaImagingButton.move(430, 40)
        for button in [self.VisualStimButton, self.LocomotionButton, self.ElectrophyButton, self.FaceCameraButton, self.CaImagingButton]:
            button.setCheckable(True)
        for button in [self.LocomotionButton, self.FaceCameraButton, self.CaImagingButton]:
            button.setChecked(True)

        # protocol choice
        QtWidgets.QLabel("Visual Protocol :", self).move(30, 90)
        self.cbp = QtWidgets.QComboBox(self)
        self.cbp.addItems(['None']+\
                          [f.replace('.json', '') for f in self.protocol_list])
        self.cbp.setMinimumWidth(350)
        self.cbp.move(150, 90)
        
        # buttons and functions
        LABELS = ["i) Initialize", "r) Run", "s) Stop", "q) Quit"]
        FUNCTIONS = [self.initialize, self.run, self.stop, self.quit]
        
        mainMenu = self.menuBar()
        self.fileMenu = mainMenu.addMenu('')

        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage('ready to select a protocol/config')

        for func, label, shift in zip(FUNCTIONS, LABELS,\
                                      100*np.arange(len(LABELS))):
            btn = QtWidgets.QPushButton(label, self)
            btn.clicked.connect(func)
            btn.setMinimumWidth(100)
            btn.move(50+shift, 140)
            action = QtWidgets.QAction(label, self)
            action.setShortcut(label.split(')')[0])
            action.triggered.connect(func)
            self.fileMenu.addAction(action)
            
        # self.dfl = QtWidgets.QLabel('Data-Folder (root): "%s"' % str(self.root_datafolder), self)
        # self.dfl.setMinimumWidth(300)
        # self.dfl.move(30, 160)
        # dfb = QtWidgets.QPushButton('Set folder', self)
        # dfb.clicked.connect(self.choose_data_folder)
        # dfb.move(370, 160)

        # naf = QtWidgets.QLabel("NI-daq Acquisition Freq. (kHz): ", self)
        # naf.setMinimumWidth(300)
        # naf.move(50, 210)
        # self.NIdaqFreq = QtWidgets.QDoubleSpinBox(self)
        # self.NIdaqFreq.move(300,210)
        # self.NIdaqFreq.setValue(self.metadata['NIdaq-acquisition-frequency']/1e3)
        
        # narc = QtWidgets.QLabel("NI-daq analog recording channels (#): ", self)
        # narc.setMinimumWidth(300)
        # narc.move(50, 250)
        # self.NIdaqNchannel = QtWidgets.QSpinBox(self)
        # self.NIdaqNchannel.move(300,250)
        # self.NIdaqNchannel.setValue(self.metadata['NIdaq-analog-input-channels'])

        # ndrc = QtWidgets.QLabel("NI-daq digital recording channels (#): ", self)
        # ndrc.setMinimumWidth(300)
        # ndrc.move(50, 290)
        # self.NIdaqNchannel = QtWidgets.QSpinBox(self)
        # self.NIdaqNchannel.move(300,290)
        # self.NIdaqNchannel.setValue(self.metadata['NIdaq-digital-input-channels'])
        
        # ffr = QtWidgets.QLabel("FaceCamera frame rate (Hz): ", self)
        # ffr.setMinimumWidth(300)
        # ffr.move(50, 330)
        # self.FaceCameraFreq = QtWidgets.QDoubleSpinBox(self)
        # self.FaceCameraFreq.move(300,330)
        # self.FaceCameraFreq.setValue(self.metadata['FaceCamera-frame-rate'])

        QtWidgets.QLabel("Mouse ID: ", self).move(40, 210)
        self.qmID = QtWidgets.QComboBox(self)
        self.qmID.addItems(['1'])
        self.qmID.setMaximumWidth(70)
        self.qmID.move(140, 210)
        self.addID = QtWidgets.QPushButton('Add new mouse', self)
        self.addID.move(300, 210)
        self.addID.setMinimumWidth(120)
        
        QtWidgets.QLabel("Notes: ", self).move(60, 260)
        self.qmNotes = QtWidgets.QTextEdit('...\n\n\n', self)
        self.qmNotes.move(130, 260)
        self.qmNotes.setMinimumWidth(250)
        self.qmNotes.setMinimumHeight(70)
        
        # ms = QtWidgets.QLabel("Mouse sex: ", self)
        # ms.move(100, 420)
        # self.qms = QtWidgets.QComboBox(self)
        # self.qms.addItems(['N/A', 'Female', 'Male'])
        # self.qms.move(250, 420)
        # mg = QtWidgets.QLabel("Mouse genotype: ", self)
        # mg.move(100, 460)
        # self.qmg = QtWidgets.QLineEdit('wild type', self)
        # self.qmg.move(250, 460)
        # for m in [mID, ms, mg]:
        #     m.setMinimumWidth(300)
        
        # self.FaceCameraFreq = QtWidgets.QDoubleSpinBox(self)
        # self.FaceCameraFreq.move(250, 380)
        # self.FaceCameraFreq.setValue(self.metadata['FaceCamera-frame-rate'])

        # LABELS = ["Set protocol folder"]
        # FUNCTIONS = [self.set_protocol_folder]
        # for func, label, shift, size in zip(FUNCTIONS, LABELS,\
        #                                     160*np.arange(len(LABELS)), [130, 130]):
        #     btn = QtWidgets.QPushButton(label, self)
        #     btn.clicked.connect(func)
        #     btn.setMinimumWidth(size)
        #     btn.move(shift+30, 350)
        #     action = QtWidgets.QAction(label, self)
        #     if len(label.split(')'))>0:
        #         action.setShortcut(label.split(')')[0])
        #         action.triggered.connect(func)
        #         self.fileMenu.addAction(action)

        self.show()
        
    def facecamera_init(self):
        # if self.FaceCamera_process is None:
        #     self.FaceCamera_process = multiprocessing.Process(target=launch_FaceCamera,
        #                                 args=(self.run_event , self.quit_event,
        #                                       self.root_datafolder,
        #                             {'frame_rate':default_settings['FaceCamera-frame-rate']}))
        #     self.FaceCamera_process.start()
        #     print('  starting FaceCamera stream [...] ')
        #     time.sleep(6)
        #     print('[ok] FaceCamera ready ! ')
        # else:
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

        # Setup configuration
        for modality, button in zip(['VisualStim', 'Locomotion', 'Electrophy', 'FaceCamera', 'CaImaging'],
                                    [self.VisualStimButton, self.LocomotionButton, self.ElectrophyButton, self.FaceCameraButton, self.CaImagingButton]):
            self.metadata[modality] = bool(button.isChecked())
        # Protocol
        self.metadata['protocol'] = self.cbp.currentText()
            

        if self.cbp.currentText()=='None':
            self.statusBar.showMessage('[...] initializing acquisition')
        else:
            self.statusBar.showMessage('[...] initializing acquisition & stimulation')
            

        self.filename = generate_filename_path(self.root_datafolder,
                                               filename='metadata', extension='.npy',
                                               with_FaceCamera_frames_folder=self.metadata['FaceCamera'],
                                               with_screen_frames_folder=self.metadata['VisualStim'])
        self.datafolder = os.path.dirname(self.filename)
            
        if self.metadata['protocol']!='None':
            with open(os.path.join(self.protocol_folder,
                                   self.metadata['protocol']+'.json'), 'r') as fp:
                self.protocol = json.load(fp)
        else:
                self.protocol = {}
                    
        # init facecamera
        # if self.metadata['FaceCamera']:
        #     self.statusBar.showMessage('Initializing Camera stream [...]')
        #     self.facecamera_init()
                
        # init visual stimulation
        if self.metadata['VisualStim'] and len(self.protocol.keys())>0:
            self.stim = build_stim(self.protocol)
            np.save(os.path.join(self.datafolder, 'visual-stim.npy'), self.stim.experiment)
            print('[ok] Visual-stimulation data saved as "%s"' % os.path.join(self.datafolder, 'visual-stim.npy'))
            if 'time_stop' in self.stim.experiment:
                max_time = self.stim.experiment['time_stop'][-1]+20
            elif 'refresh_times' in self.stim.experiment:
                max_time = self.stim.experiment['refresh_times'][-1]+20
            else:
                max_time = 1*60*60 # 1 hour, should be stopped manually
        else:
            max_time = 1*60*60 # 1 hour, should be stopped manually
            self.stim = None

        output_steps = []
        if self.metadata['CaImaging']:
            output_steps.append(STEP_FOR_CA_IMAGING)
        if self.metadata['FaceCamera']:
            if not self.metadata['CaImaging']:
                output_steps.append(STEP_FOR_CA_IMAGING) # we add anyway the step for Ca Imaging to create the output analog
            output_steps = output_steps+STEPS_FOR_CAMERA_FRAME_TRIGGER

        # --------------- #
        ### NI daq init ###
        # --------------- #
        if self.metadata['VisualStim']:
            Nchannel_analog_in = 1
        else:
            Nchannel_analog_in = 0
        if self.metadata['Electrophy']:
            Nchannel_analog_in = 2
        if self.metadata['Locomotion']:
            Nchannel_digital_in = 2
        else:
            Nchannel_digital_in = 0
        try:
            self.acq = Acquisition(dt=1./self.metadata['NIdaq-acquisition-frequency'],
                                   Nchannel_analog_in=Nchannel_analog_in,
                                   Nchannel_digital_in=Nchannel_digital_in,
                                   max_time=max_time,
                                   output_steps=output_steps,
                                   filename= self.filename.replace('metadata', 'NIdaq'))
        except BaseException as e:
            print(e)
            print(' /!\ PB WITH NI-DAQ /!\ ')
            self.acq = None

        self.init = True
        self.save_experiment() # saving all metadata after full initialization
        
        if self.cbp.currentText()=='None':
            self.statusBar.showMessage('Acquisition ready !')
        else:
            self.statusBar.showMessage('Acquisition & Stimulation ready !')


        
    def run(self):
        self.stop_flag=False
        self.run_event.set() # start the run flag for the facecamera
        
        if ((self.acq is None) and (self.stim is None)) or not self.init:
            self.statusBar.showMessage('Need to initialize the stimulation !')
        elif self.stim is None and self.acq is not None:
            self.acq.launch()
            self.statusBar.showMessage('Acquisition running [...]')
        else:
            self.statusBar.showMessage('Stimulation & Acquisition running [...]')
            # Ni-Daq
            if self.acq is not None:
                self.acq.launch()
            # run visual stim
            if self.metadata['VisualStim']:
                self.stim.run(self)
            # ========================
            # ---- HERE IT RUNS [...]
            # ========================
            # stop and clean up things
            if self.metadata['FaceCamera']:
                self.run_event.clear() # this will close the camera process
            # close visual stim
            if self.metadata['VisualStim']:
                self.stim.close() # close the visual stim
            if self.acq is not None:
                self.acq.close()
            if self.metadata['CaImaging'] and not self.stop_flag: # outside the pure acquisition case
                self.send_CaImaging_Stop_signal()
                
        self.init = False
        print(100*'-', '\n', 50*'=')
        
    
    def stop(self):
        self.run_event.clear() # this will close the camera process
        self.stop_flag=True
        if self.acq is not None:
            self.acq.close()
        if self.stim is not None:
            self.stim.close()
            self.init = False
        if self.metadata['CaImaging']:
            self.send_CaImaging_Stop_signal()
        self.statusBar.showMessage('stimulation stopped !')
        print(100*'-', '\n', 50*'=')
        
    def send_CaImaging_Stop_signal(self):
        self.statusBar.showMessage('sending stop signal for 2-Photon acq.')
        acq = Acquisition(dt=1e-3, # 1kHz
                          Nchannel_analog_in=1, Nchannel_digital_in=0,
                          max_time=1.1,
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
        files = sorted(os.listdir(self.protocol_folder))
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
        
def run(app, args=None):
    print(args)
    return MainWindow(app, args)
    
if __name__=='__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = run(app)
    sys.exit(app.exec_())
