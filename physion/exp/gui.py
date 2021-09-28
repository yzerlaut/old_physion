from PyQt5 import QtGui, QtWidgets, QtCore
import sys, time, tempfile, os, pathlib, json, subprocess
import multiprocessing # for the camera streams !!
from ctypes import c_char_p 
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import *

if not sys.argv[-1]=='no-stim':
    from visual_stim.psychopy_code.stimuli import build_stim
    from visual_stim.screens import SCREENS
else:
    SCREENS = []
    
from misc.style import set_app_icon, set_dark_style
try:
    from hardware_control.NIdaq.main import Acquisition
    from hardware_control.FLIRcamera.recording import launch_FaceCamera
    from hardware_control.LogitechWebcam.preview import launch_RigView
except ModuleNotFoundError:
    # just to be able to work on the UI without the modules
    print('The hardware control modules were not found...')

# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
## NASTY workaround to the error:
# ** OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized. **

base_path = str(pathlib.Path(__file__).resolve().parents[0])
settings_filename = os.path.join(base_path, 'settings.npy')

class MainWindow(QtWidgets.QMainWindow):
    
    def __init__(self, app, args=None):
        """
        """
        super(MainWindow, self).__init__()
        
        self.setWindowTitle('Experimental module')
        self.setGeometry(400, 50, 550, 400)

        ##########################################################
        ######## Multiprocessing quantities
        ##########################################################
        self.run_event = multiprocessing.Event() # to turn on and off recordings execute through multiprocessing.Process
        self.run_event.clear()
        self.closeFaceCamera_event = multiprocessing.Event()
        self.closeFaceCamera_event.clear()
        self.quit_event = multiprocessing.Event()
        self.quit_event.clear()
        self.manager = multiprocessing.Manager() # Useful to share a string across processes :
        self.datafolder = self.manager.Value(c_char_p, str(os.path.join(os.path.expanduser('~'), 'DATA', 'trash')))
        
        ##########################################################
        ######## class values
        ##########################################################
        self.stim, self.acq, self.init, self.screen, self.stop_flag = None, None, False, None, False
        self.FaceCamera_process = None
        self.RigView_process = None
        self.params_window = None

        ##########################################################
        ####### GUI settings
        ##########################################################
        rml = QtWidgets.QLabel('   '+'-'*40+" Recording modalities "+'-'*40, self)
        rml.move(30, 5)
        rml.setMinimumWidth(500)
        self.VisualStimButton = QtWidgets.QPushButton("Visual-Stim", self)
        self.VisualStimButton.move(30, 40)
        self.LocomotionButton = QtWidgets.QPushButton("Locomotion", self)
        self.LocomotionButton.move(130, 40)
        self.LFPButton = QtWidgets.QPushButton("LFP", self)
        self.LFPButton.move(230, 40)
        self.LFPButton.setFixedWidth(50)
        self.VmButton = QtWidgets.QPushButton("Vm", self)
        self.VmButton.move(280, 40)
        self.VmButton.setFixedWidth(50)
        self.FaceCameraButton = QtWidgets.QPushButton("FaceCamera", self)
        self.FaceCameraButton.clicked.connect(self.toggle_FaceCamera_process)
        self.FaceCameraButton.move(330, 40)
        self.CaImagingButton = QtWidgets.QPushButton("CaImaging", self)
        self.CaImagingButton.move(430, 40)
        for button in [self.VisualStimButton, self.LocomotionButton, self.LFPButton, self.VmButton,
                       self.FaceCameraButton, self.CaImagingButton]:
            button.setCheckable(True)
        for button in [self.VisualStimButton, self.LocomotionButton]:
            button.setChecked(True)
            
        # screen choice
        QtWidgets.QLabel(" Screen :", self).move(250, 90)
        self.cbsc = QtWidgets.QComboBox(self)
        self.cbsc.setMinimumWidth(200)
        self.cbsc.move(320, 90)
        self.cbsc.activated.connect(self.update_screen)
        self.cbsc.addItems(SCREENS.keys())
        
        # config choice
        QtWidgets.QLabel("  => Config :", self).move(160, 125)
        self.cbc = QtWidgets.QComboBox(self)
        self.cbc.setMinimumWidth(270)
        self.cbc.move(250, 125)
        self.cbc.activated.connect(self.update_config)

        # subject choice
        QtWidgets.QLabel("-> Subject :", self).move(100, 160)
        self.cbs = QtWidgets.QComboBox(self)
        self.cbs.setMinimumWidth(340)
        self.cbs.move(180, 160)
        self.cbs.activated.connect(self.update_subject)
        
        # protocol choice
        QtWidgets.QLabel(" Visual Protocol :", self).move(20, 195)
        self.cbp = QtWidgets.QComboBox(self)
        self.cbp.setMinimumWidth(390)
        self.cbp.move(130, 195)
        self.cbp.activated.connect(self.update_protocol)
        
        # buttons and functions
        LABELS = ["i) Initialize", "r) Run", "s) Stop", "q) Quit"]
        FUNCTIONS = [self.initialize, self.run, self.stop, self.quit]
        
        mainMenu = self.menuBar()
        self.fileMenu = mainMenu.addMenu('')

        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage('ready to select a protocol/config')
        for func, label, shift in zip(FUNCTIONS, LABELS,\
                                      110*np.arange(len(LABELS))):
            btn = QtWidgets.QPushButton(label, self)
            btn.clicked.connect(func)
            btn.setMinimumWidth(110)
            btn.move(50+shift, 250)
            action = QtWidgets.QAction(label, self)
            action.setShortcut(label.split(')')[0])
            action.triggered.connect(func)
            self.fileMenu.addAction(action)
            
        QtWidgets.QLabel("Notes: ", self).move(40, 300)
        self.qmNotes = QtWidgets.QTextEdit('', self)
        self.qmNotes.move(90, 300)
        self.qmNotes.setMinimumWidth(250)
        self.qmNotes.setMinimumHeight(60)
        
        btn = QtWidgets.QPushButton('Save\nSettings', self)
        btn.clicked.connect(self.save_settings)
        btn.setMinimumWidth(70)
        btn.setMinimumHeight(50)
        btn.move(380,310)

        ##########################################################
        ##########################################################
        ##########################################################
        self.config, self.protocol, self.subject = None, None, None
        
        self.get_config_list()
        self.load_settings()
	# self.toggle_FaceCamera_process() # initialize if pre-set
        
        self.experiment = {} # storing the specifics of an experiment
        self.show()

    ### GUI FUNCTIONS ###
    def toggle_FaceCamera_process(self):
        if self.FaceCameraButton.isChecked() and (self.FaceCamera_process is None):
            # need to launch it
            self.statusBar.showMessage('  starting FaceCamera stream [...] ')
            self.show()
            self.closeFaceCamera_event.clear()
            self.FaceCamera_process = multiprocessing.Process(target=launch_FaceCamera,
                                        args=(self.run_event , self.closeFaceCamera_event, self.datafolder,
                                              {'frame_rate':self.config['FaceCamera-frame-rate']}))
            self.FaceCamera_process.start()
            time.sleep(6)
            self.statusBar.showMessage('[ok] FaceCamera ready ! ')
            
        elif (not self.FaceCameraButton.isChecked()) and (self.FaceCamera_process is not None):
            # need to shut it down
            self.closeFaceCamera_event.set()
            self.statusBar.showMessage(' FaceCamera stream interupted !')
            self.FaceCamera_process = None
            
        
    def save_settings(self):
        settings = {'config':self.cbc.currentText(),
                    'protocol':self.cbp.currentText(),
                    'subject':self.cbs.currentText()}
        for label, button in zip(['VisualStimButton', 'LocomotionButton', 'LFPButton',
                                  'FaceCameraButton', 'CaImagingButton'],
                                 [self.VisualStimButton, self.LocomotionButton, self.LFPButton,
                                  self.FaceCameraButton, self.CaImagingButton]):
            settings[label] = button.isChecked()
        np.save(settings_filename, settings)
        self.statusBar.showMessage('settings succesfully saved !')

    def load_settings(self):
        if os.path.isfile(settings_filename):
            settings = np.load(settings_filename, allow_pickle=True).item()
            if settings['config'] in self.config_list:
                self.cbc.setCurrentText(settings['config'])
                self.update_config()
            if settings['protocol'] in self.protocol_list:
                self.cbp.setCurrentText(settings['protocol'])
                self.update_protocol()
            if settings['subject'] in self.subjects:
                self.cbs.setCurrentText(settings['subject'])
                self.update_subject()
            for label, button in zip(['VisualStimButton', 'LocomotionButton', 'LFPButton',
                                      'FaceCameraButton', 'CaImagingButton'],
                                     [self.VisualStimButton, self.LocomotionButton, self.LFPButton,
                                      self.FaceCameraButton, self.CaImagingButton]):
                button.setChecked(settings[label])
        if (self.config is None) or (self.protocol is None) or (self.subject is None):
            self.statusBar.showMessage(' /!\ Problem in loading settings /!\  ')
    
    def get_config_list(self):
        files = os.listdir(os.path.join(base_path, 'configs'))
        self.config_list = [f.replace('.json', '') for f in files if f.endswith('.json')]
        self.cbc.addItems(self.config_list)
        self.update_config()
        
    def update_config(self):
        fn = os.path.join(base_path, 'configs', self.cbc.currentText()+'.json')
        with open(fn) as f:
            self.config = json.load(f)
        self.get_protocol_list()
        self.get_subject_list()
        self.root_datafolder = os.path.join(os.path.expanduser('~'), self.config['root_datafolder'])
        self.Screen = self.config['Screen']

    def get_protocol_list(self):
        if self.config['protocols']=='all':
            files = os.listdir(os.path.join(base_path, 'protocols'))
            self.protocol_list = [f.replace('.json', '') for f in files if f.endswith('.json')]
        else:
            self.protocol_list = self.config['protocols']
        self.cbp.clear()
        self.cbp.addItems(['None']+self.protocol_list)

    def update_protocol(self):
        if self.cbp.currentText()=='None':
            self.protocol = {}
        else:
            fn = os.path.join(base_path, 'protocols', self.cbp.currentText()+'.json')
            with open(fn) as f:
                self.protocol = json.load(f)
            self.protocol['Screen'] = self.config['Screen'] # override params
            self.protocol['data-folder'] = self.root_datafolder

    def update_screen(self):
        print(self.cbsc.currentText())
            
    def get_subject_list(self):
        with open(os.path.join(base_path, 'subjects', self.config['subjects_file'])) as f:
            self.subjects = json.load(f)
        self.cbs.clear()
        self.cbs.addItems(self.subjects.keys())
        
    def update_subject(self):
        self.subject = self.subjects[self.cbs.currentText()]
        
    def rigview(self):
        if self.RigView_process is not None:
            self.RigView_process.terminate()
        self.statusBar.showMessage('Initializing RigView stream [...]')
        self.RigView_process = multiprocessing.Process(target=launch_RigView,
                          args=(self.run_event , self.quit_event, self.datafolder))
        self.RigView_process.start()
        time.sleep(5)
        self.statusBar.showMessage('Screen ready')
        
    def initialize(self):

        ### set up all metadata
        self.metadata = {'config':self.cbc.currentText(),
                         'protocol':self.cbp.currentText(),
                         'notes':self.qmNotes.toPlainText(),
                         'subject_ID':self.cbs.currentText(),
                         'subject_props':self.subjects[self.cbs.currentText()]}

        for d in [self.config, self.protocol]:
            if d is not None:
                for key in d:
                    self.metadata[key] = d[key]
        
        # Setup configuration
        for modality, button in zip(['VisualStim', 'Locomotion', 'LFP', 'Vm',
                                     'FaceCamera', 'CaImaging'],
                                    [self.VisualStimButton, self.LocomotionButton, self.LFPButton, self.VmButton,
                                     self.FaceCameraButton, self.CaImagingButton]):
            self.metadata[modality] = bool(button.isChecked())

        if self.cbp.currentText()=='None':
            self.statusBar.showMessage('[...] initializing acquisition')
        else:
            self.statusBar.showMessage('[...] initializing acquisition & stimulation')

        self.filename = generate_filename_path(self.root_datafolder,
                                               filename='metadata', extension='.npy',
                                               with_FaceCamera_frames_folder=self.metadata['FaceCamera'],
                                               with_screen_frames_folder=self.metadata['VisualStim'])
        self.datafolder.set(os.path.dirname(self.filename))
            
        if self.metadata['protocol']!='None':
            with open(os.path.join(base_path, 'protocols', self.metadata['protocol']+'.json'), 'r') as fp:
                self.protocol = json.load(fp)
        else:
                self.protocol = {}
                
        # init visual stimulation
        if self.metadata['VisualStim'] and len(self.protocol.keys())>0:
            self.protocol['screen'] = self.metadata['Screen']
            self.stim = build_stim(self.protocol)
            np.save(os.path.join(str(self.datafolder.get()), 'visual-stim.npy'), self.stim.experiment)
            print('[ok] Visual-stimulation data saved as "%s"' % os.path.join(str(self.datafolder.get()), 'visual-stim.npy'))
            if 'time_stop' in self.stim.experiment:
                max_time = int(3*self.stim.experiment['time_stop'][-1]) # for security
                print('max_time', max_time)
            else:
                max_time = 1*60*60 # 1 hour, should be stopped manually
        else:
            max_time = 1*60*60 # 1 hour, should be stopped manually
            self.stim = None

        output_steps = []
        if self.metadata['CaImaging']:
            output_steps.append(self.config['STEP_FOR_CA_IMAGING_TRIGGER'])

        # --------------- #
        ### NI daq init ###   ## we override parameters based on the chosen modalities if needed
        # --------------- #
        if self.metadata['VisualStim'] and (self.metadata['NIdaq-analog-input-channels']<1):
            self.metadata['NIdaq-analog-input-channels'] = 1 # at least one (AI0), -> the photodiode
        if self.metadata['Locomotion'] and (self.metadata['NIdaq-digital-input-channels']<2):
            self.metadata['NIdaq-digital-input-channels'] = 2
        if self.metadata['LFP'] and self.metadata['Vm']:
            self.metadata['NIdaq-analog-input-channels'] = 3 # both channels, -> channel AI1 for Vm, AI2 for LFP 
        elif self.metadata['LFP']:
            self.metadata['NIdaq-analog-input-channels'] = 2 # AI1 for LFP 
        elif self.metadata['Vm']:
            self.metadata['NIdaq-analog-input-channels'] = 2 # AI1 for Vm
            
        try:
            self.acq = Acquisition(dt=1./self.metadata['NIdaq-acquisition-frequency'],
                                   Nchannel_analog_in=self.metadata['NIdaq-analog-input-channels'],
                                   Nchannel_digital_in=self.metadata['NIdaq-digital-input-channels'],
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
                          output_steps= [self.config['STEP_FOR_CA_IMAGING_TRIGGER']],
                          filename=None)
        acq.launch()
        time.sleep(1.1)
        acq.close()
    
    def quit(self):
        self.quit_event.set()
        if self.FaceCamera_process is not None:
            self.closeFaceCamera_event.set()
        if self.acq is not None:
            self.acq.close()
        if self.stim is not None:
            self.stim.quit()
        QtWidgets.QApplication.quit()

    def save_experiment(self):
        # SAVING THE METADATA FILES
        self.metadata['filename'] = str(self.datafolder.get())
        for key in self.protocol:
            self.metadata[key] = self.protocol[key]
        np.save(os.path.join(str(self.datafolder.get()), 'metadata.npy'), self.metadata)
        print('[ok] Metadata data saved as: %s ' % os.path.join(str(self.datafolder.get()), 'metadata.npy'))
        self.statusBar.showMessage('Metadata saved as: "%s" ' % os.path.join(str(self.datafolder.get()), 'metadata.npy'))

        
def run(app, args=None):
    return MainWindow(app, args)
    
if __name__=='__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = run(app)
    sys.exit(app.exec_())
