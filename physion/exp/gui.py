from PyQt5 import QtGui, QtWidgets, QtCore
import sys, time, tempfile, os, pathlib, json, subprocess
import multiprocessing # for the camera streams !!
from ctypes import c_char_p
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import *

if not sys.argv[-1]=='no-stim':
    from visual_stim.stimuli import build_stim
    from visual_stim.screens import SCREENS
else:
    SCREENS = []

from misc.style import set_app_icon, set_dark_style
try:
    from hardware_control.NIdaq.main import Acquisition
except ModuleNotFoundError:
    print(' /!\ Problem with the NIdaq module /!\ ')
try:
    from hardware_control.FLIRcamera.recording import launch_FaceCamera
except ModuleNotFoundError:
    print(' /!\ Problem with the FLIR camera module /!\ ')

# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
## NASTY workaround to the error:
# ** OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized. **

base_path = str(pathlib.Path(__file__).resolve().parents[0])
settings_filename = os.path.join(base_path, 'settings.npy')

class MainWindow(QtWidgets.QMainWindow):

    MODALITIES = ['Locomotion', 'FaceCamera', 'NeuroPixels', 'EphysLFP', 'EphysVm', 'CaImaging']

    def __init__(self, app, args=None, demo=False):
        """
        """
        super(MainWindow, self).__init__()
        self.app = app

        self.setWindowTitle('Experimental module')
        self.setGeometry(400, 50, 550, 390)

        Y = 5 # coordinates of the current buttons

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
        rml = QtWidgets.QLabel('     '+'-'*40+" Recording modalities "+'-'*40, self)
        rml.move(35, Y)
        rml.setMinimumWidth(500)

        Y+=35
        for i, k in enumerate(self.MODALITIES):
            setattr(self, k+'Button', QtWidgets.QPushButton(k, self))
            getattr(self, k+'Button').move(30+80*i, Y)
            getattr(self, k+'Button').setMaximumWidth(75)
            getattr(self, k+'Button').setCheckable(True)

        self.FaceCameraButton.clicked.connect(self.toggle_FaceCamera_process)

        Y+=50
        # screen choice
        QtWidgets.QLabel(" Screen :", self).move(250, Y)
        self.cbsc = QtWidgets.QComboBox(self)
        self.cbsc.setMinimumWidth(200)
        self.cbsc.move(320, Y)
        self.cbsc.activated.connect(self.update_screen)
        self.cbsc.addItems(SCREENS.keys())
        
        self.demoW = QtWidgets.QCheckBox('demo', self)
        self.demoW.move(40, Y)
        if demo:
            self.demoW.setChecked(True)

        Y+=35
        # config choice
        QtWidgets.QLabel("  => Config :", self).move(160, Y)
        self.cbc = QtWidgets.QComboBox(self)
        self.cbc.setMinimumWidth(270)
        self.cbc.move(250, Y)
        self.cbc.activated.connect(self.update_config)

        Y+=35
        # subject choice
        QtWidgets.QLabel("-> Subject :", self).move(100, Y)
        self.cbs = QtWidgets.QComboBox(self)
        self.cbs.setMinimumWidth(340)
        self.cbs.move(180, Y)
        self.cbs.activated.connect(self.update_subject)
        
        Y+=35
        # protocol choice
        QtWidgets.QLabel(" Visual Protocol :", self).move(20, Y)
        self.cbp = QtWidgets.QComboBox(self)
        self.cbp.setMinimumWidth(390)
        self.cbp.move(130, Y)
        self.cbp.activated.connect(self.update_protocol)
       
        
        Y+=45
        # buttons and functions
        LABELS = ["i) Initialize", "b) Buffer", "r) Run", "s) Stop", "q) Quit"]
        FUNCTIONS = [self.initialize, self.buffer, self.run, self.stop, self.quit]
        
        mainMenu = self.menuBar()
        self.fileMenu = mainMenu.addMenu('')

        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage('ready to select a protocol/config')
        for func, label, name, shift in zip(FUNCTIONS, LABELS,
                ['initButton', 'bufferButton','runButton', 'stopButton', 'quitButton'],
                90*np.arange(5)):
            setattr(self, name, QtWidgets.QPushButton(label, self))
            getattr(self,name).clicked.connect(func)
            getattr(self,name).setFixedWidth(85)
            getattr(self,name).move(50+shift, Y)
            action = QtWidgets.QAction(label, self)
            action.setShortcut(label.split(')')[0])
            action.triggered.connect(func)
            self.fileMenu.addAction(action)

        self.bufferButton.setEnabled(False)
        self.runButton.setEnabled(False)

        Y+=45
        QtWidgets.QLabel("Notes: ", self).move(25, Y+10)
        self.qmNotes = QtWidgets.QTextEdit('', self)
        self.qmNotes.move(90, Y)
        self.qmNotes.setMinimumWidth(250)
        self.qmNotes.setMinimumHeight(60)
        
        btn = QtWidgets.QPushButton('Save\nSettings', self)
        btn.clicked.connect(self.save_settings)
        btn.setMinimumWidth(70)
        btn.setMinimumHeight(50)
        btn.move(380,Y+5)

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
            self.statusBar.showMessage('[ok] FaceCamera initialized (in 5-6s) ! ')
            
        elif (not self.FaceCameraButton.isChecked()) and (self.FaceCamera_process is not None):
            # need to shut it down
            self.closeFaceCamera_event.set()
            self.statusBar.showMessage(' FaceCamera stream interupted !')
            self.FaceCamera_process = None
            
        
    def save_settings(self):
        settings = {'config':self.cbc.currentText(),
                    'protocol':self.cbp.currentText(),
                    'subject':self.cbs.currentText()}
        
        for i, k in enumerate(self.MODALITIES):
            settings[k] = getattr(self, k+'Button').isChecked()
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
            for i, k in enumerate(self.MODALITIES):
                getattr(self, k+'Button').setChecked(settings[k])
        if (self.config is None) or (self.protocol is None) or (self.subject is None):
            self.statusBar.showMessage(' /!\ Problem in loading settings /!\  ')
    
    def get_config_list(self):
        files = os.listdir(os.path.join(base_path, 'configs'))
        self.config_list = [f.replace('.json', '') for f in files[::-1] if f.endswith('.json')]
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
        
    def NIdaq_metadata_init(self):
        # --------------- #
        ### NI daq init ###   ## we override parameters based on the chosen modalities if needed
        # --------------- #
        if self.metadata['VisualStim'] and (self.metadata['NIdaq-analog-input-channels']<1):
            self.metadata['NIdaq-analog-input-channels'] = 1 # at least one (AI0), -> the photodiode
        if self.metadata['Locomotion'] and (self.metadata['NIdaq-digital-input-channels']<2):
            self.metadata['NIdaq-digital-input-channels'] = 2
        if self.metadata['EphysLFP'] and self.metadata['EphysVm']:
            self.metadata['NIdaq-analog-input-channels'] = 3 # both channels, -> channel AI1 for Vm, AI2 for LFP 
        elif self.metadata['EphysLFP']:
            self.metadata['NIdaq-analog-input-channels'] = 2 # AI1 for LFP 
        elif self.metadata['EphysVm']:
            self.metadata['NIdaq-analog-input-channels'] = 2 # AI1 for Vm

    def initialize(self):

        self.runButton.setEnabled(False) # acq blocked during init
        self.bufferButton.setEnabled(False) # should be already blocked, but for security 

        ### set up all metadata
        self.metadata = {'config':self.cbc.currentText(),
                         'protocol':self.cbp.currentText(),
                         'VisualStim':self.cbp.currentText()!='None',
                         'notes':self.qmNotes.toPlainText(),
                         'subject_ID':self.cbs.currentText(),
                         'subject_props':self.subjects[self.cbs.currentText()]}

        for d in [self.config, self.protocol]:
            if d is not None:
                for key in d:
                    self.metadata[key] = d[key]
        
        for k in self.MODALITIES:
            self.metadata[k] = bool(getattr(self, k+'Button').isChecked())

        if self.cbp.currentText()=='None':
            self.statusBar.showMessage('[...] initializing acquisition')
        else:
            self.statusBar.showMessage('[...] initializing acquisition & stimulation')

        self.filename = generate_filename_path(self.root_datafolder,
                                               filename='metadata', extension='.npy',
                                               with_FaceCamera_frames_folder=self.metadata['FaceCamera'])
        self.datafolder.set(os.path.dirname(self.filename))

        if self.metadata['protocol']!='None':
            with open(os.path.join(base_path, 'protocols', self.metadata['protocol']+'.json'), 'r') as fp:
                self.protocol = json.load(fp)
        else:
                self.protocol = {}

        # init visual stimulation
        if not self.metadata['protocol']!='None':

            self.protocol['screen'] = self.metadata['Screen']

            if self.demoW.isChecked():
                self.protocol['demo'] = True
            else:
                self.protocol['demo'] = False

            #if not self.bufferButton.isChecked():
            #    # when we re-check the buffer after a run, no need to re-init the stimulation
            #    self.stim = build_stim(self.protocol)
            self.stim = build_stim(self.protocol)

            np.save(os.path.join(str(self.datafolder.get()), 'visual-stim.npy'), self.stim.experiment)
            print('[ok] Visual-stimulation data saved as "%s"' % os.path.join(str(self.datafolder.get()), 'visual-stim.npy'))

            if 'time_stop' in self.stim.experiment:
                max_time = min([4*60*60, int(10*np.max(self.stim.experiment['time_stop']))]) # 10 times for security, 4h max
            else:
                max_time = 1*60*60 # 1 hour, should be stopped manually
        else:
            max_time = 1*60*60 # 1 hour, should be stopped manually
            self.stim = None

        print('max_time of NIdaq recording: %.2dh:%.2dm:%.2ds' % (max_time/3600, (max_time%3600)/60, (max_time%60)))

        output_steps = []
        if self.metadata['CaImaging']:
            output_steps.append(self.config['STEP_FOR_CA_IMAGING_TRIGGER'])

        self.NIdaq_metadata_init()

        if not self.demoW.isChecked():
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
        self.bufferButton.setEnabled(True)
        self.runButton.setEnabled(True)
        self.save_experiment() # saving all metadata after full initialization

        if self.cbp.currentText()=='None':
            self.statusBar.showMessage('Acquisition ready !')
        else:
            self.statusBar.showMessage('Acquisition & Stimulation ready !')

    def buffer(self):
        # buffers the visual stimulus
        self.stim.buffer_stim(self, gui_refresh_func=self.app.processEvents)
        self.update()
        self.show()
        self.bufferButton.setEnabled(False)

    def run(self):
        self.stop_flag=False
        self.run_event.set() # start the run flag for the facecamera

        if ((self.acq is None) and (self.stim is None)) or not self.init:
            self.statusBar.showMessage('Need to initialize the stimulation !')
        elif (self.stim is None) and (self.acq is not None):
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
        self.runButton.setEnabled(False)
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
        self.runButton.setEnabled(False)
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

        
def run(app, demo=False):
    return MainWindow(app, demo=demo)
    
if __name__=='__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = run(app, demo='demo' in sys.argv)
    sys.exit(app.exec_())
