import sys, os, shutil, glob, time, subprocess, pathlib, json, tempfile
import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
from scipy.interpolate import interp1d
try:
    from pycromanager import Bridge
except ModuleNotFoundError:
    print('camera support not available !')

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from misc.folders import FOLDERS
from misc.guiparts import NewWindow
from assembling.saving import generate_filename_path
from visual_stim.psychopy_code.stimuli import visual_stim, visual
import multiprocessing # for the camera streams !!

subjects_path = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'exp', 'subjects')

class df:
    def __init__(self):
        pass
    def get(self):
        return tempfile.gettempdir()

class dummy_parent:
    def __init__(self):
        self.stop_flag = False
        self.datafolder = df()

class MainWindow(NewWindow):
    
    def __init__(self, app,
                 args=None,
                 parent=None,
                 spatial_subsampling=1,
                 time_subsampling=1):
        """
        Intrinsic Imaging GUI
        """
        self.app = app
        
        super(MainWindow, self).__init__(i=1,
                                         title='intrinsic imaging')

        ###
        try:
            bridge = Bridge()
            core = bridge.get_core()
            self.exposure = core.get_exposure()
        except BaseException as be:
            print(be)
            print('')
            print(' /!\ Problem with the Camera /!\ ')
            print('        --> no camera found ')
            print('')
            self.exposure = -1 # flag for no camera
        

        # some initialisation
        self.running, self.stim = False, None
        self.datafolder = ''        
        
        ########################
        ##### building GUI #####
        ########################
        
        self.minView = False
        self.showwindow()
        # central widget
        self.cwidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.cwidget)

        # layout
        Nx_wdgt, Ny_wdgt, self.wdgt_length = 20, 20, 3
        self.i_wdgt = 0
        self.l0 = QtWidgets.QGridLayout()
        self.cwidget.setLayout(self.l0)
        
        self.win = pg.GraphicsLayoutWidget()
        self.l0.addWidget(self.win,0, self.wdgt_length,
                          Nx_wdgt, Ny_wdgt-self.wdgt_length)
        
        # -- A plot area (ViewBox + axes) for displaying the image ---
        self.view = self.win.addViewBox(lockAspect=False,row=0,col=0,invertY=True,
                                      border=[100,100,100])
        self.pimg = pg.ImageItem()
        self.view.setAspectLocked()
        self.view.setRange(QtCore.QRectF(0, 0, 1280, 1024))
        self.view.addItem(self.pimg)
        
        # ---  setting subject information ---
        self.add_widget(QtWidgets.QLabel('subjects file:'))
        self.subjectFileBox = QtWidgets.QComboBox(self)
        self.subjectFileBox.addItems([f for f in os.listdir(subjects_path) if f.endswith('.json')])
        self.subjectFileBox.activated.connect(self.get_subject_list)
        self.add_widget(self.subjectFileBox)

        self.add_widget(QtWidgets.QLabel('subject:'))
        self.subjectBox = QtWidgets.QComboBox(self)
        self.get_subject_list()
        self.add_widget(self.subjectBox)

        self.add_widget(QtWidgets.QLabel(20*' - '))
        
        # ---  data acquisition properties ---
        self.add_widget(QtWidgets.QLabel('data folder:'), spec='small-left')
        self.folderB = QtWidgets.QComboBox(self)
        self.folderB.addItems(FOLDERS.keys())
        self.add_widget(self.folderB, spec='large-right')

        self.add_widget(QtWidgets.QLabel('  - exposure (ms):'),
                        spec='large-left')
        self.exposureBox = QtWidgets.QLineEdit()
        self.exposureBox.setText(str(self.exposure))
        self.add_widget(self.exposureBox, spec='small-right')

        self.add_widget(QtWidgets.QLabel('  - speed (degree/s):'),
                        spec='large-left')
        self.speedBox = QtWidgets.QLineEdit()
        self.speedBox.setText('10')
        self.add_widget(self.speedBox, spec='small-right')

        self.add_widget(QtWidgets.QLabel('  - bar size (degree):'),
                        spec='large-left')
        self.barBox = QtWidgets.QLineEdit()
        self.barBox.setText('4')
        self.add_widget(self.barBox, spec='small-right')

        self.add_widget(QtWidgets.QLabel('  - acq. freq. max. (Hz):'),
                        spec='large-left')
        self.freqBox = QtWidgets.QLineEdit()
        self.freqBox.setText('30')
        self.add_widget(self.freqBox, spec='small-right')
        
        self.demoBox = QtWidgets.QCheckBox("demo mode")
        self.demoBox.setStyleSheet("color: gray;")
        self.add_widget(self.demoBox, spec='large-right')
        self.demoBox.setChecked(True)
        
        # ---  launching acquisition ---
        self.liveButton = QtWidgets.QPushButton("--   live view    -- ", self)
        self.liveButton.clicked.connect(self.live_view)
        self.add_widget(self.liveButton)
        
        # ---  launching acquisition ---
        self.acqButton = QtWidgets.QPushButton("-- RUN PROTOCOL -- ", self)
        self.acqButton.clicked.connect(self.launch_protocol)
        self.add_widget(self.acqButton, spec='large-left')
        self.stopButton = QtWidgets.QPushButton(" STOP ", self)
        self.stopButton.clicked.connect(self.stop_protocol)
        self.add_widget(self.stopButton, spec='small-right')

        # ---  launching analysis ---
        self.add_widget(QtWidgets.QLabel(20*' - '))
        
        self.folderButton = QtWidgets.QPushButton("load data [Ctrl+O]", self)
        self.folderButton.clicked.connect(self.open_file)
        self.add_widget(self.folderButton, spec='large-left')
        self.lastBox = QtWidgets.QCheckBox("last ")
        self.lastBox.setStyleSheet("color: gray;")
        self.add_widget(self.lastBox, spec='small-right')
        self.lastBox.setChecked(True)

        self.add_widget(QtWidgets.QLabel('  - spatial smoothing (px):'),
                        spec='large-left')
        self.spatialSmoothingBox = QtWidgets.QLineEdit()
        self.spatialSmoothingBox.setText('5')
        self.add_widget(self.spatialSmoothingBox, spec='small-right')

        self.add_widget(QtWidgets.QLabel('  - temporal smoothing (ms):'),
                        spec='large-left')
        self.temporalSmoothingBox = QtWidgets.QLineEdit()
        self.temporalSmoothingBox.setText('100')
        self.add_widget(self.temporalSmoothingBox, spec='small-right')
        
        self.add_widget(QtWidgets.QLabel(' '))
        self.analysisButton = QtWidgets.QPushButton("- RUN ANALYSIS - ", self)
        self.analysisButton.clicked.connect(self.launch_analysis)
        self.add_widget(self.analysisButton)
        
        self.add_widget(QtWidgets.QLabel(20*' - '))

    def add_widget(self, wdgt, spec='None'):
        if 'small' in spec:
            wdgt.setFixedWidth(70)
            
        if spec=='small-left':
            self.l0.addWidget(wdgt, self.i_wdgt, 0, 1, 1)
        elif spec=='large-left':
            self.l0.addWidget(wdgt, self.i_wdgt, 0, 1, self.wdgt_length-1)
        elif spec=='small-right':
            self.l0.addWidget(wdgt, self.i_wdgt, self.wdgt_length-1, 1, 1)
            self.i_wdgt += 1
        elif spec=='large-right':
            self.l0.addWidget(wdgt, self.i_wdgt, 1, 1, self.wdgt_length-1)
            self.i_wdgt += 1
        else:
            self.l0.addWidget(wdgt, self.i_wdgt, 0, 1, self.wdgt_length)
            self.i_wdgt += 1
        
    def get_subject_list(self):
        with open(os.path.join(subjects_path, self.subjectFileBox.currentText())) as f:
            self.subjects = json.load(f)
        self.subjectBox.clear()
        self.subjectBox.addItems(self.subjects.keys())

    def init_visual_stim(self, demo=True):

        with open(os.path.join(pathlib.Path(__file__).resolve().parents[1], 'intrinsic', 'vis_stim', 'up.json'), 'r') as fp:
            protocol = json.load(fp)

        if self.demoBox.isChecked():
            protocol['demo'] = True

        self.stim = visual_stim.build_stim(protocol)
        self.parent = dummy_parent()

    def init_camera(self):
        self.bridge = Bridge()
        self.core = self.bridge.get_core()
        self.core.set_exposure(int(self.exposureBox.text()))
        # SHUTTER PROPS ???
        # auto_shutter = self.core.get_property('Core', 'AutoShutter')
        # self.core.set_property('Core', 'AutoShutter', 0)

    def get_pattern(self, direction, angle, size):

        if direction=='horizontal':
            return visual.Rect(win=self.stim.win,
                               size=(self.stim.angle_to_pix(size), 2000),
                               pos=(self.stim.angle_to_pix(angle), 0),
                               units='pix', fillColor=1, color=-1)
        elif direction=='vertical':
            return visual.Rect(win=self.stim.win,
                               size=(2000, self.stim.angle_to_pix(size)),
                               pos=(0, self.stim.angle_to_pix(angle)),
                               units='pix', fillColor=1, color=-1)
        
    def run(self):

        self.stim = visual_stim({"Screen": "Dell-2020",
                                 "presentation-prestim-screen": -1,
                                 "presentation-poststim-screen": -1}, demo=self.demoBox.isChecked())
        
        self.speed = float(self.speedBox.text()) # degree / second
        self.bar_size = float(self.barBox.text()) # degree / second
            
        xmin, xmax = 1.2*np.min(self.stim.x), 1.2*np.max(self.stim.x)
        zmin, zmax = 1.2*np.min(self.stim.z), 1.2*np.max(self.stim.z)

        self.angle_start, self.angle_max, self.direction, self.label = 0, 0, '', ''

        self.STIM = {'angle_start':[zmin, xmax, zmax, xmin],
                     'angle_stop':[zmax, xmin, zmin, xmax],
                     'direction':['vertical', 'horizontal', 'vertical', 'horizontal'],
                     'label':['up', 'left', 'down', 'right']}
        
        self.index, self.tstart = 0, time.time()
        
        self.update_dt()
                
        if self.exposure>0: # at the end we close the camera
            self.bridge.close()
        

    def update_dt(self):

        new_time = time.time()-self.tstart
        
        # update stim angle
        self.angle = self.STIM['angle_start'][self.index%4]+\
            self.speed*(new_time)*np.sign(self.STIM['angle_stop'][self.index%4]-self.STIM['angle_start'][self.index%4])
        
        # print('angle=%.1f' % self.angle, 'dt=%.1f' % (time.time()-self.tstart), 'label: ', self.STIM['label'][self.index%4])
        
        # grab frame
        frame = self.get_frame()
        if True: # live display
            self.pimg.setImage(frame)

        # NEED TO STORE DATA HERE (time, angle, frame)
        self.TIMES.append(new_time)
        self.ANGLES.append(self.angle)
        self.FRAMES.append(frame)
        
        # update stim image
        pattern = self.get_pattern(self.STIM['direction'][self.index%4], self.angle, self.bar_size)
        pattern.draw()
        try:
            self.stim.win.flip()
        except BaseException as be:
            pass
            
        # continuing ?
        if self.running:

            time.sleep(1./float(self.freqBox.text())) # max acq freq. here !
            
            # checking if not episode over
            if (np.abs(self.angle-self.STIM['angle_start'][self.index%4])>=np.abs(self.STIM['angle_stop'][self.index%4]-self.STIM['angle_start'][self.index%4])):
                self.write_data() # writing data when over
                self.index += 1
                self.tstart=time.time()
                
            QtCore.QTimer.singleShot(1, self.update_dt)

    def write_data(self):

        data = {'times':self.TIMES,
                'angles':self.ANGLES,
                'frames':self.FRAMES}

        np.save(os.path.join(self.datafolder,
            '%s-%i.npy' % (self.STIM['label'][self.index%4], int(self.index/4)+1)), data)
        
        
    def launch_protocol(self):

        if not self.running:
            self.running = True

            # initialization of camera:
            if self.exposure>0:
                self.init_camera()

            # initialization of data
            self.TIMES, self.ANGLES, self.FRAMES = [], [], []

            # init
            filename = generate_filename_path(FOLDERS[self.folderB.currentText()],
                                                   filename='metadata', extension='.npy')
            metadata = {'subject':str(self.subjectBox.currentText()),
                        'exposure':float(self.exposureBox.text()),
                        'bar-size':float(self.barBox.text()),
                        'acq-freq':float(self.freqBox.text()),
                        'speed':float(self.speedBox.text())}
            np.save(filename, metadata)
            self.datafolder = os.path.dirname(filename)

            print('acquisition running [...]')
            self.run()
            
        else:
            print(' /!\  --> pb in launching acquisition (either already running or missing camera)')

    def live_view(self):
        self.running = True
        self.update_Image()
        
    def stop_protocol(self):
        if self.running:
            self.running = False
            if self.stim is not None:
                self.stim.close()
        else:
            print('acquisition not launched')

    def get_frame(self):
        if self.exposure>0:
            self.core.snap_image()
            tagged_image = self.core.get_tagged_image()
            #pixels by default come out as a 1D array. We can reshape them into an image
            return np.reshape(tagged_image.pix,
                              newshape=[tagged_image.tags['Height'], tagged_image.tags['Width']])
        else:
            return np.random.randn(720, 1080)
        
    def update_Image(self):
        # plot it
        self.pimg.setImage(self.get_frame())
        if self.running:
            QtCore.QTimer.singleShot(1, self.update_Image)
        elif self.exposure>0: # at the end
            self.bridge.close()

    def hitting_space(self):
        if not self.running:
            self.launch_protocol()
        else:
            self.stop_protocol()

    def launch_analysis(self):
        print('launching analysis [...]')

    def process(self):
        self.launch_analysis()
    

    def open_file(self):

        self.lastBox.setChecked(False)
        folder = QtWidgets.QFileDialog.getExistingDirectory(self,\
                                                            "Choose datafolder",
                                                            FOLDERS[self.folderB.currentText()])

        if folder!='':
            self.datafolder = folder
        else:
            print('data-folder not set !')
        
        
def run(app, args=None, parent=None):
    return MainWindow(app,
                      args=args,
                      parent=parent)
    
if __name__=='__main__':
    from misc.colors import build_dark_palette
    import tempfile, argparse, os
    parser=argparse.ArgumentParser(description="Experiment interface",
                       formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-f', "--datafile", type=str,default='')
    parser.add_argument('-rf', "--root_datafolder", type=str,
                        default=os.path.join(os.path.expanduser('~'), 'DATA'))
    parser.add_argument('-v', "--verbose", action="store_true")
    args = parser.parse_args()
    app = QtWidgets.QApplication(sys.argv)
    build_dark_palette(app)
    main = MainWindow(app,
                      args=args)
    sys.exit(app.exec_())
