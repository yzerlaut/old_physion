import sys, os, shutil, glob, time, subprocess, pathlib, json
import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
from scipy.interpolate import interp1d
from pycromanager import Bridge

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from misc.folders import FOLDERS
from misc.guiparts import NewWindow
from assembling.saving import *
import multiprocessing # for the camera streams !!

subjects_path = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'exp', 'subjects')

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
        self.running = False
        
        self.l0 = QtWidgets.QGridLayout()
        self.cwidget.setLayout(self.l0)
        
        self.win = pg.GraphicsLayoutWidget()
        self.l0.addWidget(self.win,0, self.wdgt_length,
                          Nx_wdgt, Ny_wdgt-self.wdgt_length)
        
        # layout = self.win.ci.layout

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

        self.add_widget(QtWidgets.QLabel('  - speed (degree/min):'),
                        spec='large-left')
        self.speedBox = QtWidgets.QLineEdit()
        self.speedBox.setText(str(self.exposure))
        self.add_widget(self.speedBox, spec='small-right')

        self.demoBox = QtWidgets.QCheckBox("demo mode")
        self.demoBox.setStyleSheet("color: gray;")
        self.add_widget(self.demoBox, spec='large-right')
        
        # ---  launching acquisition ---
        self.acqButton = QtWidgets.QPushButton("-- RUN PROTOCOL -- ", self)
        self.acqButton.clicked.connect(self.launch_protocol)
        self.add_widget(self.acqButton, spec='large-left')
        self.stopButton = QtWidgets.QPushButton(" STOP ", self)
        self.stopButton.clicked.connect(self.stop_protocol)
        self.add_widget(self.stopButton, spec='small-right')

        
        # ---  launching analysis ---
        self.add_widget(QtWidgets.QLabel(20*' - '))
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

    def set_datafolder(self):
        self.filename = generate_filename_path(FOLDERS[self.folderB.currentText()],
                                               filename='metadata', extension='.npy')
        self.datafolder.set(os.path.dirname(self.filename))
        
    def launch_protocol(self):

        if self.screen is None:
            if self.demoBox.isChecked():
                self.window = visual.Window(SCREEN,monitor="testMonitor", units="deg", color=-1) #create a window
            else:
                # create monitor and window
                self.window = visual.Window(SCREEN,monitor="testMonitor", units="deg", color=-1) #create a window

        if not self.running:
            self.running = True

            if self.exposure>0:
                self.bridge = Bridge()
                self.core = self.bridge.get_core()
                self.core.set_exposure(int(self.exposureBox.text()))
            # SHUTTER PROPS ???
            # auto_shutter = self.core.get_property('Core', 'AutoShutter')
            # self.core.set_property('Core', 'AutoShutter', 0)
            self.start = time.time()
            print('acquisition running [...]')
            
            if self.exposure>0:
                self.update_Image() # ~ while loop
        else:
            print(' /!\  --> pb in launching acquisition (either already running or missing camera)')

    def stop_protocol(self):
        if self.running:
            self.running = False
        else:
            print('acquisition not launched')
            
    def update_Image(self):
        self.core.snap_image()
        tagged_image = self.core.get_tagged_image()
        #pixels by default come out as a 1D array. We can reshape them into an image
        frame = np.reshape(tagged_image.pix,
                           newshape=[tagged_image.tags['Height'], tagged_image.tags['Width']])
        #plot it
        self.pimg.setImage(frame)
        if self.running:
            QtCore.QTimer.singleShot(1, self.update_Image)
        else:
            self.bridge.close()

        
        
    def hitting_space(self):
        self.launch_protocol()

    def launch_analysis(self):
        print('launching analysis [...]')

    def process(self):
        self.launch_analysis()
    
    
        
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