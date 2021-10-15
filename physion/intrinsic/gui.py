import sys, os, shutil, glob, time, subprocess, pathlib, json
import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
from scipy.interpolate import interp1d

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from misc.folders import FOLDERS, python_path
from misc.style import set_dark_style, set_app_icon
from misc.guiparts import NewWindow, Slider
from assembling.tools import load_FaceCamera_data
from facemotion import roi, process

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
        
        # layout = self.win.ci.layout

        # -- A plot area (ViewBox + axes) for displaying the image ---
        self.p0 = self.win.addViewBox(lockAspect=False,row=0,col=0,invertY=True,
                                      border=[100,100,100])
        self.pimg = pg.ImageItem()
        self.p0.setAspectLocked()
        self.p0.addItem(self.pimg)
        
        self.pimg.setImage(np.random.randn(100,100))

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

        # ---  launching acquisition ---
        self.add_widget(QtWidgets.QLabel(' '))
        self.acqButton = QtWidgets.QPushButton("- RUN PROTOCOL = ", self)
        self.acqButton.clicked.connect(self.launch_protocol)
        self.add_widget(self.acqButton)

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
        
    def launch_protocol(self):
        print('launching protocol [...]')
        
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



    
        
