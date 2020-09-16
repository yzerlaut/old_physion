import sys, time, tempfile, os, pathlib, json, subprocess
import threading # for the camera stream
import numpy as np
from PyQt5 import QtGui, QtWidgets, QtCore
import pyqtgraph as pg

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import day_folder, generate_filename_path, save_dict, load_dict
from assembling.analysis import quick_data_view, analyze_data, last_datafile

from hardware_control.FLIRcamera.recording import CameraAcquisition
import matplotlib.pylab as plt

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
## NASTY workaround to the error:
# ** OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized. **

class MasterWindow(QtWidgets.QMainWindow):
    
    def __init__(self, app,
                 saturation=100):
        
        super(MasterWindow, self).__init__()

        
        # adding a "quit" keyboard shortcut
        self.quitSc = QtWidgets.QShortcut(QtGui.QKeySequence('Q'), self) # or 'Ctrl+Q'
        self.quitSc.activated.connect(self.quit)

        
        self.setWindowTitle('FaceCamera Configuration')
        pg.setConfigOptions(imageAxisOrder='row-major')
        self.setGeometry(100,100,800,350)
        self.setStyleSheet("QMainWindow {background: 'black';}")

        
        self.cwidget = QtGui.QWidget(self)
        self.setCentralWidget(self.cwidget)
        self.l0 = QtGui.QGridLayout()
        self.cwidget.setLayout(self.l0)
        self.win = pg.GraphicsLayoutWidget()
        self.win.move(50,50)
        self.win.resize(300,300)
        self.l0.addWidget(self.win,1,3,37,15)

        
        self.p0 = self.win.addViewBox(lockAspect=True,row=0,col=0,invertY=True,border=[100,100,100])
        self.p0.setAspectLocked()
        self.p0img = pg.ImageItem(None)
        self.p0.addItem(self.p0img)
        self.pROI = self.win.addViewBox(lockAspect=True,row=0,col=1,invertY=True, border=[100,100,100])
        self.pROI.setAspectLocked()
        self.pROIimg = pg.ImageItem(None)
        self.pROI.addItem(self.pROIimg)

        self.pROI.show()
        self.win.show()
        self.show()

        self.run()

    def run(self):

        try:
            self.camera = CameraAcquisition(folder=tempfile.gettempdir(),
                            settings={'frame_rate':20., 'gain':10., 'exposure_time':10000})
            self.camera.cam.start()
        except Exception:
            pass

        i=0
        while i<1000:
            # image = self.camera.cam.get_array()
            # self.p0img.setImage(image)
            i+=1
            

    def quit(self):
        try:
            self.camera.cam.stop()
        except Exception:
            pass
        sys.exit()

    
import argparse

parser=argparse.ArgumentParser()
parser.add_argument("--shape", default='circle')
parser.add_argument("--sampling_rate", type=float, default=5.)
parser.add_argument("--saturation", type=float, default=75)
# parser.add_argument("--ellipse", type=float, default=[], nargs=)
parser.add_argument("--gaussian_smoothing", type=float, default=2)
parser.add_argument('-df', "--datafolder", default='./')
parser.add_argument('-f', "--saving_filename", default='pupil-data.npy')
parser.add_argument("-nv", "--non_verbose", help="decrease output verbosity", action="store_true")
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

app = QtWidgets.QApplication([''])
main = MasterWindow(app)
sys.exit(app.exec_())
