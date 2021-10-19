import sys, time, tempfile, os, pathlib, json, subprocess
import threading # for the camera stream
import numpy as np
from PyQt5 import QtGui, QtWidgets, QtCore
import pyqtgraph as pg

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import day_folder, generate_filename_path, save_dict, load_dict
from pupil import process

from hardware_control.FLIRcamera.recording import CameraAcquisition
import matplotlib.pylab as plt

# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
## NASTY workaround to the error:
# ** OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized. **

class MasterWindow(QtWidgets.QMainWindow):
    
    def __init__(self, app,
                 saturation=100):
        
        super(MasterWindow, self).__init__()

        # adding a "quit" keyboard shortcut
        self.quitSc = QtWidgets.QShortcut(QtGui.QKeySequence('Q'), self) # or 'Ctrl+Q'
        self.quitSc.activated.connect(self.quit)

        # self.runC = QtWidgets.QShortcut(QtGui.QKeySequence('R'), self) # or 'Ctrl+R'
        # self.runC.activated.connect(self.run)
        
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

        self.camera = CameraAcquisition(folder=tempfile.gettempdir(),
                                        settings={'frame_rate':args.frame_rate})
                                                  # 'gain':args.gain,
                                                  # 'exposure_time':args.exposure_time})
        self.camera.cam.start()

        image = self.camera.cam.get_array()
        self.p0img.setImage(image)
        print(image.shape)
        Lx, Ly = image.shape
        self.x,self.y = np.meshgrid(np.arange(0,Lx), np.arange(0,Ly), indexing='ij')
        cx, cy, sx, sy = args.ellipse
        self.ellipse = ((self.y - cy)**2 / (sy/2)**2 +
                        (self.x - cx)**2 / (sx/2)**2) <= 1
        zoom = process.preprocess(args, img=image, ellipse=args.ellipse)
        
        self.pROIimg.setImage(zoom[np.min(self.x[self.ellipse]):np.max(self.x[self.ellipse]):,\
                                   np.min(self.y[self.ellipse]):np.max(self.y[self.ellipse])])
        
        self.pROI.show()
        self.p0.show()
        self.pROIimg.show()
        self.p0img.show()
        self.win.show()
        self.show()

        self.run()

    def run(self):

        t0 = time.time()
        while (time.time()-t0)<args.tstop:
            image = self.camera.cam.get_array()
            self.p0img.setImage(image)
            self.p0img.setLevels([0,255])
            zoom = process.preprocess(args, img=image, ellipse=args.ellipse)
            self.pROIimg.setImage(zoom[np.min(self.x[self.ellipse]):np.max(self.x[self.ellipse]):,\
                                       np.min(self.y[self.ellipse]):np.max(self.y[self.ellipse])])
            self.pROIimg.setLevels([0,255])
            pg.QtGui.QApplication.processEvents()
            

    def quit(self):
        self.camera.cam.stop()
        QtWidgets.QApplication.quit()


        
if __name__ == '__main__':
    
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--shape", default='circle')
    parser.add_argument("--tstop", type=float, default=5.)
    parser.add_argument("--frame_rate", type=float, default=5.)
    parser.add_argument("--saturation", type=float, default=75)
    # parser.add_argument("--gain", type=float, default=10)
    # parser.add_argument("--exposure_time", type=float, default=10000)
    parser.add_argument("--ellipse", type=float, default=(251, 316, 251, 254), nargs=4)
    parser.add_argument("--gaussian_smoothing", type=float, default=2)
    parser.add_argument('-df', "--datafolder", default='./')
    parser.add_argument('-f', "--saving_filename", default='pupil-data.npy')
    parser.add_argument("-nv", "--non_verbose", help="decrease output verbosity", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = QtWidgets.QApplication([''])
    main = MasterWindow(app)
    sys.exit(app.exec_())
