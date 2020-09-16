import sys, time, tempfile, os, pathlib, json, subprocess
import numpy as np
from PyQt5 import QtGui, QtWidgets, QtCore
import pyqtgraph as pg

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import day_folder, generate_filename_path, save_dict, load_dict


## NASTY workaround to the error:
# ** OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized. **

class MasterWindow(QtWidgets.QMainWindow):
    
    def __init__(self, app,
                 saturation=100):
        
        super(MasterWindow, self).__init__()

        
        # adding a "quit" keyboard shortcut
        self.quitSc = QtWidgets.QShortcut(QtGui.QKeySequence('Q'), self) # or 'Ctrl+Q'
        self.quitSc.activated.connect(self.quit)

        
        self.setWindowTitle('Analysis Program -- Physiology of Visual Circuits')
        pg.setConfigOptions(imageAxisOrder='row-major')
        # self.setGeometry(0,0,1600,1200)
        self.showFullScreen()
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


    def quit(self):
        try:
            self.camera.cam.stop()
        except Exception:
            pass
        sys.exit()

app = QtWidgets.QApplication(sys.argv)
main = MasterWindow(app)
sys.exit(app.exec_())
