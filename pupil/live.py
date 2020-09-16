import sys, time, tempfile, os, pathlib, json, subprocess
import threading # for the camera stream
import numpy as np
from PyQt5 import QtGui, QtWidgets, QtCore

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
                 parent=None,
                 button_length = 100):
        
        super(MasterWindow, self).__init__(parent)
        
        self.data_folder = tempfile.gettempdir()
        
        self.setWindowTitle('FaceCamera Configuration')
        self.setGeometry(100, 100, 500, 100)

        mainMenu = self.menuBar()
        self.fileMenu = mainMenu.addMenu('')

        btn1 = QtWidgets.QPushButton('Choose config file', self)
        btn1.clicked.connect(self.choose_config)
        btn1.setMinimumWidth(100)
        btn1.move(30, 20)

        btn2 = QtWidgets.QPushButton('(re-)load config file and show image', self)
        btn2.clicked.connect(self.reload_config)
        btn2.setMinimumWidth(200)
        btn2.move(140, 20)

        btn3 = QtWidgets.QPushButton('quit', self)
        btn3.clicked.connect(self.quit)
        btn3.setMinimumWidth(100)
        btn3.move(350, 20)
        
        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage('')

        self.fig, self.ax = plt.subplots()
        self.ax.axis('off')
        self.show()
        self.fig.show()
        self.camera = None

    def choose_config(self):
        path = os.path.join(str(pathlib.Path(__file__).resolve().parents[1]), 'master', 'configs') #, 'VisualStim+NIdaq+FaceCamera.json')
        filename, success = QtWidgets.QFileDialog.getOpenFileName(self, 'Open protocol file', path ,"Config files (*.json)")
        if success:
            with open(filename, 'r') as fp:
                self.config = json.load(fp)
            if 'FaceCamera' not in self.config:
                self.statusBar.showMessage(' This config file has no "FaceCamera" key, see doc')
            else:
                self.statusBar.showMessage('"%s" successfully loaded' % os.path.basename(filename))
            self.camera = CameraAcquisition(folder=self.data_folder, settings=self.config['FaceCamera'])
            self.camera.cam.start()
        else:
            self.statusBar.showMessage(' config file not loaded')
        

    def reload_config(self):
        self.stop = True
        if self.camera is None:
            self.statusBar.showMessage('Need to load a config file first')
        else:
            self.camera.cam.stop()
            self.statusBar.showMessage('Showing images [...]')
            self.camera = CameraAcquisition(folder=self.data_folder, settings=self.config['FaceCamera'])
            self.camera.cam.start()
            self.stop = False
            self.ax.cla()
            image = self.camera.cam.get_array()
            self.ax.imshow(image, aspect='equal', interpolation='none', cmap=plt.cm.binary_r)
            self.ax.axis('off')
            self.fig.canvas.draw()
        
    def quit(self):
        sys.exit()

if __name__ == '__main__':
    
    app = QtWidgets.QApplication(sys.argv)
    main = MasterWindow(app)
    sys.exit(app.exec_())
