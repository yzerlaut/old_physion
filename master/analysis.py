import sys, time, tempfile, os, pathlib, json, subprocess, datetime
import numpy as np
from PyQt5 import QtGui, QtWidgets, QtCore
import pyqtgraph as pg

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import day_folder, generate_filename_path, save_dict, load_dict
from master import guiparts

class MasterWindow(QtWidgets.QMainWindow):
    
    def __init__(self, app,
                 saturation=100,
                 fullscreen=False):

        guiparts.build_dark_palette(app)
        
        super(MasterWindow, self).__init__()

        # adding a "quit" keyboard shortcut
        self.quitSc = QtWidgets.QShortcut(QtGui.QKeySequence('Q'), self) # or 'Ctrl+Q'
        self.quitSc.activated.connect(self.quit)
        self.refreshSc = QtWidgets.QShortcut(QtGui.QKeySequence('R'), self) # or 'Ctrl+Q'
        self.refreshSc.activated.connect(self.refresh)
        self.maxSc = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+M'), self)
        self.maxSc.activated.connect(self.showwindow)

        self.setWindowTitle('Analysis Program -- Physiology of Visual Circuits')
        pg.setConfigOptions(imageAxisOrder='row-major')

        self.setGeometry(200,200,1000,600)

        # config in guiparts.py
        guiparts.load_config1(self)

        self.minView = False
        self.showwindow()

    def showwindow(self):
        if self.minView:
            self.minView = self.maxview()
        else:
            self.minView = self.minview()
            
    def maxview(self):
        self.showFullScreen()
        return False

    def minview(self):
        self.showNormal()
        return True

    def pick_date(self):
        pass

    def pick_datafolder(self):
        pass

    def display_quantities(self):
        pass
    
    def play(self):
        pass

    def pause(self):
        pass

    def refresh(self):
        guiparts.load_config2(self)
    
    def update_frame(self):
        pass
    
    def quit(self):
        try:
            self.camera.cam.stop()
        except Exception:
            pass
        sys.exit()

app = QtWidgets.QApplication(sys.argv)
main = MasterWindow(app)
sys.exit(app.exec_())
