import sys, time, tempfile, os, pathlib, json, subprocess
import threading # for the camera stream
import numpy as np
from PyQt5 import QtGui, QtWidgets, QtCore

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import *
from assembling.analysis import quick_data_view, analyze_data, last_datafile


class MasterWindow(QtWidgets.QMainWindow):
    
    def __init__(self, app,
                 parent=None,
                 button_length = 110):
        
        super(MasterWindow, self).__init__(parent)
        
        self.data_folder = get_data_folder()
        
        self.setWindowTitle('Analysis Program -- Physiology of Visual Circuits')
        self.setGeometry(50, 50, 500, 130)

        # buttons and functions
        LABELS = ["l) Load data", "v) View data", "r) Run analysis", "q) Quit"]
        FUNCTIONS = [self.load_data, self.view_data, self.run, self.quit]
        
        mainMenu = self.menuBar()
        self.fileMenu = mainMenu.addMenu('')

        for func, label, shift in zip(FUNCTIONS, LABELS,\
                                      button_length*np.arange(len(LABELS))):
            btn = QtWidgets.QPushButton(label, self)
            btn.clicked.connect(func)
            btn.setMinimumWidth(button_length)
            btn.move(shift+30, 20)
            action = QtWidgets.QAction(label, self)
            action.setShortcut(label.split(')')[0])
            action.triggered.connect(func)
            self.fileMenu.addAction(action)

        self.dfl = QtWidgets.QLabel('Data-Folder (root): "%s"' % str(self.data_folder), self)
        self.dfl.setMinimumWidth(300)
        self.dfl.move(30, 70)
        dfb = QtWidgets.QPushButton('Set folder', self)
        dfb.clicked.connect(self.choose_data_folder)
        dfb.move(350, 70)
            
        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage('ready for analysis')
        
        self.show()

    def run(self):
        self.statusBar.showMessage('Analyzing last recording [...]')
        data, fig1 = quick_data_view(last_datafile(tempfile.gettempdir()), realign=True)
        _, fig2 = analyze_data(data=data)
        fig1.show()
        fig2.show()

    def choose_data_folder(self):
        fd = str(QtWidgets.QFileDialog.getExistingDirectory(self,
                                                            "Select Root Data Folder", self.data_folder))
        if os.path.isdir(fd):
            self.data_folder = fd
            set_data_folder(fd)
            self.dfl.setText('Data-Folder (root): "%s"' % str(self.data_folder))
        else:
            self.statusBar.showMessage('Invalid folder -> folder unchanged')
        
    def view_data(self):
        _, fig = quick_data_view(last_datafile(self.data_folder))
        fig.show()
        
    def load_data(self):
        pass
    
    def run(self):
        pass
    
    def quit(self):
        sys.exit()

if __name__ == '__main__':
    
    app = QtWidgets.QApplication(sys.argv)
    main = MasterWindow(app)
    sys.exit(app.exec_())