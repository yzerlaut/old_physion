import sys, time, tempfile, os, pathlib, json, datetime, string, subprocess
import numpy as np
from PyQt5 import QtGui, QtWidgets, QtCore
import pyqtgraph as pg

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from misc.folders import FOLDERS, python_path

class MainWindow(QtWidgets.QMainWindow):
    
    def __init__(self, app,
                 args=None,
                 parent=None):
        
        super(MainWindow, self).__init__()

        self.setWindowTitle('Lab Notebook')
        
        self.setGeometry(650, 300, 500, 600)
        # adding a "quit" keyboard shortcut
        self.quitSc = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+Q'), self) # or 'Ctrl+Q'
        self.quitSc.activated.connect(self.quit)
            
        HEIGHT = 0

        HEIGHT += 20
        QtWidgets.QLabel("Folder:", self).move(10, HEIGHT)
        self.sourceBox = QtWidgets.QComboBox(self)
        self.sourceBox.setMinimumWidth(150)
        self.sourceBox.move(70, HEIGHT)
        self.sourceBox.addItems(FOLDERS)
        
        self.gen = QtWidgets.QPushButton('Pick datafile', self)
        self.gen.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.gen.clicked.connect(self.pick_datafile)
        self.gen.setMinimumWidth(100)
        self.gen.move(250, HEIGHT)

        self.gen = QtWidgets.QPushButton(' Build PDF', self)
        self.gen.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.gen.clicked.connect(self.build_pdf)
        self.gen.setMinimumWidth(100)
        self.gen.move(370, HEIGHT)

        self.process_script = os.path.join(str(pathlib.Path(__file__).resolve().parents[1]),
                                           'analysis',  'summary_pdf.py')
        self.filename = ''
        self.show()


    def pick_datafile(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self,\
            "Open datafile (through metadata file)", FOLDERS[self.sourceBox.currentText()], filter="*.nwb")
        if filename!='':
            self.filename = filename
        else:
            self.filename = ''

    def build_cmd(self):
        return '%s %s %s' % (python_path,
                             self.process_script,
                             self.filename)
    
    def build_pdf(self):
        if self.filename!='':
            p = subprocess.Popen(self.build_cmd(), shell=True)
            print('"%s" launched as a subprocess' % self.build_cmd())
        else:
            print(' /!\ Need a valid folder !  /!\ ')

    def quit(self):
        QtWidgets.QApplication.quit()
        
    
def run(app, args=None, parent=None):
    return MainWindow(app, args=args,
                      parent=parent)
        
    
if __name__=='__main__':
    
    # filename = '/home/yann/DATA/Wild_Type/2021_03_11-17-13-03.nwb'
    
    app = QtWidgets.QApplication(sys.argv)
    main = run(app)
    sys.exit(app.exec_())
    
