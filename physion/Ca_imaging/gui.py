import sys, time, os, pathlib
from PyQt5 import QtGui, QtWidgets, QtCore

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import list_dayfolder, get_TSeries_folders
from misc.folders import FOLDERS

class MainWindow(QtWidgets.QMainWindow):
    
    def __init__(self, app,
                 args=None,
                 parent=None):
        """
        sampling in Hz
        """
        super(MainWindow, self).__init__()

        self.setGeometry(350, 700, 300, 300)
        # adding a "quit" keyboard shortcut
        self.quitSc = QtWidgets.QShortcut(QtGui.QKeySequence('Q'), self) # or 'Ctrl+Q'
        self.quitSc.activated.connect(self.quit)
            
        self.setWindowTitle('Physion -- Assembling')
        
        self.script = os.path.join(\
                str(pathlib.Path(__file__).resolve().parents[1]),\
                'script.sh')

        HEIGHT = 0

        HEIGHT += 10
        QtWidgets.QLabel("Root Data:", self).move(10, HEIGHT)
        self.cbc = QtWidgets.QComboBox(self)
        self.cbc.setMinimumWidth(150)
        self.cbc.move(100, HEIGHT)
        self.cbc.activated.connect(self.update_setting)
        self.cbc.addItems(FOLDERS.keys())

        HEIGHT += 30
        QtWidgets.QLabel("Root Imaging:", self).move(10, HEIGHT)
        self.cbc = QtWidgets.QComboBox(self)
        self.cbc.setMinimumWidth(150)
        self.cbc.move(100, HEIGHT)
        self.cbc.activated.connect(self.update_setting)
        self.cbc.addItems(FOLDERS.keys())
        
        HEIGHT += 40
        self.load = QtWidgets.QPushButton('Load data \u2b07', self)
        self.load.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.load.clicked.connect(self.load_folder)
        self.load.setMinimumWidth(200)
        self.load.move(50, HEIGHT)
        self.loadSc = QtWidgets.QShortcut(QtGui.QKeySequence('L'), self)
        self.loadSc.activated.connect(self.load_folder)

        HEIGHT += 40
        self.load = QtWidgets.QPushButton('Load imaging \u2b07', self)
        self.load.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.load.clicked.connect(self.load_folder)
        self.load.setMinimumWidth(200)
        self.load.move(50, HEIGHT)
        self.loadSc = QtWidgets.QShortcut(QtGui.QKeySequence('L'), self)
        self.loadSc.activated.connect(self.load_folder)

        HEIGHT += 50
        QtWidgets.QLabel("=> Setting :", self).move(10, HEIGHT)
        self.cbc = QtWidgets.QComboBox(self)
        self.cbc.setMinimumWidth(150)
        self.cbc.move(100, HEIGHT)
        self.cbc.activated.connect(self.update_setting)
        self.cbc.addItems(['standard', 'lightweight', 'full', 'NIdaq-only', 'custom'])

        HEIGHT +=40 
        s = QtWidgets.QLabel("CaImaging-Sampling (Hz)    ", self)
        s.move(10, HEIGHT)
        s.setMinimumWidth(200)
        self.PsamplingBox = QtWidgets.QLineEdit('', self)
        self.PsamplingBox.setText('1.0')
        self.PsamplingBox.setFixedWidth(40)
        self.PsamplingBox.move(200, HEIGHT)

        HEIGHT +=50 
        self.gen = QtWidgets.QPushButton('-=- Run -=-', self)
        self.gen.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.gen.clicked.connect(self.run)
        self.gen.setMinimumWidth(200)
        self.gen.move(50, HEIGHT)
        
        self.folder = ''
        self.show()

    def update_setting(self):
        pass
        if self.cbc.currentText()=='custom':
            print('kjshdf')

    
    def load_folder(self):

        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self,
                                                            "Open datafile (through metadata file) )",
                                                            FOLDERS[self.folderB.currentText()],
                                                            filter="*.npy")
        if filename!='':
            self.folder = os.path.dirname(filename)
        else:
            self.folder = ''
            

    def clean_folder(self):
        
        if len(self.folder[-8:].split('_'))==3:
            print(list_dayfolder(self.folder))
        else:
            print(self.folder)
            
    
    def run(self):
        pass
                
    def quit(self):
        QtWidgets.QApplication.quit()
        
def run(app, args=None, parent=None):
    return MainWindow(app,
                      args=args,
                      parent=parent)

if __name__=='__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = run(app)
    sys.exit(app.exec_())
        

