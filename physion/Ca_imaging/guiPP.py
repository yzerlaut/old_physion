import sys, time, os, pathlib, subprocess
from PyQt5 import QtGui, QtWidgets, QtCore

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import list_dayfolder, get_TSeries_folders
from assembling.tools import find_matching_CaImaging_data
from misc.folders import FOLDERS
from Ca_imaging.preprocessing import PREPROCESSING_SETTINGS

class MainWindow(QtWidgets.QMainWindow):
    
    def __init__(self, app,
                 args=None,
                 parent=None):
        """
        sampling in Hz
        """
        super(MainWindow, self).__init__()

        self.setGeometry(350, 470, 300, 220)
        # adding a "quit" and "load" keyboard shortcuts
        self.quitSc = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+Q'), self)
        self.quitSc.activated.connect(self.quit)
        self.loadSc = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+O'), self)
        self.loadSc.activated.connect(self.load_imaging)
            
        self.setWindowTitle('Ca-Preprocessing')
        
        self.process_script = os.path.join(str(pathlib.Path(__file__).resolve().parents[1]),
                                           'Ca_imaging',  'preprocessing.py')

        HEIGHT = 0

        HEIGHT += 10
        QtWidgets.QLabel("Root Folder:", self).move(10, HEIGHT)
        self.folderI = QtWidgets.QComboBox(self)
        self.folderI.setMinimumWidth(150)
        self.folderI.move(100, HEIGHT)
        self.folderI.activated.connect(self.update_setting)
        self.folderI.addItems(FOLDERS.keys())
        
        HEIGHT += 40
        self.load = QtWidgets.QPushButton('Upload (set of) TSeries \u2b07', self)
        self.load.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.load.clicked.connect(self.load_imaging)
        self.load.setMinimumWidth(200)
        self.load.move(50, HEIGHT)

        HEIGHT += 40
        QtWidgets.QLabel("=> Setting :", self).move(10, HEIGHT)
        self.cbc = QtWidgets.QComboBox(self)
        self.cbc.setMinimumWidth(150)
        self.cbc.move(100, HEIGHT)
        self.cbc.activated.connect(self.update_setting)
        self.cbc.addItems(list(PREPROCESSING_SETTINGS.keys()))

        HEIGHT +=40 
        self.gen = QtWidgets.QPushButton('-=- Run -=-', self)
        self.gen.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.gen.clicked.connect(self.run)
        self.gen.setMinimumWidth(200)
        self.gen.move(50, HEIGHT)
        
        HEIGHT +=40 
        self.gen = QtWidgets.QPushButton(' open Suite2p ', self)
        self.gen.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.gen.clicked.connect(self.open_suite2p)
        self.gen.setMinimumWidth(200)
        self.gen.move(50, HEIGHT)
        
        self.CMDS = []
        self.show()

    def update_setting(self):
        self.CMDS = []
        print('command set is reset !')
    
    def build_cmd(self, folder, key):
        return 'python %s --CaImaging_folder %s --setting_key %s -v' % (self.process_script, folder, key)
    
    def load_imaging(self):

        folder = QtWidgets.QFileDialog.getExistingDirectory(self,\
                                    "Choose datafolder",
                                    FOLDERS[self.folderI.currentText()])
        if folder!='':
            if ('Tseries' in str(folder)):
                self.CMDS.append(self.build_cmd(folder, self.cbc.currentText()))
            else:
                folders = get_TSeries_folders(folder)
                for f in folders:
                    self.CMDS.append(self.build_cmd(f, self.cbc.currentText()))
            for cmd in self.CMDS:
                print('"%s" added to command set' % cmd)

        if len(self.CMDS)==0:
            print('\n /!\ no "TSeries" folder found, set of command is empty ! \n')
                
    def run(self):
        if len(self.CMDS)==0:
            print('\n /!\ set of command is empty /!\  \n')
        for cmd in self.CMDS:
            p = subprocess.Popen(cmd, shell=True)
            print('"%s" launched as a subprocess' % cmd)

    def open_suite2p(self):
        p = subprocess.Popen('python -m suite2p',
                             shell=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
            
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
        

