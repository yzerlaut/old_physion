import sys, time, os, pathlib
from PyQt5 import QtGui, QtWidgets, QtCore

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import list_dayfolder
from organize.compress import compress_datafolder
from organize.process import list_TSeries_folder
from misc.style import set_dark_style, set_app_icon

class MainWindow(QtWidgets.QMainWindow):
    
    def __init__(self, parent=None):
        """
        sampling in Hz
        """
        super(MainWindow, self).__init__()

        self.setGeometry(100,100,550,400)
        # adding a "quit" keyboard shortcut
        self.quitSc = QtWidgets.QShortcut(QtGui.QKeySequence('Q'), self) # or 'Ctrl+Q'
        self.quitSc.activated.connect(self.quit)
            
        self.setWindowTitle('preprocess and [re]organize data')

        self.script = os.path.join(\
                str(pathlib.Path(__file__).resolve().parents[1]),\
                'script.sh')

        self.load = QtWidgets.QPushButton(' [L]oad folder  \u2b07', self)
        self.load.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.load.clicked.connect(self.load_folder)
        self.load.setMinimumWidth(350)
        self.load.move(100, 25)
        self.loadSc = QtWidgets.QShortcut(QtGui.QKeySequence('L'), self)
        self.loadSc.activated.connect(self.load_folder)
        
        self.clean = QtWidgets.QPushButton('   [C]lean up folder', self)
        self.clean.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.clean.clicked.connect(self.clean_folder)
        self.clean.setMinimumWidth(350)
        self.clean.move(100, 60)
        self.cleanSc = QtWidgets.QShortcut(QtGui.QKeySequence('C'), self)
        self.cleanSc.activated.connect(self.clean_folder)
        
        self.bCa = QtWidgets.QCheckBox("Pre-process Ca-Imaging data", self)
        self.bCa.setMinimumWidth(350)
        self.bCa.move(50, 120)
        
        self.bMove = QtWidgets.QCheckBox("move Ca-imaging data to corresponding folder", self)
        self.bMove.setMinimumWidth(400)
        self.bMove.move(50, 160)
        
        self.bFace = QtWidgets.QCheckBox("convert FaceCamera data to mp4 movie", self)
        self.bFace.setMinimumWidth(350)
        self.bFace.move(50, 200)
        
        self.gen = QtWidgets.QPushButton(' [A]dd to script ', self)
        self.gen.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.gen.clicked.connect(self.gen_script)
        self.gen.setMinimumWidth(350)
        self.gen.move(100, 300)
        self.genSc = QtWidgets.QShortcut(QtGui.QKeySequence('A'), self)
        self.genSc.activated.connect(self.gen_script)

        # default activated options
        # for b in [self.bCa, self.bMove, self.bFace]:
        for b in [self.bFace]:
            b.setChecked(True)
        
        self.folder = ''
        self.show()

    def load_folder(self):
        df = QtWidgets.QFileDialog.getExistingDirectory(self,\
                                "Choose folder", os.path.expanduser('~'))
        if df!='':
            self.folder = df
        else:
            pass

    def clean_folder(self):
        
        if len(self.folder[-8:].split('_'))==3:
            print(list_dayfolder(self.folder))
        else:
            print(self.folder)
    
    def gen_script(self):
        with open(self.script, 'w') as f:
            if self.bCa.isChecked():
                print('ok')
                # f.write('conda activate suite2p \n')
                # folders = list_TSeries_folder(self.folder)
                # for fn in folders:
                #     print(fn)
            if self.bFace.isChecked():
                print('ok')
                
    def quit(self):
        QtWidgets.QApplication.quit()
        
def run(app):
    set_app_icon(app)
    return MainWindow(app)

if __name__=='__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = run(app)
    sys.exit(app.exec_())
        

