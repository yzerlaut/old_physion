import sys, time, os, pathlib
from PyQt5 import QtGui, QtWidgets, QtCore

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
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

        self.load = QtWidgets.QPushButton(' [L]oad root folder  \u2b07', self)
        self.load.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.load.clicked.connect(self.load_folder)
        self.load.setMinimumWidth(350)
        self.load.move(100, 25)
        self.loadSc = QtWidgets.QShortcut(QtGui.QKeySequence('L'), self)
        self.loadSc.activated.connect(self.load_folder)
        
        self.bCa = QtWidgets.QCheckBox("Pre-process Ca-Imaging data", self)
        self.bCa.setMinimumWidth(350)
        self.bCa.setChecked(True)
        self.bCa.move(50, 80)

        self.bMove = QtWidgets.QCheckBox("move Ca-imaging data to corresponding folder", self)
        self.bMove.setMinimumWidth(400)
        self.bMove.setChecked(True)
        self.bMove.move(50, 120)
        
        self.bFace = QtWidgets.QCheckBox("convert FaceCamera data to mp4 movie", self)
        self.bFace.setMinimumWidth(350)
        self.bFace.setChecked(True)
        self.bFace.move(50, 160)
        
        self.gen = QtWidgets.QPushButton(' [G]enerate script ', self)
        self.gen.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.gen.clicked.connect(self.gen_script)
        self.gen.setMinimumWidth(350)
        self.gen.move(100, 300)
        self.genSc = QtWidgets.QShortcut(QtGui.QKeySequence('G'), self)
        self.genSc.activated.connect(self.gen_script)

        
        self.datafolder = ''
        self.show()

    def load_folder(self):
        self.datafolder = QtWidgets.QFileDialog.getExistingDirectory(self,\
                                     "Choose folder", os.path.expanduser('~'))

    def gen_script(self):
        with open(self.script, 'w') as f:
            if self.bCa.isChecked():
                print('ok')
                # f.write('conda activate suite2p \n')
                # folders = list_TSeries_folder(self.datafolder)
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
        

