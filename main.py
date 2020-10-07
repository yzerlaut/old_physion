import sys, pathlib
from PyQt5 import QtGui, QtWidgets, QtCore

sys.path.append(str(pathlib.Path(__file__).resolve()))


class MasterWindow(QtWidgets.QMainWindow):
    
    def __init__(self, app,
                 parent=None,
                 button_height = 20):

        self.app = app
        super(MasterWindow, self).__init__(parent)
        self.setWindowTitle('Physiology of Visual Circuits')

        # buttons and functions
        LABELS = ["e) launch Experiment",
                  "v) prepare Visual stim. protocols",
                  "c) Compress data",
                  "t) Transfer data",
                  "p) preprocess Pupil",
                  "a) Analyze data",
                  "q) Quit"]
        lmax = max([len(l) for l in LABELS])
        # LABELS = [l.replace(') ', ') '+(lmax-int(len(l)/2))*' ') for l in LABELS]
        FUNCTIONS = [self.launch_exp,
                     self.launch_visual_stim,
                     self.launch_compress,
                     self.launch_transfer,
                     self.launch_pupil,
                     self.launch_analysis,
                     self.quit]
        
        self.setGeometry(50, 50, 300, 50*len(LABELS))
        
        mainMenu = self.menuBar()
        self.fileMenu = mainMenu.addMenu('')

        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage('ready for initialization/analysis')
        
        for func, label, ishift in zip(FUNCTIONS, LABELS,\
                                       range(len(LABELS))):
            btn = QtWidgets.QPushButton(label, self)
            btn.clicked.connect(func)
            btn.setMinimumHeight(button_height)
            btn.setMinimumWidth(250)
            btn.move(25, 20+2*button_height*ishift)
            action = QtWidgets.QAction(label, self)
            action.setShortcut(label.split(')')[0])
            action.triggered.connect(func)
            self.fileMenu.addAction(action)
            
        self.show()

    def launch_exp(self):
        from protocols.exp import MasterWindow as ExpMasterWindow
        self.child = ExpMasterWindow(self.app)
    def launch_visual_stim(self):
        from visual_stim.gui.main import Window as VisualMasterWindow
        self.child = VisualMasterWindow(self.app)
    def launch_compress(self):
        pass
        # from assembling.compress import MainWindow as CompressMasterWindow
        # self.child = CompressMasterWindow(self.app)
    def launch_transfer(self):
        from assembling.transfer import MainWindow as TransferMasterWindow
        self.child = TransferMasterWindow(self.app)
    def launch_pupil(self):
        self.statusBar.showMessage('Loading Pupil-Tracking Module [...]')
        from pupil.gui import MainW as PupilMasterWindow
        self.child = PupilMasterWindow(self.app)
    def launch_analysis(self):
        self.statusBar.showMessage('Loading Analysis Module [...]')
        self.show()
        from analysis.gui import MasterWindow as AnalysisMasterWindow
        self.child = AnalysisMasterWindow(self.app)
    def quit(self):
        QtWidgets.QApplication.quit()
        

app = QtWidgets.QApplication(sys.argv)
main = MasterWindow(app)
sys.exit(app.exec_())
