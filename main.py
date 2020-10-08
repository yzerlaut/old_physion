import sys, pathlib
from PyQt5 import QtGui, QtWidgets, QtCore

sys.path.append(str(pathlib.Path(__file__).resolve()))
from analysis import guiparts

class MainWindow(QtWidgets.QMainWindow):
    
    def __init__(self, app,
                 button_height = 20):

        super(MainWindow, self).__init__()
        self.setWindowTitle('Physiology of Visual Circuits')

        # buttons and functions
        LABELS = ["e) launch Experiment",
                  "v) prepare Visual stim. protocols",
                  "c) Compress data",
                  "t) Transfer data",
                  "p) preprocess Pupil",
                  "i) preprocess ca2+ Imaging",
                  "a) Analyze data",
                  "n) lab Notebook ",
                  "q) Quit"]
        lmax = max([len(l) for l in LABELS])
        # LABELS = [l.replace(') ', ') '+(lmax-int(len(l)/2))*' ') for l in LABELS]
        FUNCTIONS = [self.launch_exp,
                     self.launch_visual_stim,
                     self.launch_compress,
                     self.launch_transfer,
                     self.launch_pupil,
                     self.launch_caimaging,
                     self.launch_analysis,
                     self.launch_notebook,
                     self.quit]
        
        self.setGeometry(50, 50, 300, 50*len(LABELS))
        
        mainMenu = self.menuBar()
        self.fileMenu = mainMenu.addMenu('')

        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage('select a module')
        
        for func, label, ishift in zip(FUNCTIONS, LABELS,\
                                       range(len(LABELS))):
            btn = QtWidgets.QPushButton(label, self)
            btn.clicked.connect(func)
            btn.setMinimumHeight(button_height)
            btn.setMinimumWidth(250)
            btn.move(25, 30+2*button_height*ishift)
            action = QtWidgets.QAction(label, self)
            action.setShortcut(label.split(')')[0])
            action.triggered.connect(func)
            self.fileMenu.addAction(action)
            
        self.show()

    def launch_exp(self):
        from exp.gui import MainWindow as ExpMainWindow
        self.child = ExpMainWindow()
    def launch_visual_stim(self):
        from visual_stim.gui.main import Window as VisualMainWindow
        self.child = VisualMainWindow(self.app)
    def launch_compress(self):
        pass
        # from assembling.compress import MainWindow as CompressMainWindow
        # self.child = CompressMainWindow(self.app)
    def launch_transfer(self):
        from assembling.transfer import MainWindow as TransferMainWindow
        self.child = TransferMainWindow(self.app)
    def launch_pupil(self):
        self.statusBar.showMessage('Loading Pupil-Tracking Module [...]')
        self.show()
        from pupil.gui import MainWindow as PupilMainWindow
        self.child = PupilMainWindow()
    def launch_caimaging(self):
        pass
    def launch_analysis(self):
        self.statusBar.showMessage('Loading Analysis Module [...]')
        self.show()
        from analysis.gui import MainWindow as AnalysisMainWindow
        self.child = AnalysisMainWindow()
    def launch_notebook(self):
        pass
    def quit(self):
        QtWidgets.QApplication.quit()
        

app = QtWidgets.QApplication(sys.argv)
guiparts.build_dark_palette(app)
main = MainWindow(app)
sys.exit(app.exec_())
