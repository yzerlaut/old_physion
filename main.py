# Generate a executable on Windows by making a shortcut of props:
# - Target:
#   C:\ProgramData\Anaconda3\python.exe main.py
# - Start in:
#   C:\Users\VisualNeuroscientist\work\cortical-physio-icm

import sys, pathlib, os
from PyQt5 import QtGui, QtCore, QtWidgets

sys.path.append(str(pathlib.Path(__file__).resolve()))
from misc.style import set_dark_style, set_app_icon

if not sys.argv[-1]=='no-stim':
    from psychopy import visual, core, event, clock, monitors # some libraries from PsychoPy
else:
    print('Experiment & Visual-Stim modules disabled !')

class MainWindow(QtWidgets.QMainWindow):
    
    def __init__(self, app,
                 button_height = 20):

        self.app = app
        set_app_icon(app)
        super(MainWindow, self).__init__()
        self.setWindowTitle('Physiology of Visual Circuits    ')

        # buttons and functions
        LABELS = ["r) [R]un experiment",
                  "h) [H]habituation protocols",
                  "s) prepare visual [S]timulation",
                  "c) reformat/[C]ompress data",
                  "t) [T]ransfer data",
                  "p) [P]upil preprocessing",
                  "i) [I]maging preprocessing",
                  "e) [E]lectrophy preprocessing",
                  "v) [V]isualize data",
                  "a) [A]nalyze data",
                  "n) lab [N]otebook ",
                  "q) [Q]uit"]
        lmax = max([len(l) for l in LABELS])
        # LABELS = [l.replace(') ', ') '+(lmax-int(len(l)/2))*' ') for l in LABELS]
        FUNCTIONS = [self.launch_exp,
                     self.launch_habituation,
                     self.launch_visual_stim,
                     self.launch_compress,
                     self.launch_transfer,
                     self.launch_pupil,
                     self.launch_caimaging,
                     self.launch_electrophy,
                     self.launch_visualization,
                     self.launch_analysis,
                     self.launch_notebook,
                     self.quit]
        
        self.setGeometry(50, 50, 300, 47*len(LABELS))
        
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
        from exp.gui import run
        self.child = run(self.app)
    def launch_habituation(self):
        self.statusBar.showMessage('Habituation module not implemented yet')
    def launch_visual_stim(self):
        from visual_stim.gui import run as RunVisualStim
        self.child = RunVisualStim(self.app)
    def launch_compress(self):
        self.statusBar.showMessage('Compression module not implemented yet')
    def launch_transfer(self):
        self.statusBar.showMessage('Transfer module not implemented yet')
    def launch_pupil(self):
        self.statusBar.showMessage('Loading Pupil-Tracking Module [...]')
        from pupil.gui import run as RunPupilGui
        self.child = RunPupilGui(self.app)
    def launch_caimaging(self):
        self.statusBar.showMessage('Ca-Imaging module not implemented yet')
    def launch_electrophy(self):
        self.statusBar.showMessage('Electrophy module not implemented yet')
    def launch_visualization(self):
        self.statusBar.showMessage('Loading Visualization Module [...]')
        from analysis.gui import run as RunAnalysisGui
        self.child = RunAnalysisGui(self.app,
                                    raw_data_visualization=True)
    def launch_analysis(self):
        from analysis.gui import run as RunAnalysisGui
        self.child = RunAnalysisGui(self.app)
    def launch_notebook(self):
        self.statusBar.showMessage('Notebook module not implemented yet')
    def quit(self):
        QtWidgets.QApplication.quit()
        
def run():
    # Always start by initializing Qt (only once per application)
    app = QtWidgets.QApplication(sys.argv)
    # set_dark_style(app)
    # set_app_icon(app)
    GUI = MainWindow(app)
    ret = app.exec_()
    sys.exit(ret)

if __name__=='__main__':
    run()

