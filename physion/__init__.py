import sys, pathlib, os, subprocess
from PyQt5 import QtGui, QtCore, QtWidgets

sys.path.append(str(pathlib.Path(__file__).resolve().parent))
from misc.style import set_dark_style, set_app_icon
from misc.colors import build_dark_palette

if not sys.argv[-1]=='no-stim':
    from psychopy import visual, core, event, clock, monitors # some libraries from PsychoPy
else:
    print('Experiment & Visual-Stim modules disabled !')

class MainWindow(QtWidgets.QMainWindow):
    
    def __init__(self, app,
                 args=None,
                 button_height = 20):

        self.app = app
        self.args = args
        set_app_icon(app)
        super(MainWindow, self).__init__()
        self.setWindowTitle('Physion - Vision Physiology ')

        # buttons and functions
        LABELS = ["r) [R]un experiments",
                  "s) [S]timulus design",
                  # "o) Assemble and [O]rganize data",
                  # "t) [T]ransfer data",
                  "p) [P]upil preprocessing",
                  "w) [W]hisking preprocessing",
                  "i) [I]maging preprocessing (suite2p)",
                  "e) [E]lectrophy preprocessing",
                  "b) launch [B]atch processing",
                  "v) [V]isualize data",
                  # "a) [A]nalyze data",
                  "n) lab [N]otebook ",
                  "q) [Q]uit"]
        lmax = max([len(l) for l in LABELS])

        FUNCTIONS = [self.launch_exp,
                     self.launch_visual_stim,
                     # self.launch_organize,
                     # self.launch_transfer,
                     self.launch_pupil,
                     self.launch_whisking,
                     self.launch_caimaging,
                     self.launch_electrophy,
                     self.launch_batch,
                     self.launch_visualization,
                     # self.launch_analysis,
                     self.launch_notebook,
                     self.quit]
        
        self.setGeometry(50, 100, 300, 46*len(LABELS))
        
        mainMenu = self.menuBar()
        self.fileMenu = mainMenu.addMenu('')

        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage('select a module')
        
        for func, label, ishift in zip(FUNCTIONS, LABELS,
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
        from physion.exp.gui import run as RunExp
        self.child = RunExp(self.app, self.args)
        
    def launch_whisking(self):
        p = subprocess.Popen('python -m facemap', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        
    def launch_visual_stim(self):
        from physion.visual_stim.gui import run as RunVisualStim
        self.child = RunVisualStim(self.app, self.args)
        
    def launch_organize(self):
        from physion.assemble.gui import run as RunOrganize
        self.child = RunOrganize(self.app, self.args)
        
    def launch_transfer(self):
        self.statusBar.showMessage('Transfer module not implemented yet')
        
    def launch_pupil(self):
        self.statusBar.showMessage('Loading Pupil-Tracking Module [...]')
        from physion.pupil.gui import run as RunPupilGui
        self.child = RunPupilGui(self.app, self.args)
        
    def launch_caimaging(self):
        # p = subprocess.Popen('conda activate suite2p; python -m suite2p',
        p = subprocess.Popen('python -m suite2p',
                             shell=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)

    def launch_electrophy(self):
        self.statusBar.showMessage('Electrophy module not implemented yet')

    def launch_batch(self):
        self.statusBar.showMessage('Batch processing launched !')
        import subprocess
        script = os.path.join(str(pathlib.Path(__file__).resolve().parent),'script.sh')
        fileS = open(script, 'r')
        Lines = fileS.readlines()
        # subprocess.run('bash %s' % script, shell=True)
        PROCESSES = []
        for i, l in enumerate(Lines):
            print(""" %i) launching process:
            %s """ % (i+1, l))
            subprocess.Popen(l, shell=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        # then clean batch file
        fileS.close()
        open(script, 'w').close()

        
    def launch_visualization(self):
        self.statusBar.showMessage('Loading Visualization Module [...]')
        from physion.dataviz.gui import run as RunAnalysisGui
        self.child = RunAnalysisGui(self.app, self.args,
                                    raw_data_visualization=True)
        
    def launch_analysis(self):
        from physion.dataviz.gui import run as RunAnalysisGui
        self.child = RunAnalysisGui(self.app, self.args)
        pass
    
    def launch_notebook(self):
        self.statusBar.showMessage('Notebook module not implemented yet')
        
    def quit(self):
        QtWidgets.QApplication.quit()
        
def run(args):
    # Always start by initializing Qt (only once per application)
    app = QtWidgets.QApplication(sys.argv)
    build_dark_palette(app)
    # set_dark_style(app)
    set_app_icon(app)
    GUI = MainWindow(app)
    sys.exit(app.exec_())
