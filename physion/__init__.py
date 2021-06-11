import sys, pathlib, os, subprocess
from PyQt5 import QtGui, QtCore, QtWidgets

sys.path.append(str(pathlib.Path(__file__).resolve().parent))
from misc.style import set_dark_style, set_app_icon
from misc.colors import build_dark_palette

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
try:
    from psychopy import visual, core, event, clock, monitors # some libraries from PsychoPy
    no_psychopy = False
except ModuleNotFoundError:
    print('Experiment & Visual-Stim modules disabled !')
    no_psychopy = True

CHILDREN_PROCESSES = []
class MainWindow(QtWidgets.QMainWindow):
    
    def __init__(self, app,
                 args=None,
                 button_height = 20):

        self.app = app
        self.args = args
        set_app_icon(app)
        super(MainWindow, self).__init__()
        self.setWindowTitle('Physion ')

        # buttons and functions
        LABELS = ["r) [R]un experiments",
                  "s) [S]timulus design",
                  "p) [P]upil preprocessing",
                  "f) [F]acemotion preprocessing",
                  "i) [I]maging preprocessing",
                  "e) [E]lectrophy preprocessing",
                  # "b) run [B]ash script",
                  "a) [A]ssemble data",
                  "c) Add [C]a2+ data",
                  "t) [T]ransfer data",
                  "v) [V]isualize data",
                  "n) summary [P]DF ",
                  "q) [Q]uit"]
        lmax = max([len(l) for l in LABELS])

        FUNCTIONS = [self.launch_exp,
                     self.launch_visual_stim,
                     # self.launch_organize,
                     self.launch_pupil,
                     self.launch_facemotion,
                     self.launch_CaProprocessing,
                     self.launch_electrophy,
                     # self.launch_bash_script,
                     self.launch_assembling,
                     self.launch_CaAddition,
                     self.launch_transfer,
                     self.launch_visualization,
                     self.summary_pdf,
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
        if not no_psychopy:
            from physion.exp.gui import run as RunExp
            child = RunExp(self.app, self.args)
            CHILDREN_PROCESSES.append(child)
        else:
            self.statusBar.showMessage('Module cant be launched, PsychoPy is missing !')
            
        
    def launch_facemotion(self):
        from physion.facemotion.gui import run as RunFacemotion
        child = RunFacemotion(self.app, self.args)
        CHILDREN_PROCESSES.append(child)
        
    def launch_visual_stim(self):
        from physion.visual_stim.gui import run as RunVisualStim
        child = RunVisualStim(self.app, self.args)
        CHILDREN_PROCESSES.append(child)
        
    def launch_assembling(self):
        from physion.assembling.gui import run as RunAssembling
        child = RunAssembling(self.app, self.args)
        CHILDREN_PROCESSES.append(child)
        
    def launch_transfer(self):
        from physion.transfer.gui import run as RunTransfer
        child = RunTransfer(self.app, self.args)
        CHILDREN_PROCESSES.append(child)
        
    def launch_pupil(self):
        from physion.pupil.gui import run as RunPupilGui
        child = RunPupilGui(self.app, self.args)
        CHILDREN_PROCESSES.append(child)
        
    def launch_CaProprocessing(self):
        from physion.Ca_imaging.guiPP import run as RunCaPreprocessing
        child = RunCaPreprocessing(self.app, self.args)
        CHILDREN_PROCESSES.append(child)

    def launch_CaAddition(self):
        from physion.Ca_imaging.guiAdd import run as RunCaAddition
        child = RunCaAddition(self.app, self.args)
        CHILDREN_PROCESSES.append(child)
        
    def launch_electrophy(self):
        self.statusBar.showMessage('Electrophy module not implemented yet')

    # def launch_bash_script(self):
    #     self.statusBar.showMessage('Batch processing launched !')
    #     import subprocess
    #     script = os.path.join(str(pathlib.Path(__file__).resolve().parent),'script.sh')
    #     fileS = open(script, 'r')
    #     Lines = fileS.readlines()
    #     # subprocess.run('bash %s' % script, shell=True)
    #     PROCESSES = []
    #     for i, l in enumerate(Lines):
    #         print(""" %i) launching process:
    #         %s """ % (i+1, l))
    #         subprocess.Popen(l, shell=True,
    #                          stdout=subprocess.PIPE,
    #                          stderr=subprocess.STDOUT)
    #     # then clean batch file
    #     fileS.close()
    #     open(script, 'w').close()

        
    def launch_visualization(self):
        self.statusBar.showMessage('Loading Visualization Module [...]')
        from physion.dataviz.gui import run as RunAnalysisGui
        self.child = RunAnalysisGui(self.app, self.args,
                                    raw_data_visualization=True)
        
    def summary_pdf(self):
        from physion.misc.notebook import run as RunNotebook
        self.child = RunNotebook(self.app, self.args)
        
    def quit(self):
        QtWidgets.QApplication.quit()
        
def run(args):
    # Always start by initializing Qt (only once per application)
    app = QtWidgets.QApplication(sys.argv)
    build_dark_palette(app)
    # set_dark_style(app)
    set_app_icon(app)
    GUI = MainWindow(app, args=args)
    sys.exit(app.exec_())
