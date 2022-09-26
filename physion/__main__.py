import sys, pathlib, os, subprocess, argparse
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


# then physion specific modules
import analysis, dataviz, visual_stim, assembling, intrinsic


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
                  "c) [C]a2+ imaging preprocessing",
                  "e) [E]lectrophy preprocessing",
                  "a) [A]ssemble data",
                  "t) [T]ransfer data",
                  "v) [V]isualize data",
                  "n) launch [Notebook] ",
                  "i) [I]ntrinsic Imaging",
                  "q) [Q]uit"]
        lmax = max([len(l) for l in LABELS])

        FUNCTIONS = [self.launch_exp,
                     self.launch_visual_stim,
                     self.launch_pupil,
                     self.launch_facemotion,
                     self.launch_CaProprocessing,
                     self.launch_electrophy,
                     self.launch_assembling,
                     self.launch_transfer,
                     self.launch_visualization,
                     self.launch_notebook,
                     self.launch_intrinsic,
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
            child = RunExp(self.app, demo=False)
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
        child1 = RunAssembling(self.app, self.args)
        CHILDREN_PROCESSES.append(child1)
        from physion.Ca_imaging.guiAdd import run as RunCaAddition
        child2 = RunCaAddition(self.app, self.args)
        CHILDREN_PROCESSES.append(child2)
        
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

    def launch_intrinsic(self):
        from physion.intrinsic.gui import run as RunIntrinsic
        child = RunIntrinsic(self.app, self.args)
        CHILDREN_PROCESSES.append(child)
        
    def launch_electrophy(self):
        self.statusBar.showMessage('Electrophy module not implemented yet')

    def launch_visualization(self):
        self.statusBar.showMessage('Loading Visualization Module [...]')
        from physion.dataviz.gui import run as RunAnalysisGui
        self.child = RunAnalysisGui(self.app, self.args,
                                    raw_data_visualization=True)
        
    def launch_notebook(self):
        import subprocess
        from physion.misc.folders import python_path
        nb_dir = os.path.expanduser('~')
        nb_dirs = [os.path.join(os.path.expanduser('~'), 'physion', 'notebooks'),
                   os.path.join(os.path.expanduser('~'),'work', 'physion', 'notebooks')]
        for d in nb_dirs:
            if os.path.isdir(d):
                nb_dir = d
        cmd = '%s notebook %s' % (python_path.replace('python', 'jupyter'), d)
        os.system(cmd)
        
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



if __name__=='__main__':

    import argparse, os
    parser=argparse.ArgumentParser(description="Physion",
                       formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-rf', "--root_datafolder", type=str,
                        default=os.path.join(os.path.expanduser('~'), 'DATA'))
    parser.add_argument('-f', "--file")
    parser.add_argument('-sd', "--stim_demo", action="store_true")
    parser.add_argument('-d', "--demo", action="store_true")
    args = parser.parse_args()
    
    if args.stim_demo:

        if os.path.isfile(args.file):

            # load protocol
            with open(args.file, 'r') as fp:
                protocol = json.load(fp)
            protocol['demo'] = True

            # launch protocol
            stim = visual_stim.stimuli.build_stim(protocol)
            parent = visual_stim.stimuli.dummy_parent()
            stim.run(parent)
            stim.close()

        else:
            print('Need to provide a valid (json) stimulus file as a "--file" argument ! ')

    else:
        run(args)


