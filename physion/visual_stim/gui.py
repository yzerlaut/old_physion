import sys, time
import numpy as np
from PyQt5 import QtGui, QtWidgets, QtCore

import sys, os, pathlib, json

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import create_day_folder, generate_filename_path

sys.path.append(str(pathlib.Path(__file__).resolve().parents[0]))
from psychopy_code.stimuli import build_stim
from default_params import STIMULI, PRESENTATIONS
from screens import SCREENS
from guiparts import *


class MainWindow(QtWidgets.QMainWindow):
    
    def __init__(self, app, args=None, parent=None):
        
        super(MainWindow, self).__init__(parent)
        
        self.protocol = None # by default, can be loaded by the interface
        self.experiment = {} # storing the specifics of an experiment
        self.stim, self.init, self.stop_flag = None, False, False
        self.screen = '' # N.B. self.screen is a string here, it is a dictionary in "stimuli.py"
        self.params_window = None
        self.protocol_folder = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'exp',
                                            'protocols')

        self.root_datafolder = args.root_datafolder
        self.demo = args.demo
        self.datafolder = mp_string('')
        
        # buttons and functions
        LABELS = ["i) Initialize", "r) Run", "s) Stop", "q) Quit"]
        FUNCTIONS = [self.initialize, self.run, self.stop, self.quit]
        button_length = 100
        
        self.setWindowTitle('Stimulation Design')
        self.setGeometry(450, 100, int(1.01*button_length*len(LABELS)), 310)

        # protocol change
        label1 = QtWidgets.QLabel("/|===> Presentation <===|\\", self)
        label1.setMinimumWidth(320)
        label1.move(100, 50)
        self.cbp = QtWidgets.QComboBox(self)
        self.cbp.addItems(['']+PRESENTATIONS)
        self.cbp.currentIndexChanged.connect(self.change_protocol)
        self.cbp.setMinimumWidth(250)
        self.cbp.move(70, 80)

        # stimulus pick
        label2 = QtWidgets.QLabel("   /|===> Stimulus <===|\\", self)
        label2.setMinimumWidth(330)
        label2.move(100, 110)
        self.cbs = QtWidgets.QComboBox(self)
        self.cbs.addItems(['']+list(STIMULI.keys()))
        self.cbs.currentIndexChanged.connect(self.change_stimulus)
        self.cbs.setMinimumWidth(250)
        self.cbs.move(70, 140)

        # screen pick
        label3 = QtWidgets.QLabel("     /|===>  Screen  <===|\\", self)
        label3.setMinimumWidth(320)
        label3.move(100, 170)
        self.cbsc = QtWidgets.QComboBox(self)
        self.cbsc.addItems(['']+list(SCREENS.keys()))
        self.cbsc.currentIndexChanged.connect(self.change_screen)
        self.cbsc.setMinimumWidth(250)
        self.cbsc.move(70, 200)

        mainMenu = self.menuBar()
        self.fileMenu = mainMenu.addMenu('&File')

        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage('...')
        
        for func, label, shift in zip(FUNCTIONS, LABELS,\
                                      button_length*np.arange(len(LABELS))):
            btn = QtWidgets.QPushButton(label, self)
            btn.clicked.connect(func)
            btn.setMinimumWidth(button_length)
            btn.move(shift, 20)
            action = QtWidgets.QAction(label, self)
            action.setShortcut(label.split(')')[0])
            action.triggered.connect(func)
            self.fileMenu.addAction(action)
            
        LABELS = ["Load Protocol [Ctrl+O]", " Save Protocol", "Set folders"]
        FUNCTIONS = [self.load_protocol, self.save_protocol, self.set_folders]
        for func, label, shift, size in zip(FUNCTIONS, LABELS,\
                                            150*np.arange(len(LABELS)), [150, 150, 100]):
            btn = QtWidgets.QPushButton(label, self)
            btn.clicked.connect(func)
            btn.setMinimumWidth(size)
            btn.move(shift, 250)
            action = QtWidgets.QAction(label, self)
            if len(label.split('['))>1:
                action.setShortcut(label.split('[')[1].replace(']',''))
                action.triggered.connect(func)
                self.fileMenu.addAction(action)

        self.show()

        
    def initialize(self):
        if (self.protocol is None) and (\
           (self.cbp.currentText()=='') or (self.cbs.currentText()=='')):
            self.statusBar.showMessage('Need to set parameters in the GUI or to load a protocol !')
        elif self.cbp.currentText()!='multiprotocol' and self.cbsc.currentText()!='':
            self.statusBar.showMessage('[...] preparing stimulation')
            self.protocol = extract_params_from_window(self)
            print(self.protocol)
            self.protocol['demo'] = True
            self.stim = build_stim(self.protocol)
            self.statusBar.showMessage('stimulation ready !')
            self.init = True
        else:
            self.statusBar.showMessage('init failed !')
            
        
    def run(self):
        self.stop_flag=False
        if (self.stim is None) or not self.init:
            self.statusBar.showMessage('Need to initialize the stimulation !')
        else:
            self.save_experiment()
            self.statusBar.showMessage('stimulation running [...]')
            self.stim.run(self)
            self.stim.close()
            self.init = False
    
    def stop(self):
        self.stop_flag=True
        self.statusBar.showMessage('stimulation stopped !')
        if self.stim is not None:
            self.stim.close()
            self.init = False
    
    def quit(self):
        if self.stim is not None:
            self.stim.quit()
        sys.exit()

    def save_experiment(self):
        full_exp = dict(**self.protocol, **self.experiment)
        filename = generate_filename_path(self.root_datafolder,
                                          filename='visual-stim', extension='.npz',
                                          with_screen_frames_folder=True)
        self.datafolder.set(os.path.dirname(filename))
        np.savez(filename, full_exp, allow_pickle=True)
        print('Stimulation data saved as: %s ' % filename)
        self.statusBar.showMessage('Stimulation data saved as: %s ' % filename)
        
        
    def save_protocol(self):
        if self.params_window is not None:
            self.protocol = extract_params_from_window(self)
            self.protocol['data-folder'] = self.datafolder.get()
            self.protocol['protocol-folder'] = self.protocol_folder
            self.protocol['Screen'] = self.screen
            filename = QtWidgets.QFileDialog.getSaveFileName(self, 'Save protocol file',
                                                             self.protocol_folder, "Protocol files (*.json)")
            if filename[0]!='':
                if len(filename[0].split('.json'))>0:
                    filename = filename[0]
                else:
                    filename = filename[0]+'.json'
                with open(filename, 'w') as fp:
                    json.dump(self.protocol, fp, indent=2)
                    self.statusBar.showMessage('protocol saved as "%s"' % filename)
            else:
                self.statusBar.showMessage('protocol file "%s" not valid' % filename)
        else:
            self.statusBar.showMessage('protocol file "%s" not valid' % filename)
            
    def load_protocol(self):
        filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Open protocol file', self.protocol_folder,"Protocol files (*.json)")
        try:
            with open(filename[0], 'r') as fp:
                self.protocol = json.load(fp)
            # self.protocol_folder = self.protocol['protocol-folder']
            self.screen = self.protocol['Screen']
            # update main window
            try:
                s1, s2, s3 = self.protocol['Presentation'], self.protocol['Stimulus'], self.protocol['Screen']
                self.cbp.setCurrentIndex(np.argwhere(s1==np.array(list(['']+PRESENTATIONS)))[0][0])
                self.cbs.setCurrentIndex(np.argwhere(s2==np.array(list(['']+list(STIMULI.keys()))))[0][0])
                self.cbsc.setCurrentIndex(np.argwhere(s3==np.array(list(['']+list(SCREENS.keys()))))[0][0])
            except KeyError:
                s1 = self.protocol['Presentation']
                self.cbp.setCurrentIndex(np.argwhere(s1==np.array(list(['']+PRESENTATIONS)))[0][0])
                
            self.statusBar.showMessage('successfully loaded "%s"' % filename[0])
            # draw params window
            self.params_window = draw_window(self, self.protocol)
            self.params_window.show()
        except FileNotFoundError:
            self.statusBar.showMessage('protocol file "%s" not found !' % filename[0])

    def set_folders(self):
        self.protocol_folder = str(QtWidgets.QFileDialog.getExistingDirectory(self,
                                                                              "Select Protocol Folder"))
        # self.datafolder = str(QtWidgets.QFileDialog.getExistingDirectory(self,
        #                                                                   "Select Data Folder"))
        # self.datafolder
        self.statusBar.showMessage('Protocol folder: "%s", Data folder "%s"' %\
                                   (self.protocol_folder, self.datafolder))

    def change_protocol(self):
        self.params_window = draw_window(self, None)
        self.params_window.show()
        
    def change_stimulus(self):
        self.params_window = draw_window(self, None)
        self.params_window.show()

    def change_screen(self):
        self.screen = self.cbsc.currentText()
        
    def create_params_window(self):
        window = QtWidgets.QDialog()


class mp_string:
    """ 
    dummy class behaving like: multiprocessing.Manager().manager.Value()
    """
    def __init__(self, string=''):
        self.value = string
    def get(self):
        return self.value
    def set(self, string):
        self.value = string
    
def run(app, args=None, parent=None):
    return MainWindow(app, args=args, parent=parent)


if __name__=='__main__':
    import tempfile

    import argparse, os
    parser=argparse.ArgumentParser(description="Experiment interface",
                       formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-rf', "--root_datafolder", type=str,
                        default=tempfile.gettempdir())
    parser.add_argument('-d', "--demo", action="store_true")
    args = parser.parse_args()
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow(app, args=args)
    sys.exit(app.exec_())
