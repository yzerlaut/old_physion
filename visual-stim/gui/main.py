import sys
import numpy as np
from PyQt5 import QtGui, QtWidgets, QtCore

import sys, os, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from psychopy_code.stimuli import visual_stim
from params_window import *
from default_params import STIMULI
import json

PROTOCOLS = ['Single-Stimulus', 'Stimuli-Sequence', 'Randomized-Sequence']

class Window(QtWidgets.QMainWindow):
    
    def __init__(self, app, parent=None):
        
        super(Window, self).__init__(parent)
        
        self.protocol = None # by default, can be loaded by the interface
        self.params_window = None
        
        # buttons and functions
        LABELS = ["i) Initialize", "r) Run", "s) Stop", "q) Quit"]
        FUNCTIONS = [self.initialize, self.run, self.stop, self.quit]
        button_length = 100
        
        self.setWindowTitle('Visual Stimulation Program')
        self.setGeometry(50, 50, 1.01*button_length*len(LABELS), 250)

        # protocol change
        label1 = QtWidgets.QLabel("/|===> Presentation <===|\\", self)
        label1.setMinimumWidth(320)
        label1.move(100, 50)
        self.cbp = QtWidgets.QComboBox(self)
        self.cbp.addItems(['']+PROTOCOLS)
        self.cbp.currentIndexChanged.connect(self.change_protocol)
        self.cbp.setMinimumWidth(250)
        self.cbp.move(70, 80)

        # stimulus pick
        label2 = QtWidgets.QLabel("  /|===> Stimulus <===|\\", self)
        label2.setMinimumWidth(330)
        label2.move(100, 110)
        self.cbs = QtWidgets.QComboBox(self)
        self.cbs.addItems(['']+list(STIMULI.keys()))
        self.cbs.currentIndexChanged.connect(self.change_stimulus)
        self.cbs.setMinimumWidth(250)
        self.cbs.move(70, 140)

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
            
        LABELS = ["o) Load Protocol", " Save Protocol", "Set folder"]
        FUNCTIONS = [self.load_protocol, self.save_protocol, self.set_folder]
        for func, label, shift, size in zip(FUNCTIONS, LABELS,\
                                            150*np.arange(len(LABELS)), [150, 150, 100]):
            btn = QtWidgets.QPushButton(label, self)
            btn.clicked.connect(func)
            btn.setMinimumWidth(size)
            btn.move(shift, 190)
            action = QtWidgets.QAction(label, self)
            if len(label.split(')'))>0:
                action.setShortcut(label.split(')')[0])
                action.triggered.connect(func)
                self.fileMenu.addAction(action)

        self.show()

        
    def initialize(self):
        self.statusBar.showMessage('[...] preparing stimulation')
        self.stim = visual_stim(protocol=self.cbp.currentText(),
                                stimulus=self.cbs.currentText())
        self.stim.build_protocol(a=0)
        self.statusBar.showMessage('stimulation ready. WAITING FOR THE USB TRIGGER !!')

        
    def run(self):
        self.statusBar.showMessage('[...] preparing stimulation')
        self.stim = visual_stim(protocol=self.cbp.currentText(),
                                stimulus=self.cbs.currentText())
        self.stim.build_protocol(a=0)
        self.stim.show()
    
    def stop(self):
        self.close()
        core.quit()
    
    def quit(self):
        sys.exit()
        
    def save_protocol(self):
        if self.params_window is not None:
            self.protocol = extract_params_from_window(self)
            with open('protocol.json', 'w') as fp:
                json.dump(self.protocol, fp)
                self.statusBar.showMessage('protocol saved as "protocol.json"')
        else:
            self.statusBar.showMessage('No protocol data available')
            
    def load_protocol(self):
        try:
            with open('protocol.json', 'r') as fp:
                self.protocol = json.load(fp)
            self.statusBar.showMessage('successfully loaded "protocol.json"')
            self.params_window = draw_window(self)
            self.params_window.show()
        except FileNotFoundError:
            self.statusBar.showMessage('protocol file not found !')

    def set_folder(self):
        pass

    def change_protocol(self):
        self.protocol = None
        self.params_window = draw_window(self)
        self.params_window.show()
        
    def change_stimulus(self):
        self.protocol = None
        self.params_window = draw_window(self)
        self.params_window.show()

    def create_params_window(self):
        window = QtWidgets.QDialog()

        
    def save_results(self):
        if 'NSI' in self.data:
            results_filename = '.'.join(self.filename.split('.')[:-1]) if '.' in self.filename else self.filename
            results_filename += '_NSI_'+datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')+'.h5'
            print(self.data.keys())
            to_save = {'validated_times': self.data['new_t'][self.data['NSI_validated']],
                       'validated_NSI':self.data['NSI'][self.data['NSI_validated']]}
            save_dict_to_hdf5(to_save, results_filename)
            self.statusBar.showMessage('Results of analysis saved as : '+results_filename)
        else:
            self.statusBar.showMessage('Need to perform analysis first...')
    
        
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = Window(app)
    sys.exit(app.exec_())
    

# def main():

#     app = QApplication(sys.argv)

#     w = QWidget()
#     w.resize(600, 300)
#     w.move(300, 300)
#     w.setWindowTitle('Visual Stimulation Program')
#     w.show()

#     sys.exit(app.exec_())


# if __name__ == '__main__':
#     main()
