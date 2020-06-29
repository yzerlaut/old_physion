import sys, time, tempfile
import numpy as np
from PyQt5 import QtGui, QtWidgets, QtCore

import sys, os, pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import create_day_folder, generate_filename_path
from visual_stim.psychopy_code.stimuli import build_stim
from visual_stim.default_params import SETUP

class MasterWindow(QtWidgets.QMainWindow):
    
    def __init__(self, app,
                 parent=None,
                 button_length = 100):
        
        super(MasterWindow, self).__init__(parent)
        
        self.protocol, self.protocol_folder = None, os.path.join('visual_stim', 'protocols')
        self.experiment = {} # storing the specifics of an experiment
        
        self.stim, self.init, self.setup, self.stop_flag = None, False, SETUP[0], False
        self.params_window = None
        self.data_folder = tempfile.gettempdir()
        self.protocol_folder = './'
        
        self.setWindowTitle('Master Program -- Physiology of Visual Circuits')
        self.setGeometry(50, 50, 500, 300)

        # buttons and functions
        LABELS = ["i) Initialize", "r) Run", "s) Stop", "q) Quit"]
        FUNCTIONS = [self.initialize, self.run, self.stop, self.quit]
        
        # protocol choice
        QtWidgets.QLabel(" /|=>  Protocol <=|\\", self).move(30, 80)
        self.cbp = QtWidgets.QComboBox(self)
        self.cbp.addItems(self.protocol_list)
        self.cbp.currentIndexChanged.connect(self.change_protocol)
        self.cbp.setMinimumWidth(200)
        self.cbp.move(150, 80)
        self.pbtn = QtWidgets.QPushButton('Set folder', self)
        self.pbtn.clicked.connect(self.set_protocol_folder)
        self.pbtn.move(370, 80)


        # # stimulus pick
        QtWidgets.QLabel("  /|=>  Setup <=|\\", self).move(30, 120)
        # label2 = QtWidgets.QLabel("   /|===> Stimulus <===|\\", self)
        # label2.setMinimumWidth(330)
        # label2.move(100, 110)
        # self.cbs = QtWidgets.QComboBox(self)
        # self.cbs.addItems(['']+list(STIMULI.keys()))
        # self.cbs.currentIndexChanged.connect(self.change_stimulus)
        # self.cbs.setMinimumWidth(250)
        # self.cbs.move(70, 140)

        # # setup pick
        # label3 = QtWidgets.QLabel("     /|===>  Setup  <===|\\", self)
        # label3.setMinimumWidth(320)
        # label3.move(100, 170)
        # self.cbst = QtWidgets.QComboBox(self)
        # self.cbst.addItems(SETUP)
        # self.cbst.currentIndexChanged.connect(self.change_setup)
        # self.cbst.setMinimumWidth(250)
        # self.cbst.move(70, 200)

        mainMenu = self.menuBar()
        self.fileMenu = mainMenu.addMenu('')

        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage('choose and initialize an experiment to start...')
        
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
            
        LABELS = ["o) Load Protocol", " Save Protocol", "Set folders"]
        FUNCTIONS = [self.load_protocol, self.save_protocol, self.set_folders]
        for func, label, shift, size in zip(FUNCTIONS, LABELS,\
                                            150*np.arange(len(LABELS)), [150, 150, 100]):
            btn = QtWidgets.QPushButton(label, self)
            btn.clicked.connect(func)
            btn.setMinimumWidth(size)
            btn.move(shift, 250)
            action = QtWidgets.QAction(label, self)
            if len(label.split(')'))>0:
                action.setShortcut(label.split(')')[0])
                action.triggered.connect(func)
                self.fileMenu.addAction(action)

        self.show()

    
    def initialize(self):
        if (self.protocol is None) and (\
           (self.cbp.currentText()=='') or (self.cbs.currentText()=='')):
            self.statusBar.showMessage('Need to set parameters in the GUI or to load a protocol !')
        else:
            self.statusBar.showMessage('[...] preparing stimulation')
            self.protocol = extract_params_from_window(self)
            self.stim = build_stim(self.protocol)
            # self.statusBar.showMessage('stimulation ready. WAITING FOR THE USB TRIGGER !!')
            self.statusBar.showMessage('stimulation ready !')
            self.init = True
        
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

    def view_data(self):
        pass
    
    def save_experiment(self):
        full_exp = dict(**self.protocol, **self.experiment)
        create_day_folder(self.data_folder)
        filename = generate_filename_path(self.data_folder, extension='.npz')
        np.savez(filename, full_exp, allow_pickle=True)
        print('Stimulation data saved as: %s ' % filename)
        self.statusBar.showMessage('Stimulation data saved as: %s ' % filename)

    def get_protocol_list(self):
        os.listdir(self.protocol_folder):

        self.protocol_list = get_protocol_list(self)
        
    def set_protocol_folder(self):
        pass
        
    def save_protocol(self):
        if self.params_window is not None:
            self.protocol = extract_params_from_window(self)
            self.protocol['data-folder'] = self.data_folder
            self.protocol['protocol-folder'] = self.protocol_folder
            self.protocol['Setup'] = self.setup
            with open('protocol.json', 'w') as fp:
                json.dump(self.protocol, fp, indent=2)
                self.statusBar.showMessage('protocol saved as "protocol.json"')
        else:
            self.statusBar.showMessage('No protocol data available')
            
    def load_protocol(self):
        filename = 'protocol.json'
        try:
            with open(filename, 'r') as fp:
                self.protocol = json.load(fp)
            self.data_folder = self.protocol['data-folder']
            self.protocol_folder = self.protocol['protocol-folder']
            self.setup = self.protocol['Setup']
            # update main window
            s1, s2, s3 = self.protocol['Presentation'], self.protocol['Stimulus'], self.protocol['Setup']
            self.cbp.setCurrentIndex(np.argwhere(s1==np.array(list(['']+PRESENTATIONS)))[0][0])
            self.cbs.setCurrentIndex(np.argwhere(s2==np.array(list(['']+list(STIMULI.keys()))))[0][0])
            self.cbst.setCurrentIndex(np.argwhere(s3==np.array(SETUP))[0][0])
            self.statusBar.showMessage('successfully loaded "%s"' % filename)
            # draw params window
            self.params_window = draw_window(self, self.protocol)
            # self.params_window = draw_window(self, self.protocol)
            self.params_window.show()
        except FileNotFoundError:
            self.statusBar.showMessage('protocol file "%s" not found !' % filename)

    def set_folders(self):
        self.protocol_folder = str(QtWidgets.QFileDialog.getExistingDirectory(self,
                                                                              "Select Protocol Folder"))
        self.data_folder = str(QtWidgets.QFileDialog.getExistingDirectory(self,
                                                                          "Select Data Folder"))
        self.statusBar.showMessage('Protocol folder: "%s", Data folder "%s"' %\
                                   (self.protocol_folder, self.data_folder))

    def change_protocol(self):
        self.params_window = draw_window(self, None)
        self.params_window.show()
        
    def change_stimulus(self):
        self.params_window = draw_window(self, None)
        self.params_window.show()

    def change_setup(self):
        self.setup = self.cbst.currentText()
        
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
    main = MasterWindow(app)
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
