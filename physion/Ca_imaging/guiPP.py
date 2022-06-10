import sys, time, os, pathlib, subprocess
from PyQt5 import QtGui, QtWidgets, QtCore

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import list_dayfolder, get_TSeries_folders
from assembling.tools import find_matching_CaImaging_data
from misc.folders import FOLDERS, python_path, python_path_suite2p_env
from Ca_imaging.preprocessing import PREPROCESSING_SETTINGS
from Ca_imaging.redcell_gui import run as RunRedCellGui
from Ca_imaging.zstack.cellID_gui import run as RunZstackGui

class MainWindow(QtWidgets.QMainWindow):
    
    def __init__(self, app,
                 args=None,
                 parent=None):
        """
        sampling in Hz
        """
        super(MainWindow, self).__init__()
        self.app, self.args, self.CHILDREN_PROCESSES = app, args, []
        self.folder=''
        
        self.setGeometry(350, 470, 300, 350)
        # adding a "quit" and "load" keyboard shortcuts
        self.quitSc = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+Q'), self)
        self.quitSc.activated.connect(self.quit)
        self.loadSc = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+O'), self)
        self.loadSc.activated.connect(self.load_imaging)
            
        self.setWindowTitle('Ca-Preprocessing')
        
        self.process_script = os.path.join(str(pathlib.Path(__file__).resolve().parents[1]),
                                           'Ca_imaging',  'preprocessing.py')
        
        HEIGHT = 0

        HEIGHT += 10
        QtWidgets.QLabel("Root Folder:", self).move(10, HEIGHT)
        self.folderI = QtWidgets.QComboBox(self)
        self.folderI.setMinimumWidth(150)
        self.folderI.move(100, HEIGHT)
        self.folderI.activated.connect(self.update_setting)
        self.folderI.addItems(FOLDERS.keys())
        
        HEIGHT += 40
        self.load = QtWidgets.QPushButton('Upload (set of) TSeries \u2b07', self)
        self.load.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.load.clicked.connect(self.load_imaging)
        self.load.setMinimumWidth(200)
        self.load.move(50, HEIGHT)

        HEIGHT += 40
        QtWidgets.QLabel("=> Setting :", self).move(10, HEIGHT)
        self.cbc = QtWidgets.QComboBox(self)
        self.cbc.setMinimumWidth(150)
        self.cbc.move(100, HEIGHT)
        self.cbc.activated.connect(self.update_setting)
        self.cbc.addItems(['automated']+list(PREPROCESSING_SETTINGS.keys()))

        HEIGHT += 40
        QtWidgets.QLabel("   delay ?", self).move(10, HEIGHT)
        self.delayBox = QtWidgets.QComboBox(self)
        self.delayBox.setMinimumWidth(150)
        self.delayBox.move(100, HEIGHT)
        self.delayBox.addItems(['None', '10s', '10min', '30min',
                                '1h', '2h', '3h', '5h', '10h', '1d', '2d'])
        
        HEIGHT +=40 
        self.gen = QtWidgets.QPushButton('-=- Run -=-', self)
        self.gen.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.gen.clicked.connect(self.run)
        self.gen.setMinimumWidth(200)
        self.gen.move(50, HEIGHT)
        
        HEIGHT +=40 
        self.gen = QtWidgets.QPushButton(' open Suite2p ', self)
        self.gen.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.gen.clicked.connect(self.open_suite2p)
        self.gen.setMinimumWidth(200)
        self.gen.move(50, HEIGHT)
        
        HEIGHT +=40 
        self.gen = QtWidgets.QPushButton('red-cell selection GUI ', self)
        self.gen.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.gen.clicked.connect(self.red_cell_selection)
        self.gen.setMinimumWidth(200)
        self.gen.move(50, HEIGHT)

        HEIGHT +=40 
        self.gen = QtWidgets.QPushButton('Zstack cell selection GUI ', self)
        self.gen.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.gen.clicked.connect(self.zstack_selection)
        self.gen.setMinimumWidth(200)
        self.gen.move(50, HEIGHT)
        
        self.CMDS = []
        self.show()

    def update_setting(self):
        self.CMDS = []
        print('command set is reset !')
    
    def build_cmd(self, folder, key):
        # delay = ''
        # if self.delayBox.currentText()!='None':
        #     delay = 'sleep %s; ' % self.delayBox.currentText()
        return '%s %s --CaImaging_folder "%s" --setting_key %s -v' % (python_path_suite2p_env,
                                                                      self.process_script, folder, key)

    def find_suite2p_settings(self, folder):
        settings = None
        if self.cbc.currentText()=='automated':
            potential_settings = folder.split('-')[-2]
            if potential_settings in list(PREPROCESSING_SETTINGS.keys()):
                settings = potential_settings
                print('using "%s" setting for: %s' % (settings, folder))
            else:
                print('settings not found for', folder)
        else:
            settings = self.cbc.currentText()
        return settings
        
    def load_imaging(self):

        self.folder = QtWidgets.QFileDialog.getExistingDirectory(self,\
                                    "Choose datafolder",
                                    FOLDERS[self.folderI.currentText()])
        if self.folder!='':
            if ('TSeries' in str(self.folder)):
                settings = self.find_suite2p_settings(str(self.folder))
                if settings is not None:
                    self.CMDS.append(self.build_cmd(self.folder, self.cbc.currentText()))
                else:
                    print('settings note recognized !')
            else:
                folders = get_TSeries_folders(self.folder, limit_to_subdirectories=False)
                for f in folders:
                    settings = self.find_suite2p_settings(str(f))
                    if settings is not None:
                        self.CMDS.append(self.build_cmd(f, settings))
                        
            for cmd in self.CMDS:
                print('"%s" added to command set' % cmd)

        if len(self.CMDS)==0:
            print('\n /!\ no "TSeries" folder found or added, set of command is empty ! \n')
                
    def run(self):
        if self.delayBox.currentText()!='None':
            # delaying the run by a delay
            delay = eval(self.delayBox.currentText().replace('s', '').replace('m', '*60').replace('h', '*3600').replace('d', '*86400'))
            print('sleeping for %is' % delay)
            time.sleep(delay)
            print('sleeping time over !')
            # ----------------------------------------------------
            self.CMDS = [] # we overwrite the previous commands
            # looping over folder after the delay (in case the transfer was done)
            folders = get_TSeries_folders(self.folder, limit_to_subdirectories=False)
            for f in folders:
                settings = self.find_suite2p_settings(str(f))
                if settings is not None:
                    self.CMDS.append(self.build_cmd(f, settings))
        elif len(self.CMDS)==0:
            print('\n /!\ set of command is empty /!\  \n')

        # running commands
        for cmd in self.CMDS:
            p = subprocess.Popen(cmd, shell=True)
            print('"%s" launched as a subprocess' % cmd)

    def open_suite2p(self):
        p = subprocess.Popen('%s -m suite2p' % python_path_suite2p_env,
                             shell=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)

    def red_cell_selection(self):
        self.CHILDREN_PROCESSES.append(RunRedCellGui(self.app, self.args))

    def zstack_selection(self):
        self.CHILDREN_PROCESSES.append(RunZstackGui(self.app, self.args))
        
    def quit(self):
        QtWidgets.QApplication.quit()
        
def run(app, args=None, parent=None):
    return MainWindow(app,
                      args=args,
                      parent=parent)

if __name__=='__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = run(app)
    sys.exit(app.exec_())
        

