import sys, time, os, pathlib, subprocess
from PyQt5 import QtGui, QtWidgets, QtCore

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import list_dayfolder, get_TSeries_folders
from assembling.tools import find_matching_CaImaging_data
from misc.folders import FOLDERS, python_path

class MainWindow(QtWidgets.QMainWindow):
    
    def __init__(self, app,
                 args=None,
                 parent=None):
        """
        sampling in Hz
        """
        super(MainWindow, self).__init__()

        self.setGeometry(350, 680, 300, 350)
        # adding a "quit" keyboard shortcut
        self.quitSc = QtWidgets.QShortcut(QtGui.QKeySequence('Q'), self) # or 'Ctrl+Q'
        self.quitSc.activated.connect(self.quit)
            
        self.setWindowTitle('Adding Ca-Imaging')
        
        self.process_script = os.path.join(str(pathlib.Path(__file__).resolve().parents[1]),
                                           'assembling',  'add_ophys.py')

        HEIGHT = 0

        HEIGHT += 10
        QtWidgets.QLabel("Root Data:", self).move(10, HEIGHT)
        self.folderD = QtWidgets.QComboBox(self)
        self.folderD.setMinimumWidth(150)
        self.folderD.move(100, HEIGHT)
        self.folderD.activated.connect(self.update_setting)
        self.folderD.addItems(FOLDERS.keys())

        HEIGHT += 30
        QtWidgets.QLabel("Root Imaging:", self).move(10, HEIGHT)
        self.folderI = QtWidgets.QComboBox(self)
        self.folderI.setMinimumWidth(150)
        self.folderI.move(100, HEIGHT)
        self.folderI.activated.connect(self.update_setting)
        self.folderI.addItems(FOLDERS.keys())
        
        HEIGHT += 40
        self.load = QtWidgets.QPushButton('Load data \u2b07', self)
        self.load.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.load.clicked.connect(self.load_data)
        self.load.setMinimumWidth(200)
        self.load.move(50, HEIGHT)
        self.loadSc = QtWidgets.QShortcut(QtGui.QKeySequence('L'), self)
        self.loadSc.activated.connect(self.load_data)

        HEIGHT += 40
        self.load = QtWidgets.QPushButton('Load imaging \u2b07', self)
        self.load.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.load.clicked.connect(self.load_data)
        self.load.setMinimumWidth(200)
        self.load.move(50, HEIGHT)
        self.loadSc = QtWidgets.QShortcut(QtGui.QKeySequence('L'), self)
        self.loadSc.activated.connect(self.load_imaging)

        HEIGHT += 40
        self.find = QtWidgets.QPushButton('-* Find imaging *-', self)
        self.find.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.find.clicked.connect(self.find_imaging)
        self.find.setMinimumWidth(200)
        self.find.move(50, HEIGHT)

        HEIGHT += 30
        self.imagingF = QtWidgets.QLabel("...", self)
        self.imagingF.move(10, HEIGHT)
        self.imagingF.setMinimumWidth(400)
        
        HEIGHT += 40
        QtWidgets.QLabel("=> Setting :", self).move(10, HEIGHT)
        self.cbc = QtWidgets.QComboBox(self)
        self.cbc.setMinimumWidth(150)
        self.cbc.move(100, HEIGHT)
        self.cbc.activated.connect(self.update_setting)
        self.cbc.addItems(['standard', 'lightweight', 'full', 'NIdaq-only', 'custom'])

        HEIGHT +=30 
        s = QtWidgets.QLabel("CaImaging-Sampling (Hz)    ", self)
        s.move(10, HEIGHT)
        s.setMinimumWidth(200)
        self.PsamplingBox = QtWidgets.QLineEdit('', self)
        self.PsamplingBox.setText('0.0')
        self.PsamplingBox.setFixedWidth(40)
        self.PsamplingBox.move(200, HEIGHT)

        HEIGHT +=40 
        self.gen = QtWidgets.QPushButton('-=- Run -=-', self)
        self.gen.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.gen.clicked.connect(self.run)
        self.gen.setMinimumWidth(200)
        self.gen.move(50, HEIGHT)
        
        self.filename, self.Ifolder = '', ''
        self.show()

    def update_setting(self):
        pass
        if self.cbc.currentText()=='custom':
            print('kjshdf')

    
    def load_data(self):

        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self,
                                                            "Open datafile (through metadata file) )",
                                                            FOLDERS[self.folderD.currentText()],
                                                            filter="*.nwb")
        if filename!='':
            self.filename = filename
        else:
            self.filename = ''
            

    def load_imaging(self):
        print('deprecated, use the "find" engine for security')
        # filename, _ = QtWidgets.QFileDialog.getOpenFileName(self,
        #                                                     "Open datafile (through metadata file) )",
        #                                                     FOLDERS[self.folderI.currentText()],
        #                                                     filter="*.xml")
        # if filename!='':
        #     self.Ifolder = os.path.dirname(filename)
        # else:
        #     self.Ifolder = ''

    def find_imaging(self):

        if self.filename!='':
            print('searching for a match [...]')
            self.imagingF.setText('searching [...]')
            success, folder = find_matching_CaImaging_data(self.filename,
                                                           FOLDERS[self.folderI.currentText()])
            if success:
                self.imagingF.setText('"%s"' % folder.split(os.path.sep)[-1])
                self.Ifolder = folder
            else:
                self.imagingF.setText('not found')
        else:
            print(' /!\ need to provide a NWB file /!\ ')
            
            
    def clean_folder(self):
        
        if len(self.folder[-8:].split('_'))==3:
            print(list_dayfolder(self.folder))
        else:
            print(self.folder)
            
    
    def build_cmd(self):
        return '%s %s --CaImaging_folder %s --nwb_file %s -v' % (python_path,
                                                                 self.process_script,
                                                                 self.Ifolder,
                                                                 self.filename)
    def run(self):
        
        if self.filename!='' and self.Ifolder!='':

            p = subprocess.Popen(self.build_cmd(),
                                 shell=True)
            print('"%s" launched as a subprocess' % self.build_cmd())
        else:
            print(' /!\ Need a valid folder !  /!\ ')
            
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
        

