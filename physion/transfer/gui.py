import sys, time, os, pathlib
from PyQt5 import QtGui, QtWidgets, QtCore

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import list_dayfolder, get_files_with_extension
from misc.folders import FOLDERS

class MainWindow(QtWidgets.QMainWindow):
    
    def __init__(self, app,
                 args=None,
                 parent=None):
        """
        sampling in Hz
        """
        super(MainWindow, self).__init__()

        self.setGeometry(650, 700, 300, 300)
        # adding a "quit" keyboard shortcut
        self.quitSc = QtWidgets.QShortcut(QtGui.QKeySequence('Q'), self) # or 'Ctrl+Q'
        self.quitSc.activated.connect(self.quit)
            
        self.setWindowTitle('Physion -- Transfer')
        
        self.script = os.path.join(\
                str(pathlib.Path(__file__).resolve().parents[1]),\
                'script.sh')
        
        self.source_folder, self.destination_folder = '', ''

        HEIGHT = 0

        HEIGHT += 20
        QtWidgets.QLabel("Root source:", self).move(10, HEIGHT)
        self.sourceBox = QtWidgets.QComboBox(self)
        self.sourceBox.setMinimumWidth(150)
        self.sourceBox.move(110, HEIGHT)
        self.sourceBox.activated.connect(self.update_setting)
        self.sourceBox.addItems(FOLDERS)
        
        HEIGHT += 40
        self.load = QtWidgets.QPushButton('Set source folder  \u2b07', self)
        self.load.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.load.clicked.connect(self.set_source_folder)
        self.load.setMinimumWidth(200)
        self.load.move(50, HEIGHT)

        HEIGHT += 60
        QtWidgets.QLabel("Root dest.:", self).move(10, HEIGHT)
        self.destBox = QtWidgets.QComboBox(self)
        self.destBox.setMinimumWidth(150)
        self.destBox.move(110, HEIGHT)
        self.destBox.activated.connect(self.set_destination_folder)
        self.destBox.addItems(FOLDERS)
        
        HEIGHT += 40
        self.load = QtWidgets.QPushButton('Set destination folder  \u2b07', self)
        self.load.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.load.clicked.connect(self.set_destination_folder)
        self.load.setMinimumWidth(200)
        self.load.move(50, HEIGHT)

        HEIGHT += 50
        QtWidgets.QLabel("=> What ?", self).move(10, HEIGHT)
        self.typeBox = QtWidgets.QComboBox(self)
        self.typeBox.setMinimumWidth(150)
        self.typeBox.move(100, HEIGHT)
        self.typeBox.activated.connect(self.update_setting)
        self.typeBox.addItems(['NWB', 'FULL', 'FaceCamera'])

        HEIGHT +=50 
        self.gen = QtWidgets.QPushButton(' -= RUN =-  ', self)
        self.gen.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.gen.clicked.connect(self.run)
        self.gen.setMinimumWidth(200)
        self.gen.move(50, HEIGHT)
        
        self.show()

    def update_setting(self):
        pass
        if self.cbc.currentText()=='custom':
            print('kjshdf')

    
    def set_source_folder(self):

        folder = QtWidgets.QFileDialog.getExistingDirectory(self,\
                                    "Set folder",
                                    FOLDERS[self.sourceBox.currentText()])
        if folder!='':
            self.source_folder = folder
            
    def set_destination_folder(self):

        folder = QtWidgets.QFileDialog.getExistingDirectory(self,\
                                    "Set folder",
                                    FOLDERS[self.destBox.currentText()])
        if folder!='':
            self.destination_folder = folder
            

    def clean_folder(self):
        
        if len(self.folder[-8:].split('_'))==3:
            print(list_dayfolder(self.folder))
        else:
            print(self.folder)
            
    
    def build_cmd(self):
        return 'python %s -df %s --%s' % (self.process_script,
                                          self.folder,
                                          self.cbc.currentText())

    def file_copy_command(source_file, destination_folder):
        if sys.platform.startswith("win"):
            return 'xcopy %s %s' % (source_file, destination_folder)
        else:
            return 'cp %s %s' % (source_file, destination_folder)
            

    def folder_copy_command(source_folder, destination_folder):
        pass
    
    def run(self):

        if self.destination_folder=='':
            self.destination_folder = FOLDERS[self.destBox.currentText()])
        if self.source_folder=='':
            self.source_folder = FOLDERS[self.sourceBox.currentText()])
            
        if self.typeBox.currentText()=='NWB':
            FILES = get_files_with_extension(self.source_folder,
                                             extension='.nwb', 
                                             recursive=True)
            for f in FILES:
                cmd = file_copy_command(f, self.destination_folder)
                # p = subprocess.Popen(cmd, shell=True)
                print('"%s" launched as a subprocess' % cmd)
        
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
        

