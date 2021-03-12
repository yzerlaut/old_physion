import sys, time, os, pathlib, subprocess
from PyQt5 import QtGui, QtWidgets, QtCore

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import get_files_with_extension, get_TSeries_folders
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
        self.quitSc = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+Q'), self) # or 'Ctrl+Q'
        self.quitSc.activated.connect(self.quit)
            
        self.setWindowTitle('File Transfer')
        
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
        # self.typeBox.activated.connect(self.update_setting)
        self.typeBox.addItems(['NWB', 'FULL',
                               'Imaging (processed)', 'Imaging (+binary)'])

        HEIGHT +=50 
        self.gen = QtWidgets.QPushButton(' -= RUN =-  ', self)
        self.gen.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.gen.clicked.connect(self.run)
        self.gen.setMinimumWidth(200)
        self.gen.move(50, HEIGHT)
        
        self.show()

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
            

    def file_copy_command(self, source_file, destination_folder):
        if sys.platform.startswith("win"):
            return 'xcopy %s %s' % (source_file,
                                               destination_folder)
        else:
            return 'cp %s %s' % (source_file, destination_folder)
            

    def folder_copy_command(self, source_folder, destination_folder):
        print('Full copy from ', source_folder, ' to ', destination_folder)
        print('can be long [...]')
        if sys.platform.startswith("win"):
            return 'xcopy %s %s /s /e' % (source_folder,
                                            destination_folder)
        else:
            return 'cp -r %s %s &' % (source_folder, destination_folder)
    
    def run(self):

        print('starting copy [...]')
        if self.destination_folder=='':
            self.destination_folder = FOLDERS[self.destBox.currentText()]
        if self.source_folder=='':
            self.source_folder = FOLDERS[self.sourceBox.currentText()]

        if self.typeBox.currentText()=='NWB':
            ##############################################
            #############      NWB file         ##########
            ##############################################
            FILES = get_files_with_extension(self.source_folder,
                                             extension='.nwb', 
                                             recursive=True)
            for f in FILES:
                cmd = self.file_copy_command(f, self.destination_folder)
                print('"%s" launched as a subprocess' % cmd)
                p = subprocess.Popen(cmd, shell=True)
        elif self.typeBox.currentText()=='FULL':
            self.folder_copy_command(self.source_folder,
                                     self.destination_folder)
        elif ('Imaging' in self.typeBox.currentText()):
            ##############################################
            #############      Imaging         ##########
            ##############################################
            if 'TSeries' in str(self.source_folder):
                folders = [self.source_folder]
            else:
                folders = get_TSeries_folders(self.source_folder)
            print('processing: ', folders)

            for f in folders:
                new_folder = os.path.join(self.destination_folder,
                                      'TSeries'+f.split('TSeries')[1])
                pathlib.Path(new_folder).mkdir(parents=True, exist_ok=True)
                # XML metadata file
                xml = get_files_with_extension(f, extension='.xml', recursive=False)
                if len(xml)>0:
                    print(' copying "%s" [...]' % xml[0])
                    subprocess.Popen(self.file_copy_command(xml[0], new_folder), shell=True)
                else:
                    print(' /!\ Problem no "xml" found !! /!\  ')
                # XML metadata file
                Fsuite2p = os.path.join(f, 'suite2p')
                iplane=0
                while os.path.isdir(os.path.join(Fsuite2p, 'plane%i' % iplane)):
                    npys = get_files_with_extension(os.path.join(Fsuite2p, 'plane%i' % iplane),
                                                    extension='.npy', recursive=False)
                    inewfolder = os.path.join(new_folder, 'suite2p', 'plane%i' % iplane)
                    pathlib.Path(inewfolder).mkdir(parents=True, exist_ok=True)
                    for n in npys:
                        print(' copying "%s" [...]' % n)
                        subprocess.Popen(self.file_copy_command(n, inewfolder), shell=True)
                        
                    if ('binary' in self.typeBox.currentText()) or ('full' in self.typeBox.currentText()):
                        if os.path.isfile(os.path.join(Fsuite2p, 'plane%i' % iplane, 'data.bin')):
                            print(' copying "%s" [...]' % os.path.join(Fsuite2p, 'plane%i' % iplane, 'data.bin'))
                            subprocess.Popen(self.file_copy_command(os.path.join(Fsuite2p, 'plane%i' % iplane, 'data.bin'), inewfolder), shell=True)
                        else:
                            print('In: "%s" ' % os.path.isfile(os.path.join(Fsuite2p, 'plane%i' % iplane)))
                            print(' /!\ Problem no "binary file" found !! /!\  ')

                                                    

                    iplane+=1
                    
        print('done (but cmd likely still running as subprocesses)')
                
        
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
        

