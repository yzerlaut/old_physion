import sys, os, shutil, glob, time, subprocess, pathlib
import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from PIL import Image

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from misc.folders import FOLDERS, python_path
from misc.guiparts import NewWindow, Slider
from assembling.tools import load_FaceCamera_data
from pupil.roi import extract_ellipse_props, ellipse_props_to_ROI
from assembling.IO.bruker_xml_parser import bruker_xml_parser
from assembling.saving import get_files_with_extension

class MainWindow(NewWindow):
    
    def __init__(self, app,
                 args=None,
                 parent=None,
                 gaussian_smoothing=1,
                 cm_scale_px=570,
                 subsampling=1000):
        """
        Pupil Tracking GUI
        """
        self.app = app
        
        super(MainWindow, self).__init__(i=-2,
                                         size=(1200,400),
                                         title='Cell Identification GUI for Z-stacks Imaging')


        ##############################
        ##### keyboard shortcuts #####
        ##############################

        self.refc1 = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+N'), self)
        self.refc1.activated.connect(self.next_frames)
        self.refc2 = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+B'), self)
        self.refc2.activated.connect(self.previous_frames)
        
        #############################
        ##### module quantities #####
        #############################

        self.ROI = None
        
        ########################
        ##### building GUI #####
        ########################
        
        self.cwidget = QtGui.QWidget(self)
        self.setCentralWidget(self.cwidget)
        self.grid = QtGui.QGridLayout()

        self.folderB = QtWidgets.QComboBox(self)
        self.folderB.setMinimumWidth(150)
        self.folderB.addItems(FOLDERS.keys())
        self.grid.addWidget(self.folderB, 1, 1, 1, 1)

        self.load = QtGui.QPushButton('  load data [Ctrl+O]  \u2b07')
        self.load.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.load.clicked.connect(self.load_data)
        self.grid.addWidget(self.load, 1, 2, 1, 1)

        self.channelB = QtWidgets.QComboBox(self)
        self.channelB.setMinimumWidth(150)
        self.channelB.addItems(['Ch2', 'Ch1'])
        self.grid.addWidget(self.channelB, 1, 3, 1, 1)
        
        self.newRoiBtn = QtGui.QPushButton('new ROI')
        self.newRoiBtn.clicked.connect(self.new_ROI)
        self.grid.addWidget(self.newRoiBtn, 1, 5, 1, 1)

        self.saveRoiBtn = QtGui.QPushButton('save ROI')
        self.saveRoiBtn.clicked.connect(self.save_ROI)
        self.grid.addWidget(self.saveRoiBtn, 1, 6, 1, 1)

        self.prevBtn = QtGui.QPushButton('next frames [Ctrl+N]')
        self.prevBtn.clicked.connect(self.next_frames)
        self.grid.addWidget(self.prevBtn, 1, 9, 1, 1)
        
        self.nextBtn = QtGui.QPushButton('prev. frames [Ctrl+P]')
        self.nextBtn.clicked.connect(self.previous_frames)
        self.grid.addWidget(self.nextBtn, 1, 10, 1, 1)

        self.threeDBtn = QtGui.QPushButton('show 3D view')
        self.threeDBtn.clicked.connect(self.show_3d_view)
        self.grid.addWidget(self.threeDBtn, 1, 12, 1, 1)

        self.saveCellsBtn = QtGui.QPushButton('save cells')
        self.saveCellsBtn.clicked.connect(self.save_cells)
        self.grid.addWidget(self.saveCellsBtn, 1, 11, 1, 1)
        
        self.cwidget.setLayout(self.grid)
        self.win = pg.GraphicsLayoutWidget()
        self.grid.addWidget(self.win,2,1,12,12)
        layout = self.win.ci.layout

        for i, key in enumerate(['pPrev', 'Prev', 'Center', 'Next', 'nNext']):
            # A plot area (ViewBox + axes) for displaying the image
            setattr(self, 'view_%s' % key,
                    self.win.addViewBox(lockAspect=True,
                                        row=0,col=i,
                                        border=[10,10,10],
                                        invertY=True))
            setattr(self, 'img_%s' % key, pg.ImageItem())
            getattr(self, 'view_%s' % key).addItem(getattr(self, 'img_%s' % key))

        self.win.show()
        self.show()

    def new_ROI(self):
        self.newROI = pg.EllipseROI([200, 200], [20, 20],
            pen=pg.mkPen([0,200,50], width=3, style=QtCore.Qt.SolidLine),
            movable=True)
        self.view_Center.addItem(self.newROI)

    def save_ROI(self):
        coords = extract_ellipse_props(self.newROI)
        cell = {}
        for key, c in zip(['x', 'y', 'sx', 'sy'], coords):
            cell[key] = c
        cell['z'] = self.bruker_data[self.channelB.currentText()]['depth'][self.cframe]
        self.ROIS.append(cell)
    
    def open_file(self):
        self.load_data()
        
    def load_data(self):


        # folder = QtWidgets.QFileDialog.getExistingDirectory(self,\
        #                             "Choose datafolder",
        #                             FOLDERS[self.folderB.currentText()])
        
        folder = '/home/yann/DATA/CaImaging/SSTcre_GCamp6s/Z-Stacks/In-Vitro-Mouse2'
        
        if folder!='':
            self.folder=folder
            xml_file = os.path.join(folder, get_files_with_extension(self.folder,
                                                                     extension='.xml')[0])
            self.bruker_data = bruker_xml_parser(xml_file)

            self.cframe = 0
            self.nframes = len(self.bruker_data[self.channelB.currentText()]['tifFile'])
            self.display_frames()
            self.ROIS = []

    def plot_frame(self, panel, cframe):
        
        
        if (self.cframe>0) and (self.cframe<(self.nframes-1)):
            imgP = np.array(Image.open(os.path.join(self.folder,
                    self.bruker_data[self.channelB.currentText()]['tifFile'][cframe-1])))
            imgC = np.array(Image.open(os.path.join(self.folder,
                    self.bruker_data[self.channelB.currentText()]['tifFile'][cframe])))
            imgN = np.array(Image.open(os.path.join(self.folder,
                    self.bruker_data[self.channelB.currentText()]['tifFile'][cframe+1])))
            img = .25*(imgN+imgP)+.5*imgC
        else:
            img = np.array(Image.open(os.path.join(self.folder,
                    self.bruker_data[self.channelB.currentText()]['tifFile'][cframe])))
            
        panel.setImage(img)
        panel.setLevels([img.min(), img.max()])
        # panel.setLevels([0, 5000])

    def display_frames(self):

        if self.cframe>1:
            self.plot_frame(self.img_pPrev, self.cframe-2)
        else:
            self.img_pPrev.setImage(np.zeros((2,2)))
            
        if self.cframe>0:
            self.plot_frame(self.img_Prev, self.cframe-1)
        else:
            self.img_Prev.setImage(np.zeros((2,2)))

        self.plot_frame(self.img_Center, self.cframe)
        
        if self.cframe<(self.nframes-1):
            self.plot_frame(self.img_Next, self.cframe+1)
        else:
            self.img_Next.setImage(np.zeros((2,2)))
            
        if self.cframe<(self.nframes-2):
            self.plot_frame(self.img_nNext, self.cframe+2)
        else:
            self.img_nNext.setImage(np.zeros((2,2)))
        
    def previous_frames(self):
        self.cframe = np.max([0, self.cframe-1])
        self.display_frames()

    def next_frames(self):
        self.cframe = np.min([self.nframes-1, self.cframe+1])
        self.display_frames()

    def init_cells_data(self):
        cells = {'x':[], 'y':[], 'z':[], 'sx':[], 'sy':[],
                 'dx':float(self.bruker_data['settings']['micronsPerPixel']['XAxis']),
                 'dy':float(self.bruker_data['settings']['micronsPerPixel']['YAxis']),
                 'zlim':[min(self.bruker_data[self.channelB.currentText()]['depth']),
                         max(self.bruker_data[self.channelB.currentText()]['depth'])],
                 'nxpix':int(self.bruker_data['settings']['pixelsPerLine']),
                 'nypix':int(self.bruker_data['settings']['linesPerFrame'])}
        return cells
    
    def save_cells(self):
        cells = self.init_cells_data()
        for roi in self.ROIS:
            for key in ['x', 'y', 'z', 'sx', 'sy']:
                cells[key].append(roi[key])
        print(cells)
        np.save(os.path.join(self.folder, 'cells.npy'), cells)
            
    def show_3d_view(self):

        print('3d view')
        cell_file = os.path.join(self.folder, 'cells.npy')
        if os.path.isfile(cell_file):
            cells = np.load(cell_file, allow_pickle=True).item()
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(cells['x'], cells['y'], cells['z'], marker='o', color='r', s=20)
            ax.set_xlabel('x (um)')
            ax.set_ylabel('y (um)')
            ax.set_zlabel('z (um)')
            ax.set_xlim([0, cells['dx']*cells['nxpix']])
            ax.set_ylim([0, cells['dy']*cells['nypix']])
            ax.set_zlim(cells['zlim'])
            plt.show()


    def process(self):
        self.previous_frames()
        
    def quit(self):
        QtWidgets.QApplication.quit()

def run(app, args=None, parent=None):
    return MainWindow(app,
                      args=args,
                      parent=parent)
    
if __name__=='__main__':
    from misc.colors import build_dark_palette
    import tempfile, argparse, os
    parser=argparse.ArgumentParser(description="Experiment interface",
                       formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-f', "--datafile", type=str,default='')
    parser.add_argument('-rf', "--root_datafolder", type=str,
                        default=os.path.join(os.path.expanduser('~'), 'DATA'))
    parser.add_argument('-v', "--verbose", action="store_true")
    args = parser.parse_args()
    app = QtWidgets.QApplication(sys.argv)
    build_dark_palette(app)
    main = MainWindow(app,
                      args=args)
    sys.exit(app.exec_())



    
