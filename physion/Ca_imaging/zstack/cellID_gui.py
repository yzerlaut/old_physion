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

roipen1 = pg.mkPen((255, 0, 0), width=2, style=QtCore.Qt.SolidLine) # red line
roipen2 = pg.mkPen((0, 255, 0), width=1, style=QtCore.Qt.SolidLine) # green

class MainWindow(NewWindow):
    
    def __init__(self, app,
                 args=None,
                 parent=None,
                 gaussian_smoothing=1,
                 cm_scale_px=570,
                 subsampling=1000):
        """
        Cell Identification GUI for Z-stacks Imaging
        """
        self.app = app
        
        super(MainWindow, self).__init__(i=-2,
                                         size=(1000,800),
                                         title='Cell Identification GUI for Z-stacks Imaging')


        ##############################
        ##### keyboard shortcuts #####
        ##############################

        self.refc1 = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+N'), self)
        self.refc1.activated.connect(self.next_frames)
        self.refc2 = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+B'), self)
        self.refc2.activated.connect(self.previous_frames)
        self.refc3 = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+C'), self)
        self.refc3.activated.connect(self.new_ROI)
        
        #############################
        ##### module quantities #####
        #############################

        self.ROIS = []
        
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
        self.load.clicked.connect(self.open_file)
        self.grid.addWidget(self.load, 1, 2, 1, 1)

        self.channelB = QtWidgets.QComboBox(self)
        self.channelB.addItems(['Ch2', 'Ch1'])
        self.grid.addWidget(self.channelB, 1, 3, 1, 1)

        self.grid.addWidget(QtGui.QLabel("sampling (frame):"), 1, 4, 1, 1)
        self.samplingB = QtWidgets.QComboBox(self)
        self.samplingB.addItems(['1', '2', '3', '5', '7', '10'])
        self.grid.addWidget(self.samplingB, 1, 5, 1, 1)

        self.depth = QtGui.QLineEdit()
        self.depth.setText("depth= %.1f um" % 0)
        self.grid.addWidget(self.depth, 1, 6, 1, 2)

        self.grid.addWidget(QtGui.QLabel("cell-range (um):"), 1, 8, 1, 1)
        self.cellRange = QtWidgets.QComboBox(self)
        self.cellRange.addItems(['1', '10', '20', '50', '100', '1000'])
        self.cellRange.setCurrentIndex(2)
        self.grid.addWidget(self.cellRange, 1, 9, 1, 1)

        
        self.newRoiBtn = QtGui.QPushButton('new Cell [Ctrl+C]')
        self.newRoiBtn.clicked.connect(self.new_ROI)
        self.grid.addWidget(self.newRoiBtn, 2, 1, 1, 1)

        self.showCells = QtGui.QCheckBox("show cells")
        self.showCells.clicked.connect(self.display_frames)
        self.grid.addWidget(self.showCells, 2, 2, 1, 2)
        
        self.prevBtn = QtGui.QPushButton('next frames [Ctrl+N]')
        self.prevBtn.clicked.connect(self.next_frames)
        self.grid.addWidget(self.prevBtn, 2, 4, 1, 1)
        
        self.nextBtn = QtGui.QPushButton('prev. frames [Ctrl+P]')
        self.nextBtn.clicked.connect(self.previous_frames)
        self.grid.addWidget(self.nextBtn, 2, 5, 1, 1)

        self.saveCellsBtn = QtGui.QPushButton('save cells')
        self.saveCellsBtn.clicked.connect(self.save_cells)
        self.grid.addWidget(self.saveCellsBtn, 2, 7, 1, 1)
        
        self.threeDBtn = QtGui.QPushButton('show 3D view')
        self.threeDBtn.clicked.connect(self.show_3d_view)
        self.grid.addWidget(self.threeDBtn, 2, 8, 1, 1)

        self.resetBtn = QtGui.QPushButton('reset')
        self.resetBtn.clicked.connect(self.reset)
        self.grid.addWidget(self.resetBtn, 2, 10, 1, 1)

        self.cwidget.setLayout(self.grid)
        self.win = pg.GraphicsLayoutWidget()
        self.grid.addWidget(self.win,3,1,10,10)
        layout = self.win.ci.layout

        # for i, key in enumerate(['pPrev', 'Prev', 'Center', 'Next', 'nNext']):
        for i, key in enumerate(['Center']):
            # A plot area (ViewBox + axes) for displaying the image
            setattr(self, 'view_%s' % key,
                    self.win.addViewBox(lockAspect=True,
                                        row=0,col=i,
                                        border=[10,10,10],
                                        invertY=True))
            setattr(self, 'img_%s' % key, pg.ImageItem())
            getattr(self, 'view_%s' % key).addItem(getattr(self, 'img_%s' % key))

        if hasattr(args, 'folder'):
            self.load_data(args.folder)
            
        self.win.show()
        self.show()

    def reset(self):
        self.ROIS = []
        print('[ok] cell data reset !')
    
    def hitting_space(self):

        if self.showCells.isChecked():
            self.showCells.setChecked(False)
        else:
            self.showCells.setChecked(True)
        self.display_frames()

    def new_ROI(self):
        self.ROIS.append(cellROI(depth=float(self.bruker_data[self.channelB.currentText()]['depth'][self.cframe]),
                                 parent=self))

    def open_file(self):

        folder = QtWidgets.QFileDialog.getExistingDirectory(self,\
                                    "Choose datafolder",
                                    FOLDERS[self.folderB.currentText()])
        self.load_data(folder)
        
    def load_data(self, folder):

        if folder!='':
            self.folder=folder
            xml_file = os.path.join(folder, get_files_with_extension(self.folder,
                                                                     extension='.xml')[0])
            self.bruker_data = bruker_xml_parser(xml_file)

            self.cframe = 0
            self.nframes = len(self.bruker_data[self.channelB.currentText()]['tifFile'])
            self.display_frames()
            self.ROIS = []
            if os.path.isfile(os.path.join(self.folder, 'cells.npy')):
                cells=np.load(os.path.join(self.folder, 'cells.npy'), allow_pickle=True).item()
                for x, y, z in zip(cells['x'], cells['y'], cells['z']):
                    self.ROIS.append(cellROI(depth=z,pos=(x,y),
                                             parent=self))
        else:
            print('"%s" is not a valid folder' % folder)

    def plot_frame(self, panel, cframe):
        
        N = int(self.samplingB.currentText())
        iframes = np.arange(np.max([0, cframe-N-1]), np.min([self.nframes-1, cframe+N]))

        img = np.zeros((int(self.bruker_data['settings']['pixelsPerLine']),
                       int(self.bruker_data['settings']['linesPerFrame'])))
        for i in iframes:
            img += np.array(Image.open(os.path.join(self.folder,self.bruker_data[self.channelB.currentText()]['tifFile'][i])))*np.exp(-(i-cframe)**2/N**2)

        panel.setImage(img)
        panel.setLevels([img.min(), img.max()])

    def display_frames(self):

        current_depth = float(self.bruker_data[self.channelB.currentText()]['depth'][self.cframe])
        self.depth.setText("depth= %.1f um" % current_depth)
        
        # remove previous and add rois
        for roi in self.ROIS:
            roi.remove_from_view(self.view_Center)
            if roi.depth==current_depth and self.showCells.isChecked():
                    roi.ROI.setPen(roipen1)
                    roi.add_to_view(self.view_Center)
            elif self.showCells.isChecked() and (np.abs(current_depth-roi.depth)<float(self.cellRange.currentText())):
                roi.ROI.setPen(roipen2)
                roi.add_to_view(self.view_Center)
                
        # if self.cframe>(int(self.samplingB.currentText())-1):
        #     self.plot_frame(self.img_Prev, self.cframe-int(self.samplingB.currentText()))
        # else:
        #     self.img_Prev.setImage(np.zeros((2,2)))

        self.plot_frame(self.img_Center, self.cframe)
        
        # if self.cframe<(self.nframes-int(self.samplingB.currentText())):
        #     self.plot_frame(self.img_Next, self.cframe+int(self.samplingB.currentText()))
        # else:
        #     self.img_Next.setImage(np.zeros((2,2)))
            
        
    def previous_frames(self):
        self.cframe = np.max([0, self.cframe-int(self.samplingB.currentText())])
        self.display_frames()

    def next_frames(self):
        self.cframe = np.min([self.nframes-1, self.cframe+int(self.samplingB.currentText())])
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
            cells['z'].append(roi.depth)
            for key, val in zip(['x', 'y', 'sx', 'sy'], roi.extract_props()[:-1]):
                cells[key].append(val)
        np.save(os.path.join(self.folder, 'cells.npy'), cells)
        print('[ok] cell data saved as: ', os.path.join(self.folder, 'cells.npy'))
            
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


class cellROI():
    def __init__(self, depth=0., parent=None, pos=None):
        # which ROI it belongs to
        self.depth = depth
        self.color = (255.,0.,0.)
        dx, dy = 20, 20
        if pos is None:
            view = parent.view_Center.viewRange()
            imx = (view[0][1] + view[0][0]) / 2
            imy = (view[1][1] + view[1][0]) / 2
            imx, imy = imx - dx / 2, imy - dy / 2
        else:
            imy, imx = pos[0] - dx / 2, pos[1] - dy / 2, 
        self.draw(parent, imy, imx, dy, dx)
        self.ROI.sigRemoveRequested.connect(lambda: self.remove(parent))

    def draw(self, parent, imy, imx, dy, dx, angle=0, movable = True, removable=True):
        roipen = pg.mkPen(self.color, width=2, style=QtCore.Qt.SolidLine)
        self.ROI = pg.EllipseROI(
            [imx, imy], [dx, dy], pen=roipen, 
            movable = True, removable=True, resizable=False, rotatable=False)
        self.ROI.handleSize = 1
        self.ROI.handlePen = roipen
        self.ROI.addScaleHandle([0.1, 0], [0.1, 0.2])
        self.ROI.setAcceptedMouseButtons(QtCore.Qt.LeftButton)
        self.add_to_view(parent.view_Center)

    def add_to_view(self, view):
        view.addItem(self.ROI)

    def remove_from_view(self, view):
        view.removeItem(self.ROI)

    def remove(self, parent):
        parent.view_Center.removeItem(self.ROI)
        parent.ROIS.remove(self)

    def position(self, parent):
        pass

    def extract_props(self):
        return extract_ellipse_props(self.ROI)
        
def run(app, args=None, parent=None):
    return MainWindow(app,
                      args=args,
                      parent=parent)
    
if __name__=='__main__':
    
    from misc.colors import build_dark_palette
    import tempfile, argparse, os
    parser=argparse.ArgumentParser(description="Experiment interface",
                       formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-f', "--folder", type=str,
                        default = '/home/yann/DATA/CaImaging/SSTcre_GCamp6s/Z-Stacks/In-Vivo-Mouse2')
    parser.add_argument('-v', "--verbose", action="store_true")
    
    args = parser.parse_args()
    
    app = QtWidgets.QApplication(sys.argv)
    build_dark_palette(app)
    main = MainWindow(app,
                      args=args)
    sys.exit(app.exec_())



    
