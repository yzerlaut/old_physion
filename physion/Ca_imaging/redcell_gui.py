import sys, os, shutil, glob, time, subprocess, pathlib
import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from misc.folders import FOLDERS
from misc.style import set_dark_style, set_app_icon
from misc.guiparts import NewWindow, Slider

class MainWindow(NewWindow):
    
    def __init__(self, app,
                 args=None,
                 parent=None):
        """
        FaceMotion GUI
        """
        self.app = app
        
        super(MainWindow, self).__init__(i=3,
                                         title='red-cell gui')


        ########################
        ##### building GUI #####
        ########################
        
        self.minView = False
        self.showwindow()

        # central widget
        self.cwidget = QtGui.QWidget(self)
        self.setCentralWidget(self.cwidget)
        
        # layout
        self.l0 = QtGui.QGridLayout()
        self.cwidget.setLayout(self.l0)
        self.win = pg.GraphicsLayoutWidget()
        self.win.move(600,0)
        self.win.resize(600,400)
        self.l0.addWidget(self.win,1,3,37,15)
        layout = self.win.ci.layout

        # A plot area (ViewBox + axes) for displaying the image
        self.p0 = self.win.addViewBox(lockAspect=False,row=0,col=0,invertY=True,
                                      border=[100,100,100])
        self.p0.setMouseEnabled(x=False,y=False)
        self.p0.setMenuEnabled(False)
        self.pimg = pg.ImageItem()
        self.p0.setAspectLocked()
        self.p0.addItem(self.pimg)

        self.folderB = QtWidgets.QComboBox(self)
        self.folderB.setMinimumWidth(150)
        self.folderB.addItems(FOLDERS.keys())
        
        self.load = QtGui.QPushButton('  load data [Ctrl+O]  \u2b07')
        self.load.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.load.clicked.connect(self.open_file)
        
        self.l0.addWidget(self.folderB,1,0,1,3)
        self.l0.addWidget(self.load,2,0,1,3)

        self.rois = pg.ScatterPlotItem()
        self.show()

    def refresh(self):

        if self.rois is not None:
            self.p0.removeItem(self.rois)
            self.draw_rois()

    def open_file(self):

        # folder = QtWidgets.QFileDialog.getExistingDirectory(self,\
        #                             "Choose datafolder",
        #                             FOLDERS[self.folderB.currentText()])
        self.folder = '/home/yann/UNPROCESSED/2021_06_17/TSeries-06172021-1146-001'

        if self.folder!='':
            self.draw_image()
            self.draw_rois()

    def draw_image(self):
        
        # load and plot img
        ops = np.load(os.path.join(self.folder, 'suite2p', 'plane0', 'ops.npy'), allow_pickle=True).item()
        self.pimg.setImage(ops['meanImg_chan2']**.5)


    def draw_rois(self, n=20, size=10):

        stat = np.load(os.path.join(self.folder, 'suite2p', 'plane0', 'stat.npy'), allow_pickle=True)
        redcell = np.load(os.path.join(self.folder, 'suite2p', 'plane0', 'redcell.npy'), allow_pickle=True)

        t = np.arange(n)
        x, y = [], []
        for i in range(len(stat)):
            if redcell[i,0]:
                xmean = np.mean(stat[i]['xpix'])
                ymean = np.mean(stat[i]['ypix'])
                # x.append(xmean)
                # y.append(ymean)
                x += list(xmean+size*np.cos(2*np.pi*t/n))
                y += list(ymean+size*np.sin(2*np.pi*t/n))
        
        self.rois.setData(x, y, size=1, brush=pg.mkBrush(255,0,0))
        self.p0.addItem(self.rois)
    

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



    
