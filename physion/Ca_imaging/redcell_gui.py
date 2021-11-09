import sys, os, shutil, glob, time, subprocess, pathlib
import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from misc.folders import FOLDERS
from misc.style import set_dark_style, set_app_icon
from misc.guiparts import NewWindow, Slider

class RCGwindow(NewWindow):
    
    def __init__(self, app,
                 args=None,
                 parent=None):
        """
        Red-Cell selection GUI
        """
        self.app = app
        
        super(RCGwindow, self).__init__(i=3,
                                        title='red-cell gui')

        self.nextSc = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+N'), self)
        self.nextSc.activated.connect(self.next_roi)

        self.saveSc = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+S'), self)
        self.saveSc.activated.connect(self.save)
        
        self.roi_index=0
        
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

        # buttons and widgets
        self.folderB = QtWidgets.QComboBox(self)
        self.folderB.setMinimumWidth(150)
        self.folderB.addItems(FOLDERS.keys())

        self.imgB = QtWidgets.QComboBox(self)
        self.imgB.setMinimumWidth(150)
        self.imgB.addItems(['meanImg_chan2', 'meanImg', 'max_proj'])
        self.imgB.activated.connect(self.draw_image)
        
        self.load = QtGui.QPushButton('  load data [Ctrl+O]  \u2b07')
        self.load.clicked.connect(self.open_file)

        self.nextB = QtGui.QPushButton('next roi [Ctrl+N]')
        self.nextB.clicked.connect(self.next_roi)

        self.prevB = QtGui.QPushButton('prev. roi [Ctrl+P]')
        self.prevB.clicked.connect(self.process)

        self.switchB = QtGui.QPushButton('SWITCH [Ctrl+Space]')
        self.switchB.clicked.connect(self.hitting_space)

        self.saveB = QtGui.QPushButton('save data [Ctrl+S]')
        self.saveB.clicked.connect(self.save)
        
        for wdgt, index in zip([self.folderB, self.load, self.imgB, self.nextB, self.prevB,
                                self.switchB, self.saveB], [1,2,6,10,11,12,16]):
            wdgt.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
            self.l0.addWidget(wdgt,index,0,1,3)
        

        self.rois_green = pg.ScatterPlotItem()
        self.rois_red = pg.ScatterPlotItem()
        self.rois_hl = pg.ScatterPlotItem()

        self.show()

    def refresh(self):

        self.p0.removeItem(self.rois_green)
        self.p0.removeItem(self.rois_red)
        self.draw_rois()

    def open_file(self):

        self.folder = QtWidgets.QFileDialog.getExistingDirectory(self,\
                                    "Choose datafolder",
                                    FOLDERS[self.folderB.currentText()])
        # self.folder = '/home/yann/UNPROCESSED/TSeries-001'

        if self.folder!='':

            self.stat = np.load(os.path.join(self.folder, 'suite2p', 'plane0', 'stat.npy'), allow_pickle=True)
            self.redcell = np.load(os.path.join(self.folder, 'suite2p', 'plane0', 'redcell.npy'), allow_pickle=True)
            self.iscell = np.load(os.path.join(self.folder, 'suite2p', 'plane0', 'iscell.npy'), allow_pickle=True)
            self.ops = np.load(os.path.join(self.folder, 'suite2p', 'plane0', 'ops.npy'), allow_pickle=True).item()

            self.draw_image()
            self.draw_rois()

    def draw_image(self):
        
        self.pimg.setImage(self.ops[self.imgB.currentText()]**.25)


    def draw_rois(self, size=4, t=np.arange(20)):

        x_green, y_green, x_red, y_red = [], [], [], []
        for i in range(len(self.stat)):
            if self.iscell[i,0]:
                xmean = np.mean(self.stat[i]['xpix'])
                ymean = np.mean(self.stat[i]['ypix'])
                if self.redcell[i,0]:
                    x_red += list(xmean+size*np.cos(2*np.pi*t/len(t)))
                    y_red += list(ymean+size*np.sin(2*np.pi*t/len(t)))
                else:
                    x_green += list(xmean+size*np.cos(2*np.pi*t/len(t)))
                    y_green += list(ymean+size*np.sin(2*np.pi*t/len(t)))
                    # x_green += list(self.stat[i]['xpix'])
                    # y_green += list(self.stat[i]['ypix'])
        
        self.rois_red.setData(x_red, y_red, size=3, brush=pg.mkBrush(255,0,0))
        self.rois_green.setData(x_green, y_green, size=1, brush=pg.mkBrush(0,255,0))
        self.p0.addItem(self.rois_red)
        self.p0.addItem(self.rois_green)

    def highlight_roi(self, size=6, t=np.arange(20)):
        
        x, y = [], []
        if (self.roi_index>=0) and (self.roi_index<len(self.stat)):
            xmean = np.mean(self.stat[self.roi_index]['xpix'])
            ymean = np.mean(self.stat[self.roi_index]['ypix'])
            x += list(xmean+size*np.cos(2*np.pi*t/len(t)))
            y += list(ymean+size*np.sin(2*np.pi*t/len(t)))
        else:
            print(self.roi_index, 'out of bounds')
            
        self.rois_hl.setData(x, y, size=3, brush=pg.mkBrush(0,0,255))
        self.p0.addItem(self.rois_hl)
        
    def next_roi(self):

        self.roi_index +=1
        print(self.iscell[self.roi_index,0])
        while (not self.iscell[self.roi_index,0]) and (self.roi_index<len(self.stat)):
            self.roi_index +=1
        self.highlight_roi()

    def process(self):

        self.roi_index -=1
        while (not self.iscell[self.roi_index,0]) and (self.roi_index>0):
            self.roi_index -=1
        self.highlight_roi()
            

    def hitting_space(self):
        if self.redcell[self.roi_index,0]:
            self.redcell[self.roi_index,0] = 0.
        else:
            self.redcell[self.roi_index,0] = 1.
        self.draw_rois()

    def save(self):
        np.save(os.path.join(self.folder, 'suite2p', 'plane0', 'redcell_manual.npy'), self.redcell)
        print('manual processing saved as:', os.path.join(self.folder, 'suite2p', 'plane0', 'redcell_manual.npy'))
        

def run(app, args=None, parent=None):
    return RCGwindow(app,
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
    main = RCGwindow(app,
                      args=args)
    sys.exit(app.exec_())



    
