import sys, os, shutil, glob, time, subprocess, pathlib
import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from misc.folders import FOLDERS
from misc.style import set_dark_style, set_app_icon
from misc.guiparts import NewWindow, Slider

KEYS = ['meanImg_chan2', 'meanImg', 'max_proj', 'meanImgE',
        'meanImg_chan2-X*meanImg', 'meanImg_chan2/(X*meanImg)']

class RCGwindow(NewWindow):
    
    def __init__(self, app,
                 args=None,
                 parent=None,
                 debug=False):
        """
        Red-Cell selection GUI
        """
        self.app = app
        self.debug = debug
        
        super(RCGwindow, self).__init__(i=3,
                                        title='red-cell gui')

        self.nextSc = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+N'), self)
        self.nextSc.activated.connect(self.next_roi)

        self.saveSc = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+S'), self)
        self.saveSc.activated.connect(self.save)

        self.Sc1= QtWidgets.QShortcut(QtGui.QKeySequence('1'), self)
        self.Sc1.activated.connect(self.switch_to_1)
        self.Sc2= QtWidgets.QShortcut(QtGui.QKeySequence('2'), self)
        self.Sc2.activated.connect(self.switch_to_2)
        self.Sc3= QtWidgets.QShortcut(QtGui.QKeySequence('3'), self)
        self.Sc3.activated.connect(self.switch_to_3)
        self.Sc4= QtWidgets.QShortcut(QtGui.QKeySequence('4'), self)
        self.Sc4.activated.connect(self.switch_to_4)
        self.Sc5= QtWidgets.QShortcut(QtGui.QKeySequence('5'), self)
        self.Sc5.activated.connect(self.switch_to_5)
        self.Sc6= QtWidgets.QShortcut(QtGui.QKeySequence('6'), self)
        self.Sc6.activated.connect(self.switch_to_6)
        
        self.roiSc = QtWidgets.QShortcut(QtGui.QKeySequence('Space'), self)
        self.roiSc.activated.connect(self.switch_roi_display)
        
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
        self.imgB.addItems(KEYS)
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

        self.rstRoiB = QtGui.QPushButton('reset ALL rois to green')
        self.rstRoiB.clicked.connect(self.reset_all_to_green)
        
        self.roiShapeCheckBox = QtWidgets.QCheckBox("ROIs as circle")
        self.roiShapeCheckBox.setChecked(True)

        for wdgt, index in zip([self.folderB, self.load, self.imgB, self.nextB, self.prevB,
                                self.switchB, self.saveB, self.rstRoiB, self.roiShapeCheckBox],
                               [1,2,6,10,11,12,16,21, 23]):
            wdgt.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
            self.l0.addWidget(wdgt,index,0,1,3)
        
        self.rois_green = pg.ScatterPlotItem()
        self.rois_red = pg.ScatterPlotItem()
        self.rois_hl = pg.ScatterPlotItem()

        self.folder, self.rois_on = '', True
        self.show()

    def reset_all_to_green(self):
        for i in range(len(self.stat)):
            self.redcell[i,0] = 0.
        self.refresh()
        
    def refresh(self):

        self.p0.removeItem(self.rois_green)
        self.p0.removeItem(self.rois_red)
        if self.rois_on:
            self.draw_rois()

    def switch_to(self, i):
        self.imgB.setCurrentText(KEYS[i-1])
        self.draw_image()

    def switch_to_1(self):
        self.switch_to(1)
    def switch_to_2(self):
        self.switch_to(2)
    def switch_to_3(self):
        self.switch_to(3)
    def switch_to_4(self):
        self.switch_to(4)
    def switch_to_5(self):
        self.switch_to(5)
    def switch_to_6(self):
        self.switch_to(6)

    def switch_roi_display(self):
        self.rois_on = (not self.rois_on)
        self.refresh()

    def open_file(self):

        self.folder = QtWidgets.QFileDialog.getExistingDirectory(self,\
                                    "Choose datafolder",
                                    FOLDERS[self.folderB.currentText()])
        # self.folder = '/home/yann/UNPROCESSED/TSeries-001'

        self.load_file()

    def build_linear_interpolation(self):

        x, y = np.array(self.ops['meanImg']).flatten(), np.array(self.ops['meanImg_chan2']).flatten()
        p = np.polyfit(x, y, 1)

        if self.debug:
            import matplotlib.pylab as plt
            plt.scatter(x, y)
            plt.plot(x, np.polyval(p, x), color='r')
            plt.xlabel('Ch1');plt.ylabel('Ch2')
            plt.show()

        self.ops['meanImg_chan2-X*meanImg'] = np.clip(np.array(self.ops['meanImg_chan2'])-np.polyval(p, np.array(self.ops['meanImg'])), 0, np.inf)
        self.ops['meanImg_chan2/(X*meanImg)'] = np.array(self.ops['meanImg_chan2'])/np.clip(np.polyval(p, np.array(self.ops['meanImg'])), 1, np.inf)
        
        
    def load_file(self):

        if self.folder!='':

            self.stat = np.load(os.path.join(self.folder, 'suite2p', 'plane0', 'stat.npy'), allow_pickle=True)
            self.redcell = np.load(os.path.join(self.folder, 'suite2p', 'plane0', 'redcell.npy'), allow_pickle=True)
            self.iscell = np.load(os.path.join(self.folder, 'suite2p', 'plane0', 'iscell.npy'), allow_pickle=True)
            self.ops = np.load(os.path.join(self.folder, 'suite2p', 'plane0', 'ops.npy'), allow_pickle=True).item()

            self.draw_image()
            self.draw_rois()

            self.build_linear_interpolation()
        else:
            print('empty folder ...')
        
    def draw_image(self):
        
        self.pimg.setImage(self.ops[self.imgB.currentText()]**.25)


    def add_single_roi_pix(self, i, size=4, t=np.arange(20)):

        if self.roiShapeCheckBox.isChecked():
            # drawing circles:
            
            xmean = np.mean(self.stat[i]['xpix'])
            ymean = np.mean(self.stat[i]['ypix'])
            
            if self.redcell[i,0]:
                self.x_red += list(xmean+size*np.cos(2*np.pi*t/len(t)))
                self.y_red += list(ymean+size*np.sin(2*np.pi*t/len(t)))
            else:
                self.x_green += list(xmean+size*np.cos(2*np.pi*t/len(t)))
                self.y_green += list(ymean+size*np.sin(2*np.pi*t/len(t)))

        else:
            # full ROI
            if self.redcell[i,0]:
                self.x_red += list(self.stat[i]['xpix'])
                self.y_red += list(self.stat[i]['ypix'])
            else:
                self.x_green += list(self.stat[i]['xpix'])
                self.y_green += list(self.stat[i]['ypix'])
        
    def draw_rois(self):

        self.x_green, self.y_green = [], []
        self.x_red, self.y_red = [], []
        
        for i in range(len(self.stat)):
            if self.iscell[i,0]:
                self.add_single_roi_pix(i)
        
        self.rois_red.setData(self.x_red, self.y_red, size=3, brush=pg.mkBrush(255,0,0))
        self.rois_green.setData(self.x_green, self.y_green, size=1, brush=pg.mkBrush(0,255,0))
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
    parser.add_argument('-d', "--debug", action="store_true")
    args = parser.parse_args()
    app = QtWidgets.QApplication(sys.argv)
    build_dark_palette(app)
    main = RCGwindow(app,
                      args=args, debug=args.debug)
    if args.datafile!='':
        main.folder = args.datafile
        main.load_file()
    sys.exit(app.exec_())



    
