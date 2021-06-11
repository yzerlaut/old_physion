import sys, os, shutil, glob, time, subprocess
import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
import pathlib
from scipy.interpolate import interp1d

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from misc.folders import FOLDERS
from misc.style import set_dark_style, set_app_icon
from assembling.tools import load_FaceCamera_data
from dataviz.guiparts import NewWindow
from pupil import guiparts
from whisking import roi, process

class MainWindow(NewWindow):
    
    def __init__(self, app,
                 args=None,
                 parent=None,
                 spatial_subsampling=1,
                 time_subsampling=1):
        """
        sampling in Hz
        """
        self.app = app
        
        super(MainWindow, self).__init__(i=1,
                                         title='Face-motion/Whisking tracking')
        
        # self.refSc = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+R'), self)
        # self.refSc.activated.connect(self.jump_to_frame)
        
        # self.refEx = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+E'), self)
        # self.refEx.activated.connect(self.exclude_outlier)
        # self.refPr = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+P'), self)
        # self.refPr.activated.connect(self.process_outliers)
        # self.refc1 = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+1'), self)
        # self.refc1.activated.connect(self.set_cursor_1)
        # self.refc2 = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+2'), self)
        # self.refc2.activated.connect(self.set_cursor_2)
        # self.refc3 = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+3'), self)
        # self.refc3.activated.connect(self.process_outliers)

        
        self.minView = False
        self.showwindow()
        
        pg.setConfigOptions(imageAxisOrder='row-major')
        
        self.spatial_subsampling = spatial_subsampling
        self.time_subsampling = time_subsampling
        
        self.cwidget = QtGui.QWidget(self)
        self.setCentralWidget(self.cwidget)
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

        # image ROI
        self.pFace = self.win.addViewBox(lockAspect=False,row=0,col=1,invertY=True,
                                         border=[100,100,100])
        self.pFace.setAspectLocked()
        #self.p0.setMouseEnabled(x=False,y=False)
        self.pFace.setMenuEnabled(False)
        self.pFaceimg = pg.ImageItem(None)
        self.pFace.addItem(self.pFaceimg)

        # roi initializations
        self.iROI = 0
        self.nROIs = 0
        self.saturation = 255
        self.ROI = None
        self.pupil = None
        self.iframes, self.times, self.Pr1, self.Pr2 = [], [], [], []
        self.times, self.imgfolder, self.nframes, self.FILES = None, None, None, None

        # saturation sliders
        self.sl = guiparts.Slider(0, self)
        self.l0.addWidget(self.sl,1,6,1,7)
        qlabel= QtGui.QLabel('saturation')
        qlabel.setStyleSheet('color: white;')
        self.l0.addWidget(qlabel, 0,8,1,3)

        # # # adding blanks ("corneal reflections, ...")
        # self.roiBtn = QtGui.QPushButton('set ROI')
        # self.l0.addWidget(self.roiBtn, 1, 8+6, 1, 1)
        # self.roiBtn.setEnabled(True)
        # self.roiBtn.clicked.connect(self.add_ROI)
        
        # # fit pupil
        # self.fit_pupil = QtGui.QPushButton('fit Pupil [Ctrl+F]')
        # self.l0.addWidget(self.fit_pupil, 1, 9+6, 1, 1)
        # # self.fit_pupil.setEnabled(True)
        # self.fit_pupil.clicked.connect(self.fit_pupil_size)
        # # choose pupil shape
        # self.pupil_shape = QtGui.QComboBox(self)
        # self.pupil_shape.addItem("Ellipse fit")
        # self.pupil_shape.addItem("Circle fit")
        # self.l0.addWidget(self.pupil_shape, 1, 10+6, 1, 1)
        # reset
        self.reset_btn = QtGui.QPushButton('reset')
        self.l0.addWidget(self.reset_btn, 1, 11+6, 1, 1)
        self.reset_btn.clicked.connect(self.reset)
        self.reset_btn.setEnabled(True)

        
        self.debugBtn = QtGui.QPushButton('- Debug -')
        self.l0.addWidget(self.debugBtn, 2, 11+6, 1, 1)
        self.debugBtn.setEnabled(True)
        self.debugBtn.clicked.connect(self.debug)

        self.data = None
        self.scatter, self.fit= None, None # the pupil size contour

        self.p1 = self.win.addPlot(name='plot1',row=1,col=0, colspan=2, rowspan=4,
                                   title='*face motion*')
        self.p1.setMouseEnabled(x=True,y=False)
        self.p1.setMenuEnabled(False)
        self.p1.hideAxis('left')
        self.scatter = pg.ScatterPlotItem()
        self.p1.addItem(self.scatter)
        self.p1.setLabel('bottom', 'time (frame #)')
        self.xaxis = self.p1.getAxis('bottom')
        self.p1.autoRange(padding=0.01)
        
        self.win.ci.layout.setRowStretchFactor(0,5)
        self.movieLabel = QtGui.QLabel("No datafile chosen")
        self.movieLabel.setStyleSheet("color: white;")
        self.l0.addWidget(self.movieLabel,0,1,1,5)


        # create frame slider
        self.timeLabel = QtGui.QLabel("time : ")
        self.timeLabel.setStyleSheet("color: white;")
        # self.timeLabel.setFixedWidth(300)
        self.currentTime = QtGui.QLineEdit()
        self.currentTime.setText('0 s')
        self.currentTime.setFixedWidth(70)
        # self.currentTime.returnPressed.connect(self.set_precise_time)
        
        self.frameSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.frameSlider.setMinimum(0)
        self.frameSlider.setMaximum(200)
        self.frameSlider.setTickInterval(1)
        self.frameSlider.setTracking(False)
        self.frameSlider.valueChanged.connect(self.go_to_frame)

        istretch = 23
        iplay = istretch+15
        iconSize = QtCore.QSize(20, 20)

        self.folderB = QtWidgets.QComboBox(self)
        self.folderB.setMinimumWidth(150)
        self.folderB.addItems(FOLDERS.keys())
        
        self.processBtn = QtGui.QPushButton('process data')
        self.processBtn.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.processBtn.clicked.connect(self.process)

        # self.interpolate = QtGui.QPushButton('interpolate')
        # self.interpolate.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        # self.interpolate.clicked.connect(self.interpolate_data)

        self.motionCheckBox = QtGui.QCheckBox("display motion frames")
        self.motionCheckBox.setStyleSheet("color: gray;")
        
        self.runAsSubprocess = QtGui.QPushButton('run as subprocess')
        self.runAsSubprocess.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.runAsSubprocess.clicked.connect(self.run_as_subprocess)
        # self.runAsSubprocess.setEnabled(True)

        self.load = QtGui.QPushButton('  load data [Ctrl+O]  \u2b07')
        self.load.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.load.clicked.connect(self.open_file)

        sampLabel1 = QtGui.QLabel("spatial subsampling")
        sampLabel1.setStyleSheet("color: gray;")
        sampLabel1.setFixedWidth(220)
        self.SsamplingBox = QtGui.QLineEdit()
        self.SsamplingBox.setText(str(self.spatial_subsampling))
        self.SsamplingBox.setFixedWidth(30)

        sampLabel2 = QtGui.QLabel("temporal subsampling")
        sampLabel2.setStyleSheet("color: gray;")
        sampLabel2.setFixedWidth(220)
        self.TsamplingBox = QtGui.QLineEdit()
        self.TsamplingBox.setText(str(self.time_subsampling))
        self.TsamplingBox.setFixedWidth(30)
        
        self.addROI = QtGui.QPushButton("set ROI")
        self.addROI.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.addROI.clicked.connect(self.add_ROI)

        self.saveData = QtGui.QPushButton('save data')
        self.saveData.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.saveData.clicked.connect(self.save_data)

        iconSize = QtCore.QSize(30, 30)
        self.playButton = QtGui.QToolButton()
        self.playButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPlay))
        self.playButton.setIconSize(iconSize)
        self.playButton.setToolTip("Play")
        self.playButton.setCheckable(True)
        self.pauseButton = QtGui.QToolButton()
        self.pauseButton.setCheckable(True)
        self.pauseButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPause))
        self.pauseButton.setIconSize(iconSize)
        self.pauseButton.setToolTip("Pause")

        btns = QtGui.QButtonGroup(self)
        btns.addButton(self.playButton,0)
        btns.addButton(self.pauseButton,1)

        self.l0.addWidget(self.folderB,1,0,1,3)
        self.l0.addWidget(self.load,2,0,1,3)
        self.l0.addWidget(self.addROI,14,0,1,3)
        self.l0.addWidget(self.saveData, 16, 0, 1, 3)
        self.l0.addWidget(self.processBtn, 20, 0, 1, 3)
        self.l0.addWidget(self.runAsSubprocess, 21, 0, 1, 3)
        self.l0.addWidget(self.motionCheckBox, 18, 0, 1, 3)
        
        self.l0.addWidget(sampLabel1, 8, 0, 1, 3)
        self.l0.addWidget(self.SsamplingBox, 8, 2, 1, 3)
        self.l0.addWidget(sampLabel2, 9, 0, 1, 3)
        self.l0.addWidget(self.TsamplingBox, 9, 2, 1, 3)

        self.l0.addWidget(QtGui.QLabel(''),istretch,0,1,3)
        self.l0.setRowStretch(istretch,1)
        self.l0.addWidget(self.timeLabel, istretch+10,0,1,3)
        self.l0.addWidget(self.currentTime, istretch+10,1,1,3)
        self.l0.addWidget(self.frameSlider, istretch+15,3,1,15)

        self.l0.addWidget(QtGui.QLabel(''),17,2,1,1)
        self.l0.setRowStretch(16,2)
        # self.l0.addWidget(ll, istretch+3+k+1,0,1,4)
        self.timeLabel.setEnabled(True)
        self.frameSlider.setEnabled(True)
        
        self.nframes = 0
        self.cframe = 0

        self.show()


    def add_ROI(self):
        self.ROI = roi.faceROI(moveable=True, parent=self)
        

    def open_file(self):

        self.cframe = 0
        
        folder = QtWidgets.QFileDialog.getExistingDirectory(self,\
                                    "Choose datafolder",
                                    FOLDERS[self.folderB.currentText()])

        if folder!='':
            
            self.datafolder = folder
            
            if os.path.isdir(os.path.join(folder, 'FaceCamera-imgs')):
                
                self.reset()
                self.imgfolder = os.path.join(self.datafolder, 'FaceCamera-imgs')
                process.load_folder(self) # in init: self.times, _, self.nframes, ...
                
            else:
                self.times, self.imgfolder, self.nframes, self.FILES = None, None, None, None
                print(' /!\ no raw FaceCamera data found ...')

            if os.path.isfile(os.path.join(self.datafolder, 'whisking.npy')):
                
                self.data = np.load(os.path.join(self.datafolder, 'whisking.npy'),
                                    allow_pickle=True).item()
                
                if (self.nframes is None) and ('frame' in self.data):
                    self.nframes = self.data['frame'].max()

                if 'ROI' in self.data:
                    self.ROI = roi.faceROI(moveable=True, parent=self,
                                           pos=self.data['ROI'])
                    

                if 'ROIsaturation' in self.data:
                    self.sl.setValue(int(self.data['ROIsaturation']))
                    
                if 'motion' in self.data:
                    self.plot_motion_trace()
                
            else:
                self.data = None

            if self.times is not None:
                self.jump_to_frame()
                self.timeLabel.setEnabled(True)
                self.frameSlider.setEnabled(True)
                self.movieLabel.setText(folder)


    def reset(self):
        if self.ROI is not None:
            self.ROI.remove(self)
        self.saturation = 255
        self.cframe1, self.cframe2 = 0, -1
        self.data = None


    def save_data(self):

        if self.data is None:
            self.data = {}
            
        if self.ROI is not None:
            self.data['ROI'] = self.ROI.position(self)

        np.save(os.path.join(self.datafolder, 'whisking.npy'), self.data)
        
        print('data saved as: "%s"' % os.path.join(self.datafolder, 'whisking.npy'))
        
        
    def go_to_frame(self):
        if self.FILES is not None:
            i1, i2 = self.xaxis.range
            self.cframe = max([0, int(i1+(i2-i1)*float(self.frameSlider.value()/200.))])
            self.jump_to_frame()


    def refresh(self):
        self.jump_to_frame()
        
    def jump_to_frame(self):

        if self.FILES is not None:
            # full image 
            self.fullimg = np.load(os.path.join(self.imgfolder,
                                                self.FILES[self.cframe]))
            self.pimg.setImage(self.fullimg)


            # zoomed image
            if self.ROI is not None:

                process.set_ROI_area(self)

                if self.motionCheckBox.isChecked():
                    self.fullimg2 = np.load(os.path.join(self.imgfolder,
                                                         self.FILES[self.cframe+1]))
                    
                    self.img = self.fullimg2[self.zoom_cond].reshape(self.Nx, self.Ny)-\
                        self.fullimg[self.zoom_cond].reshape(self.Nx, self.Ny)
                else:
                    
                    self.img = self.fullimg[self.zoom_cond].reshape(self.Nx, self.Ny)
                
                self.pFaceimg.setImage(self.img)
                

        if self.scatter is not None:
            self.p1.removeItem(self.scatter)
            
        if self.data is not None:

            self.iframe = np.argmin((self.data['frame']-self.cframe)**2)
            self.scatter.setData([self.cframe],
                                 [self.data['motion'][self.iframe]],
                                 size=10, brush=pg.mkBrush(255,255,255))
            self.p1.addItem(self.scatter)
            self.p1.show()

            self.currentTime.setText('%.1f s' % (self.data['t'][self.iframe]-self.data['t'][0]))

            
        self.win.show()
        self.show()

    def process(self):

        process.set_ROI_area(self)
        frames, motion = process.compute_motion(self,
                                        time_subsampling=int(self.TsamplingBox.text()),
                                        with_ProgressBar=True)
        self.data = {'frame':frames, 't':self.times[frames], 'motion':motion}
        self.plot_motion_trace()
        

    
    def run_as_subprocess(self):

        self.save_data()
        process_script = os.path.join(str(pathlib.Path(__file__).resolve().parents[0]), 'process.py')
        print('python %s -df %s' % (process_script, self.datafolder))
        p = subprocess.Popen('python %s -df %s' % (process_script, self.datafolder),
                             # stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             shell=True)
        
    def plot_motion_trace(self):
        self.p1.clear()
        self.p1.plot(self.data['frame'],
                     self.data['motion'], pen=(0,0,255))
        self.p1.setRange(xRange=(0, self.nframes),
                         yRange=(self.data['motion'].min()-.1,
                                 self.data['motion'].max()+.1),
                         padding=0.0)
        self.p1.show()
    
    def debug(self):
        pass
    

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



    
