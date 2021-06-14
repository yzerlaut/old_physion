import sys, os, shutil, glob, time, subprocess, pathlib
import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
from scipy.interpolate import interp1d

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from pupil import guiparts, process, roi
from misc.folders import FOLDERS
from misc.style import set_dark_style, set_app_icon
from assembling.saving import from_folder_to_datetime, check_datafolder
from assembling.tools import load_FaceCamera_data
from dataviz.plots import convert_index_to_time

class MainWindow(QtWidgets.QMainWindow):
    
    def __init__(self, app,
                 args=None,
                 parent=None,
                 gaussian_smoothing=0,
                 cm_scale_px=570,
                 subsampling=1000):
        """
        sampling in Hz
        """
        self.app = app
        
        super(MainWindow, self).__init__()

        self.setGeometry(500,150,400,400)

        self.process_script = os.path.join(str(pathlib.Path(__file__).resolve().parents[0]),'process.py')
        self.script = os.path.join(str(pathlib.Path(__file__).resolve().parents[1]), 'script.sh') # for batch processing

        # adding a "quit" keyboard shortcut
        self.quitSc = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+Q'), self)
        self.quitSc.activated.connect(self.quit)
        self.maxSc = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+M'), self)
        self.maxSc.activated.connect(self.showwindow)
        self.fitSc = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+F'), self)
        self.fitSc.activated.connect(self.fit_pupil_size)
        self.loadSc = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+O'), self)
        self.loadSc.activated.connect(self.load_data)
        self.refSc = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+R'), self)
        self.refSc.activated.connect(self.jump_to_frame)
        # self.refEx = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+E'), self)
        # self.refEx.activated.connect(self.exclude_outlier)
        self.refPr = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+P'), self)
        self.refPr.activated.connect(self.process_ROIs)
        self.refS = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+S'), self)
        self.refS.activated.connect(self.save_pupil_data)
        self.refc1 = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+1'), self)
        self.refc1.activated.connect(self.set_cursor_1)
        self.refc2 = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+2'), self)
        self.refc2.activated.connect(self.set_cursor_2)
        self.refc3 = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+3'), self)
        self.refc3.activated.connect(self.process_outliers)

        self.add2Bash = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+B'), self)
        self.add2Bash.activated.connect(self.add_to_bash_script)
        
        self.minView = False
        self.showwindow()
        
        pg.setConfigOptions(imageAxisOrder='row-major')
        
        self.gaussian_smoothing = gaussian_smoothing
        self.subsampling = subsampling
        self.cm_scale_px = cm_scale_px
        
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
        self.p0 = self.win.addViewBox(lockAspect=False,
                                      row=0,col=0,#border=[100,100,100],
                                      invertY=True)

        self.p0.setMouseEnabled(x=False,y=False)
        self.p0.setMenuEnabled(False)
        self.pimg = pg.ImageItem()
        self.p0.setAspectLocked()
        self.p0.addItem(self.pimg)

        # image ROI
        self.pPupil = self.win.addViewBox(lockAspect=True,#row=0,col=1,
                                          # border=[100,100,100],
                                          invertY=True)
        #self.p0.setMouseEnabled(x=False,y=False)
        self.pPupil.setMenuEnabled(False)
        self.pPupilimg = pg.ImageItem(None)
        self.pPupil.addItem(self.pPupilimg)
        self.pupilContour = pg.ScatterPlotItem()
        self.pPupil.addItem(self.pupilContour)
        self.pupilCenter = pg.ScatterPlotItem()
        self.pPupil.addItem(self.pupilCenter)

        # roi initializations
        self.iROI = 0
        self.nROIs = 0
        self.saturation = 255
        self.ROI = None
        self.pupil = None
        self.iframes, self.times, self.Pr1, self.Pr2 = [], [], [], []

        # saturation sliders
        self.sl = guiparts.Slider(0, self)
        self.l0.addWidget(self.sl,1,6,1,7)
        qlabel= QtGui.QLabel('saturation')
        qlabel.setStyleSheet('color: white;')
        self.l0.addWidget(qlabel, 0,8,1,3)

        # adding blanks ("corneal reflections, ...")
        self.reflector = QtGui.QPushButton('add blank')
        self.l0.addWidget(self.reflector, 1, 8+6, 1, 1)
        # self.reflector.setEnabled(True)
        self.reflector.clicked.connect(self.add_reflectROI)
        # fit pupil
        self.fit_pupil = QtGui.QPushButton('fit Pupil [Ctrl+F]')
        self.l0.addWidget(self.fit_pupil, 1, 9+6, 1, 1)
        # self.fit_pupil.setEnabled(True)
        self.fit_pupil.clicked.connect(self.fit_pupil_size)
        # choose pupil shape
        self.pupil_shape = QtGui.QComboBox(self)
        self.pupil_shape.addItem("Ellipse fit")
        self.pupil_shape.addItem("Circle fit")
        self.l0.addWidget(self.pupil_shape, 1, 10+6, 1, 1)
        # reset
        self.reset_btn = QtGui.QPushButton('reset')
        self.l0.addWidget(self.reset_btn, 1, 11+6, 1, 1)
        self.reset_btn.clicked.connect(self.reset)
        # self.reset_btn.setEnabled(True)
        # draw pupil
        self.refresh_pupil = QtGui.QPushButton('Refresh [Ctrl+R]')
        self.l0.addWidget(self.refresh_pupil, 2, 11+6, 1, 1)
        self.refresh_pupil.setEnabled(True)
        self.refresh_pupil.clicked.connect(self.jump_to_frame)

        self.data = None
        self.rROI= []
        self.reflectors=[]
        self.scatter, self.fit= None, None # the pupil size contour

        self.p1 = self.win.addPlot(name='plot1',row=1,col=0, colspan=2, rowspan=4,
                                   title='Pupil diameter')
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
        self.timeLabel = QtGui.QLabel("Current time (seconds):")
        self.timeLabel.setStyleSheet("color: white;")
        self.currentTime = QtGui.QLineEdit()
        self.currentTime.setText('0')
        self.currentTime.setValidator(QtGui.QDoubleValidator(0, 100000, 2))
        self.currentTime.setFixedWidth(50)
        self.currentTime.returnPressed.connect(self.set_precise_time)
        
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

        self.process = QtGui.QPushButton('process data [Ctrl+P]')
        self.process.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.process.clicked.connect(self.process_ROIs)

        self.cursor1 = QtGui.QPushButton('Set Cursor 1 [Ctrl+1]')
        self.cursor1.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.cursor1.clicked.connect(self.set_cursor_1)

        self.cursor2 = QtGui.QPushButton('Set Cursor 2 [Ctrl+2]')
        self.cursor2.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.cursor2.clicked.connect(self.set_cursor_2)
        
        self.runAsSubprocess = QtGui.QPushButton('run as subprocess')
        self.runAsSubprocess.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.runAsSubprocess.clicked.connect(self.run_as_subprocess)

        self.load = QtGui.QPushButton('  load data [Ctrl+O]  \u2b07')
        self.load.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.load.clicked.connect(self.load_data)

        sampLabel = QtGui.QLabel("Subsampling (frame)")
        sampLabel.setStyleSheet("color: gray;")
        self.samplingBox = QtGui.QLineEdit()
        self.samplingBox.setText(str(self.subsampling))
        self.samplingBox.setFixedWidth(50)

        smoothLabel = QtGui.QLabel("Smoothing (px)")
        smoothLabel.setStyleSheet("color: gray;")
        self.smoothBox = QtGui.QLineEdit()
        self.smoothBox.setText(str(self.gaussian_smoothing))
        self.smoothBox.setFixedWidth(30)

        scaleLabel = QtGui.QLabel("1cm = (px)")
        scaleLabel.setStyleSheet("color: gray;")
        self.scaleBox = QtGui.QLineEdit()
        self.scaleBox.setText(str(self.cm_scale_px))
        self.scaleBox.setFixedWidth(30)

        self.addROI = QtGui.QPushButton("add Pupil-ROI")
        self.addROI.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.addROI.clicked.connect(self.add_ROI)

        self.saverois = QtGui.QPushButton('save data [Ctrl+S]')
        self.saverois.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.saverois.clicked.connect(self.save_ROIs)

        # self.addOutlier = QtGui.QPushButton('exclude outlier [Ctrl+E]')
        # self.addOutlier.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        # self.addOutlier.clicked.connect(self.exclude_outlier)

        self.processOutliers = QtGui.QPushButton('Set blinking interval')
        self.processOutliers.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.processOutliers.clicked.connect(self.process_outliers)

        self.printSize = QtGui.QPushButton('print ROI size')
        self.printSize.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.printSize.clicked.connect(self.print_size)
        
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
        self.l0.addWidget(sampLabel, 8, 0, 1, 3)
        self.l0.addWidget(self.samplingBox, 8, 2, 1, 3)
        self.l0.addWidget(smoothLabel, 9, 0, 1, 3)
        self.l0.addWidget(self.smoothBox, 9, 2, 1, 3)
        self.l0.addWidget(scaleLabel, 10, 0, 1, 3)
        self.l0.addWidget(self.scaleBox, 10, 2, 1, 3)
        self.l0.addWidget(self.addROI,14,0,1,3)
        self.l0.addWidget(self.process, 16, 0, 1, 3)
        self.l0.addWidget(self.runAsSubprocess, 17, 0, 1, 3)
        self.l0.addWidget(self.saverois, 19, 0, 1, 3)
        self.l0.addWidget(self.cursor1, 23, 0, 1, 3)
        self.l0.addWidget(self.cursor2, 24, 0, 1, 3)
        self.l0.addWidget(self.processOutliers, 25, 0, 1, 3)
        self.l0.addWidget(self.printSize, 26, 0, 1, 3)

        self.l0.addWidget(QtGui.QLabel(''),istretch,0,1,3)
        self.l0.setRowStretch(istretch,1)
        self.l0.addWidget(self.timeLabel, istretch+13,0,1,3)
        self.l0.addWidget(self.currentTime, istretch+14,0,1,3)
        self.l0.addWidget(self.frameSlider, istretch+15,3,1,15)

        self.l0.addWidget(QtGui.QLabel(''),17,2,1,1)
        self.l0.setRowStretch(16,2)
        # self.l0.addWidget(ll, istretch+3+k+1,0,1,4)
        self.updateFrameSlider()
        
        self.nframes = 0
        self.cframe, self.cframe1, self.cframe2, = 0, 0, 0

        self.updateTimer = QtCore.QTimer()
        
        self.win.show()
        self.show()
        self.processed = False

        # self.datafile = args.datafile
        self.show()

    def showwindow(self):
        if self.minView:
            self.minView = self.maxview()
        else:
            self.minView = self.minview()
    def maxview(self):
        self.showFullScreen()
        return False
    def minview(self):
        self.showNormal()
        return True
    
    def load_data(self):

        self.cframe = 0
        
        folder = QtWidgets.QFileDialog.getExistingDirectory(self,\
                                    "Choose datafolder",
                                    FOLDERS[self.folderB.currentText()])
        # folder = '/home/yann/UNPROCESSED/13-26-53/'

        if folder!='':
            
            self.datafolder = folder
            
            if os.path.isdir(os.path.join(folder, 'FaceCamera-imgs')):
                
                # self.reset()
                self.imgfolder = os.path.join(self.datafolder, 'FaceCamera-imgs')
                self.times, self.FILES, self.nframes, self.Lx, self.Ly = load_FaceCamera_data(self.imgfolder,
                                                                                              t0=0, verbose=True)
            else:
                self.times, self.imgfolder, self.nframes, self.FILES = None, None, None, None
                print(' /!\ no raw FaceCamera data found ...')

            if os.path.isfile(os.path.join(self.datafolder, 'pupil.npy')):
                
                self.data = np.load(os.path.join(self.datafolder, 'pupil.npy'),
                                    allow_pickle=True).item()
                
                if self.nframes is None:
                    self.nframes = self.data['frame'].max()
                
                self.smoothBox.setText('%i' % self.data['gaussian_smoothing'])

                self.sl.setValue(int(self.data['ROIsaturation']))
                self.ROI = roi.sROI(parent=self,
                                    pos=roi.ellipse_props_to_ROI(self.data['ROIellipse']))
                self.plot_pupil_trace()
                
            else:
                self.data = None

            if self.times is not None:
                self.jump_to_frame()
                self.timeLabel.setEnabled(True)
                self.frameSlider.setEnabled(True)
                self.updateFrameSlider()
                self.currentTime.setValidator(\
                                              QtGui.QDoubleValidator(0, self.nframes, 2))
                self.movieLabel.setText(folder)


    def reset(self):
        for r in self.rROI:
            r.remove(self)
        if self.ROI is not None:
            self.ROI.remove(self)
        if self.pupil is not None:
            self.pupil.remove(self)
        if self.fit is not None:
            self.fit.remove(self)
        self.ROI, self.rROI = None, []
        self.fit = None
        self.reflectors=[]
        self.saturation = 255
        self.iROI=0
        self.nROIs=0
        self.cframe1, self.cframe2 = 0, -1
        
    def add_reflectROI(self):
        self.rROI.append(roi.reflectROI(len(self.rROI), moveable=True, parent=self))

    def draw_pupil(self):
        self.pupil = roi.pupilROI(moveable=True, parent=self)

    def print_size(self):
        print('x, y, sx, sy, angle = ', self.ROI.extract_props())

    def add_ROI(self):

        if self.ROI is not None:
            self.ROI.remove(self)
        for r in self.rROI:
            r.remove(self)
        self.ROI = roi.sROI(parent=self)
        self.rROI = []
        self.reflectors = []

    def process_outliers(self):

        if self.data is not None and (self.cframe1!=0) and (self.cframe2!=0):
            
            i1 = np.arange(len(self.data['frame']))[self.data['frame']>=self.cframe1][0]
            i2 = np.arange(len(self.data['frame']))[self.data['frame']>=self.cframe2][0]
            self.data['diameter'][i1:i2] = 0
            if i1>0:
                new_i1 = i1-1
            else:
                new_i1 = i2
            if i2<len(self.data['frame'])-1:
                new_i2 = i2+1
            else:
                new_i2 = i1

            if 'blinking' not in self.data:
                self.data['blinking'] = np.zeros(len(self.data['frame']), dtype=np.uint)
                
            self.data['blinking'][i1:i2] = 1
            
            for key in ['diameter', 'cx', 'cy', 'sx', 'sy', 'residual', 'angle']:
                I = np.arange(i1, i2)
                self.data[key][i1:i2] = self.data[key][new_i1]+(I-i1)/(i2-i1)*(self.data[key][new_i2]-self.data[key][new_i1])

            # FOR MORE FANCY INTERPOLATION
            # cond = (self.data['blinking']>0)
            # for key in ['diameter', 'cx', 'cy', 'sx', 'sy', 'residual']:
            #     func = interp1d(self.data['frame'][cond],
            #                     self.data[key][cond],
            #                     kind='linear')
            #     self.data[key] = func(self.data['frame'])

            self.plot_pupil_trace(xrange=self.xaxis.range)
            self.cframe1, self.cframe2 = 0, 0
        else:
            print('cursors at: ', self.cframe1, self.cframe2)
            print('blinking/outlier labelling failed')
    
        
    def debug(self):
        print('No debug function')

    def set_cursor_1(self):
        self.cframe1 = self.cframe
        print('cursor 1 set to: %i' % self.cframe1)
        
    def set_cursor_2(self):
        self.cframe2 = self.cframe
        print('cursor 2 set to: %i' % self.cframe2)

    def set_precise_time(self):
        self.time = float(self.currentTime.text())
        t1, t2 = self.xaxis.range
        frac_value = (self.time-t1)/(t2-t1)
        self.frameSlider.setValue(int(self.slider_nframes*frac_value))
        self.jump_to_frame()
        
    def go_to_frame(self):
        i1, i2 = self.xaxis.range
        self.cframe = max([0, int(i1+(i2-i1)*float(self.frameSlider.value()/200.))])
        self.jump_to_frame()

    def fitToWindow(self):
        self.movieLabel.setScaledContents(self.fitCheckBox.isChecked())

    def updateFrameSlider(self):
        self.timeLabel.setEnabled(True)
        self.frameSlider.setEnabled(True)

    def jump_to_frame(self):

        if self.FILES is not None:
            # full image 
            self.fullimg = np.load(os.path.join(self.imgfolder,
                                                self.FILES[self.cframe]))
            self.pimg.setImage(self.fullimg)

            # zoomed image
            if self.ROI is not None:
                process.init_fit_area(self)
                process.preprocess(self,\
                                gaussian_smoothing=float(self.smoothBox.text()),
                                saturation=self.sl.value())
                
                self.pPupilimg.setImage(self.img)
                self.pPupilimg.setLevels([self.img.min(), self.img.max()])

                self.reflector.setEnabled(False)
                self.reflector.setEnabled(True)
            
        if self.scatter is not None:
            self.p1.removeItem(self.scatter)
        if self.fit is not None:
            self.fit.remove(self)
            
        if self.data is not None:
            
            self.iframe = np.arange(len(self.data['frame']))[self.data['frame']>=self.cframe][0]
            self.scatter.setData(self.data['frame'][self.iframe]*np.ones(1),
                                 self.data['diameter'][self.iframe]*np.ones(1),
                                 size=10, brush=pg.mkBrush(255,255,255))
            self.p1.addItem(self.scatter)
            self.p1.show()
            coords = []
            if 'sx-corrected' in self.data:
                for key in ['cx-corrected', 'cy-corrected',
                            'sx-corrected', 'sy-corrected',
                            'angle-corrected']:
                    coords.append(self.data[key][self.iframe])
            else:
                for key in ['cx', 'cy', 'sx', 'sy', 'angle']:
                    coords.append(self.data[key][self.iframe])


            self.plot_pupil_ellipse(coords)
            # self.fit = roi.pupilROI(moveable=True,
            #                         parent=self,
            #                         color=(0, 200, 0),
            #                         pos = roi.ellipse_props_to_ROI(coords))
            
        self.win.show()
        self.show()

    def plot_pupil_ellipse(self, coords):

        self.pupilContour.setData(*process.ellipse_coords(*coords, transpose=True),
                                  size=3, brush=pg.mkBrush(255,0,0))
        self.pupilCenter.setData([coords[1]], [coords[0]],
                                 size=8, brush=pg.mkBrush(255,0,0))
        

    def extract_ROI(self, data):

        print(data)
        if len(self.rROI)>0:
            data['reflectors'] = [r.extract_props() for r in self.rROI]
        if self.ROI is not None:
            data['ROIellipse'] = self.ROI.extract_props()
        if self.pupil is not None:
            data['ROIpupil'] = self.pupil.extract_props()
        data['ROIsaturation'] = self.saturation

        boundaries = process.extract_boundaries_from_ellipse(\
                                    data['ROIellipse'], self.Lx, self.Ly)
        for key in boundaries:
            data[key]=boundaries[key]
        
        
    def save_ROIs(self):
        """ """
        self.extract_ROI(self.data)
        self.save_pupil_data()
        
    def build_cmd(self):
        return 'python %s -df %s' % (self.process_script, self.datafolder)
        
    def run_as_subprocess(self):
        self.save_pupil_data()
        cmd = self.build_cmd()
        p = subprocess.Popen(cmd, shell=True)
        print('"%s" launched as a subprocess' % cmd)

    def add_to_bash_script(self):
        self.save_pupil_data()
        cmd = self.build_cmd()
        with open(self.script, 'a') as f:
            f.write(cmd+' & \n')
        print('Command: "%s"\n successfully added to the script: "%s"' % (cmd, self.script))

    def save_pupil_data(self):
        """ """
        if self.pupil_shape.currentText()=='Ellipse fit':
            self.data['shape'] = 'ellipse'
        else:
            self.data['shape'] = 'circle'
        self.data['gaussian_smoothing'] = int(self.smoothBox.text())
        self.data['cm_to_pix'] = int(self.scaleBox.text())
        self.data = process.clip_to_finite_values(self.data)
            
        np.save(os.path.join(self.datafolder, 'pupil.npy'), self.data)
        print('Data successfully saved as "%s"' % os.path.join(self.datafolder, 'pupil.npy'))
        
            
    def process_ROIs(self):

        self.data = {}
        self.extract_ROI(self.data)
        
        self.subsampling = int(self.samplingBox.text())
        if self.pupil_shape.currentText()=='Ellipse fit':
            self.Pshape = 'ellipse'
            
        else:
            self.Pshape = 'circle'

        print('processing pupil size over the whole recording [...]')
        print(' with %i frame subsampling' % self.subsampling)

        process.init_fit_area(self)
        temp = process.perform_loop(self,
                                    subsampling=self.subsampling,
                                    shape=self.Pshape,
                                    gaussian_smoothing=int(self.smoothBox.text()),
                                    saturation=self.sl.value(),
                                    with_ProgressBar=True)

        for key in temp:
            self.data[key] = temp[key]
        self.data['times'] = self.times[self.data['frame']]
                
        self.plot_pupil_trace()
            
        self.win.show()
        self.show()

    def plot_pupil_trace(self, xrange=None):
        self.p1.clear()
        if self.data is not None:
            # self.data = process.remove_outliers(self.data)
            cond = np.isfinite(self.data['diameter'])
            self.p1.plot(self.data['frame'][cond],
                         self.data['diameter'][cond], pen=(0,255,0))
            if xrange is None:
                xrange = (0, self.data['frame'][cond][-1])
            self.p1.setRange(xRange=xrange,
                             yRange=(self.data['diameter'][cond].min()-.1,
                                     self.data['diameter'][cond].max()+.1),
                             padding=0.0)
            self.p1.show()

    def fit_pupil_size(self, value=0, coords_only=False):
        
        if not coords_only and (self.pupil is not None):
            self.pupil.remove(self)

        if self.pupil_shape.currentText()=='Ellipse fit':
            coords, _, _ = process.perform_fit(self,
                                               saturation=self.sl.value(),
                                               shape='ellipse')
        else:
            coords, _, _ = process.perform_fit(self,
                                               saturation=self.sl.value(),
                                               shape='circle')
            coords = list(coords)+[coords[-1], 0] # form circle to ellipse

        if not coords_only:
            self.plot_pupil_ellipse(coords)

        # TROUBLESHOOTING
        # from datavyz import ge
        # fig, ax = ge.figure(figsize=(1.4,2), left=0, bottom=0, right=0, top=0)
        # ax.plot(*process.ellipse_coords(*coords, transpose=True), 'r')
        # ax.plot([coords[1]], [coords[0]], 'ro')
        # ax.imshow(self.img)
        # ge.show()
        
        return coords

    def interpolate_data(self):
        for key in ['diameter', 'cx', 'cy', 'sx', 'sy', 'residual', 'angle']:
            func = interp1d(self.data['frame'], self.data[key],
                            kind='linear')
            self.data[key] = func(np.arange(self.nframes))
        self.data['frame'] = np.arange(self.nframes)
        self.data['times'] = self.times[self.data['frame']]

        self.plot_pupil_trace()
        print('[ok] interpolation successfull !')
        
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



    
