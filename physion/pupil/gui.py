import sys, os, shutil, glob, time, subprocess, pathlib
import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
from scipy.interpolate import interp1d

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from pupil import process, roi
from misc.folders import FOLDERS, python_path
from misc.guiparts import NewWindow, Slider
from assembling.tools import load_FaceCamera_data

class MainWindow(NewWindow):
    
    def __init__(self, app,
                 args=None,
                 parent=None,
                 gaussian_smoothing=1,
                 subsampling=1000):
        """
        Pupil Tracking GUI
        """
        self.app = app
        
        super(MainWindow, self).__init__(i=1,
                                         title='Pupil tracking')


        ##############################
        ##### keyboard shortcuts #####
        ##############################

        self.refc1 = QtWidgets.QShortcut(QtGui.QKeySequence('1'), self)
        self.refc1.activated.connect(self.set_cursor_1)
        self.refc2 = QtWidgets.QShortcut(QtGui.QKeySequence('2'), self)
        self.refc2.activated.connect(self.set_cursor_2)
        self.refc3 = QtWidgets.QShortcut(QtGui.QKeySequence('3'), self)
        self.refc3.activated.connect(self.process_outliers)
        self.refc4 = QtWidgets.QShortcut(QtGui.QKeySequence('4'), self)
        self.refc4.activated.connect(self.interpolate)
        self.refc5 = QtWidgets.QShortcut(QtGui.QKeySequence('E'), self)
        self.refc5.activated.connect(self.find_outliers)
        
        #############################
        ##### module quantities #####
        #############################

        self.gaussian_smoothing = gaussian_smoothing
        self.subsampling = subsampling
        self.process_script = os.path.join(str(pathlib.Path(__file__).resolve().parents[0]),
                                           'process.py')
        self.ROI, self.pupil, self.times = None, None, None
        self.data = None
        self.bROI, self.reflectors = [], []
        self.scatter, self.fit= None, None # the pupil size contour
        
        ########################
        ##### building GUI #####
        ########################
        
        self.cwidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.cwidget)
        self.l0 = QtWidgets.QGridLayout()
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

        # saturation sliders
        self.sl = Slider(0, self)
        self.sl.setValue(100)
        self.l0.addWidget(self.sl,1,6,1,7)
        qlabel= QtWidgets.QLabel('saturation')
        qlabel.setStyleSheet('color: white;')
        self.l0.addWidget(qlabel, 0,8,1,3)

        # adding blanks (eye borders, ...)
        self.blankBtn = QtWidgets.QPushButton('add blanks')
        self.l0.addWidget(self.blankBtn, 1, 8+6, 1, 1)
        self.blankBtn.setEnabled(True)
        self.blankBtn.clicked.connect(self.add_blankROI)
        
        # adding reflections ("corneal reflections, ...")
        self.reflectorBtn = QtWidgets.QPushButton('add reflect.')
        self.l0.addWidget(self.reflectorBtn, 2, 8+6, 1, 1)
        self.reflectorBtn.setEnabled(True)
        self.reflectorBtn.clicked.connect(self.add_reflectROI)

        self.keepCheckBox = QtWidgets.QCheckBox("keep ROIs")
        self.keepCheckBox.setStyleSheet("color: gray;")
        self.keepCheckBox.setChecked(True)
        self.l0.addWidget(self.keepCheckBox, 2, 8+7, 1, 1)
        
        # fit pupil
        self.fit_pupil = QtWidgets.QPushButton('fit Pupil [Ctrl+F]')
        self.l0.addWidget(self.fit_pupil, 1, 9+6, 1, 1)
        # self.fit_pupil.setEnabled(True)
        self.fit_pupil.clicked.connect(self.fit_pupil_size)
        # choose pupil shape
        self.pupil_shape = QtWidgets.QComboBox(self)
        self.pupil_shape.addItem("Ellipse fit")
        self.pupil_shape.addItem("Circle fit")
        self.l0.addWidget(self.pupil_shape, 1, 10+6, 1, 1)
        # reset
        self.reset_btn = QtWidgets.QPushButton('reset')
        self.l0.addWidget(self.reset_btn, 1, 11+6, 1, 1)
        self.reset_btn.clicked.connect(self.reset)
        # self.reset_btn.setEnabled(True)
        # draw pupil
        self.refresh_pupil = QtWidgets.QPushButton('Refresh [Ctrl+R]')
        self.l0.addWidget(self.refresh_pupil, 2, 11+6, 1, 1)
        self.refresh_pupil.setEnabled(True)
        self.refresh_pupil.clicked.connect(self.jump_to_frame)

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
        self.movieLabel = QtWidgets.QLabel("No datafile chosen")
        self.movieLabel.setStyleSheet("color: white;")
        self.l0.addWidget(self.movieLabel,0,1,1,5)


        # create frame slider
        self.timeLabel = QtWidgets.QLabel("Current time (seconds):")
        self.timeLabel.setStyleSheet("color: white;")
        self.currentTime = QtWidgets.QLineEdit()
        self.currentTime.setText('0')
        self.currentTime.setValidator(QtGui.QDoubleValidator(0, 100000, 2))
        self.currentTime.setFixedWidth(50)
        self.currentTime.returnPressed.connect(self.set_precise_time)
        
        self.frameSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
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

        self.process = QtWidgets.QPushButton('process data [Ctrl+P]')
        self.process.clicked.connect(self.process_ROIs)

        self.cursor1 = QtWidgets.QPushButton('Set Cursor 1 [Ctrl+1]')
        self.cursor1.clicked.connect(self.set_cursor_1)

        self.cursor2 = QtWidgets.QPushButton('Set Cursor 2 [Ctrl+2]')
        self.cursor2.clicked.connect(self.set_cursor_2)
        
        self.runAsSubprocess = QtWidgets.QPushButton('run as subprocess')
        self.runAsSubprocess.clicked.connect(self.run_as_subprocess)

        self.load = QtWidgets.QPushButton('  load data [Ctrl+O]  \u2b07')
        self.load.clicked.connect(self.load_data)

        self.loadLastGUIsettings = QtWidgets.QPushButton("last GUI settings")
        self.loadLastGUIsettings.clicked.connect(self.load_last_gui_settings)
        
        sampLabel = QtWidgets.QLabel("Subsampling (frame)")
        sampLabel.setStyleSheet("color: gray;")
        self.samplingBox = QtWidgets.QLineEdit()
        self.samplingBox.setText(str(self.subsampling))
        self.samplingBox.setFixedWidth(50)

        smoothLabel = QtWidgets.QLabel("Smoothing (px)")
        smoothLabel.setStyleSheet("color: gray;")
        self.smoothBox = QtWidgets.QLineEdit()
        self.smoothBox.setText(str(self.gaussian_smoothing))
        self.smoothBox.setFixedWidth(30)

        self.addROI = QtWidgets.QPushButton("add Pupil-ROI")
        
        self.addROI.clicked.connect(self.add_ROI)

        self.saverois = QtWidgets.QPushButton('save data [Ctrl+S]')
        self.saverois.clicked.connect(self.save)

        stdLabel = QtWidgets.QLabel("std excl. factor: ")
        stdLabel.setStyleSheet("color: gray;")
        self.stdBox = QtWidgets.QLineEdit()
        self.stdBox.setText('3.0')
        self.stdBox.setFixedWidth(50)

        wdthLabel = QtWidgets.QLabel("excl. width (s): ")
        wdthLabel.setStyleSheet("color: gray;")
        self.wdthBox = QtWidgets.QLineEdit()
        self.wdthBox.setText('0.1')
        self.wdthBox.setFixedWidth(50)
        
        self.excludeOutliers = QtWidgets.QPushButton('exclude outlier [Ctrl+E]')
        self.excludeOutliers.clicked.connect(self.find_outliers)

        cursorLabel = QtWidgets.QLabel("set cursor 1 [Ctrl+1], cursor 2 [Ctrl+2]")
        cursorLabel.setStyleSheet("color: gray;")
        
        self.interpBtn = QtWidgets.QPushButton('interpolate only [Ctrl+4]')
        self.interpBtn.clicked.connect(self.interpolate)

        self.processOutliers = QtWidgets.QPushButton('Set blinking interval [Ctrl+3]')
        self.processOutliers.clicked.connect(self.process_outliers)
        
        self.printSize = QtWidgets.QPushButton('print ROI size')
        self.printSize.clicked.connect(self.print_size)

        for x in [self.process, self.cursor1, self.cursor2, self.runAsSubprocess, self.load,
                  self.saverois, self.addROI, self.interpBtn, self.processOutliers,
                  self.stdBox, self.wdthBox, self.excludeOutliers, self.printSize, cursorLabel,
                  self.loadLastGUIsettings,
                  sampLabel, smoothLabel, stdLabel, wdthLabel, self.smoothBox, self.samplingBox]:
            x.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        
        
        iconSize = QtCore.QSize(30, 30)
        self.playButton = QtWidgets.QToolButton()
        self.playButton.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        self.playButton.setIconSize(iconSize)
        self.playButton.setToolTip("Play")
        self.playButton.setCheckable(True)
        self.pauseButton = QtWidgets.QToolButton()
        self.pauseButton.setCheckable(True)
        self.pauseButton.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPause))
        self.pauseButton.setIconSize(iconSize)
        self.pauseButton.setToolTip("Pause")

        btns = QtWidgets.QButtonGroup(self)
        btns.addButton(self.playButton,0)
        btns.addButton(self.pauseButton,1)

        self.l0.addWidget(self.folderB,1,0,1,3)
        self.l0.addWidget(self.load,2,0,1,3)
        self.l0.addWidget(self.loadLastGUIsettings, 7, 0, 1, 3)
        self.l0.addWidget(sampLabel, 8, 0, 1, 3)
        self.l0.addWidget(self.samplingBox, 8, 2, 1, 3)
        self.l0.addWidget(smoothLabel, 9, 0, 1, 3)
        self.l0.addWidget(self.smoothBox, 9, 2, 1, 3)
        self.l0.addWidget(self.addROI,14,0,1,3)
        self.l0.addWidget(self.process, 16, 0, 1, 3)
        self.l0.addWidget(self.runAsSubprocess, 17, 0, 1, 3)
        self.l0.addWidget(self.saverois, 19, 0, 1, 3)

        self.l0.addWidget(stdLabel, 21, 0, 1, 3)
        self.l0.addWidget(self.stdBox, 21, 2, 1, 3)
        self.l0.addWidget(wdthLabel, 22, 0, 1, 3)
        self.l0.addWidget(self.wdthBox, 22, 2, 1, 3)
        self.l0.addWidget(self.excludeOutliers, 23, 0, 1, 3)
        self.l0.addWidget(cursorLabel, 25, 0, 1, 3)
        self.l0.addWidget(self.processOutliers, 26, 0, 1, 3)
        self.l0.addWidget(self.interpBtn, 27, 0, 1, 3)
        self.l0.addWidget(self.printSize, 29, 0, 1, 3)

        self.l0.addWidget(QtWidgets.QLabel(''),istretch,0,1,3)
        self.l0.setRowStretch(istretch,1)
        self.l0.addWidget(self.timeLabel, istretch+13,0,1,3)
        self.l0.addWidget(self.currentTime, istretch+14,0,1,3)
        self.l0.addWidget(self.frameSlider, istretch+15,3,1,15)

        self.l0.addWidget(QtWidgets.QLabel(''),17,2,1,1)
        self.l0.setRowStretch(16,2)
        # self.l0.addWidget(ll, istretch+3+k+1,0,1,4)
        self.updateFrameSlider()
        
        self.nframes = 0
        self.cframe, self.cframe1, self.cframe2, = 0, 0, 0

        self.updateTimer = QtCore.QTimer()
        
        self.win.show()
        self.show()

    def open_file(self):
        self.load_data()
        
    def load_data(self):

        self.cframe = 0
        
        folder = QtWidgets.QFileDialog.getExistingDirectory(self,\
                                    "Choose datafolder",
                                    FOLDERS[self.folderB.currentText()])
        # folder = '/home/yann/UNPROCESSED/2021_09_10/13-52-49/'

        if folder!='':
            
            self.datafolder = folder
            
            if os.path.isdir(os.path.join(folder, 'FaceCamera-imgs')):

                if not self.keepCheckBox.isChecked():
                    self.reset()
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
                self.p1.clear()

            if self.times is not None:
                self.jump_to_frame()
                self.timeLabel.setEnabled(True)
                self.frameSlider.setEnabled(True)
                self.updateFrameSlider()
                self.currentTime.setValidator(QtGui.QDoubleValidator(0, self.nframes, 2))
                self.movieLabel.setText(folder)


    def save_gui_settings(self):

        settings = {'gaussian_smoothing':int(self.smoothBox.text())}
        if len(self.bROI)>0:
            settings['blanks'] = [r.extract_props() for r in self.bROI]
        if len(self.reflectors)>0:
            settings['reflectors'] = [r.extract_props() for r in self.reflectors]
        if self.ROI is not None:
            settings['ROIellipse'] = self.ROI.extract_props()
        if self.pupil is not None:
            settings['ROIpupil'] = self.pupil.extract_props()
        settings['ROIsaturation'] = self.sl.value()
        
        np.save(os.path.join(pathlib.Path(__file__).resolve().parent, '_gui_settings.npy'), settings)

    def load_last_gui_settings(self):

        try:
            settings = np.load(os.path.join(pathlib.Path(__file__).resolve().parent, '_gui_settings.npy'),
                               allow_pickle=True).item()

            self.smoothBox.setText('%i' % settings['gaussian_smoothing'])
            self.sl.setValue(int(settings['ROIsaturation']))
            self.ROI = roi.sROI(parent=self,
                                pos=roi.ellipse_props_to_ROI(settings['ROIellipse']))

            self.bROI, self.reflectors = [], [] # blanks & reflectors
            for b in settings['blanks']:
                self.bROI.append(roi.reflectROI(len(self.bROI), moveable=True, parent=self,
                                                pos=roi.ellipse_props_to_ROI(b)))
            if 'reflectors' in settings:
                for r in settings['reflectors']:
                    self.reflectors.append(roi.reflectROI(len(self.bROI), moveable=True, parent=self,
                                                          pos=roi.ellipse_props_to_ROI(r), color='green'))
                
            self.jump_to_frame()
        except FileNotFoundError:
            print('\n /!\ last GUI settings not found ... \n')

            
    def reset(self):
        for r in self.bROI:
            r.remove(self)
        for r in self.reflectors:
            r.remove(self)
        if self.ROI is not None:
            self.ROI.remove(self)
        if self.pupil is not None:
            self.pupil.remove(self)
        if self.fit is not None:
            self.fit.remove(self)
        self.ROI, self.bROI = None, []
        self.fit = None
        self.reflectors=[]
        self.cframe1, self.cframe2 = 0, -1
        
    def add_blankROI(self):
        self.bROI.append(roi.reflectROI(len(self.bROI), moveable=True, parent=self))

    def add_reflectROI(self):
        self.reflectors.append(roi.reflectROI(len(self.reflectors), moveable=True, parent=self, color='green'))
        
    def draw_pupil(self):
        self.pupil = roi.pupilROI(moveable=True, parent=self)

    def print_size(self):
        print('x, y, sx, sy, angle = ', self.ROI.extract_props())

    def add_ROI(self):

        if self.ROI is not None:
            self.ROI.remove(self)
        for r in self.bROI:
            r.remove(self)
        self.ROI = roi.sROI(parent=self)
        self.bROI = []
        self.reflectors = []


    def interpolate(self, with_blinking_flag=False):
        
        if self.data is not None and (self.cframe1!=0) and (self.cframe2!=0):
            
            i1 = np.arange(len(self.data['frame']))[self.data['frame']>=self.cframe1][0]
            i2 = np.arange(len(self.data['frame']))[self.data['frame']>=self.cframe2][0]
            if i1>0:
                new_i1 = i1-1
            else:
                new_i1 = i2
            if i2<len(self.data['frame'])-1:
                new_i2 = i2+1
            else:
                new_i2 = i1

            if with_blinking_flag:
                
                if 'blinking' not in self.data:
                    self.data['blinking'] = np.zeros(len(self.data['frame']), dtype=np.uint)

                self.data['blinking'][i1:i2] = 1
            
            for key in ['cx', 'cy', 'sx', 'sy', 'residual', 'angle']:
                I = np.arange(i1, i2)
                self.data[key][i1:i2] = self.data[key][new_i1]+(I-i1)/(i2-i1)*(self.data[key][new_i2]-self.data[key][new_i1])

            self.plot_pupil_trace(xrange=self.xaxis.range)
            self.cframe1, self.cframe2 = 0, 0

        elif self.cframe1==0:
            i2 = np.arange(len(self.data['frame']))[self.data['frame']>=self.cframe2][0]
            for key in ['cx', 'cy', 'sx', 'sy', 'residual', 'angle']:
                self.data[key][self.cframe1:i2] = self.data[key][i2] # set to i2 level !!
            self.plot_pupil_trace(xrange=self.xaxis.range)
            self.cframe1, self.cframe2 = 0, 0
        elif self.cframe2==(len(self.data['frame'])-1):
            i1 = np.arange(len(self.data['frame']))[self.data['frame']>=self.cframe1][0]
            for key in ['cx', 'cy', 'sx', 'sy', 'residual', 'angle']:
                self.data[key][i1:self.cframe2] = self.data[key][i1] # set to i2 level !!
            self.plot_pupil_trace(xrange=self.xaxis.range)
            self.cframe1, self.cframe2 = 0, 0
        else:
            print('cursors at: ', self.cframe1, self.cframe2)
            print('blinking/outlier labelling failed')

    def process_outliers(self):
        self.interpolate(with_blinking_flag=True)

    def find_outliers(self):
        if not hasattr(self, 'data_before_outliers') or (self.data_before_outliers==None):

            self.data['std_exclusion_factor'] = float(self.stdBox.text())
            self.data['exclusion_width'] = float(self.wdthBox.text())
            self.data_before_outliers = {}
            for key in self.data:
                self.data_before_outliers[key] = self.data[key]
            process.remove_outliers(self.data,
                                    std_criteria=self.data['std_exclusion_factor'],
                                    width_criteria=self.data['exclusion_width'])
        else:
            # we revert to before
            for key in self.data_before_outliers:
                self.data[key] = self.data_before_outliers[key]
            self.data['blinking'] = 0*self.data['frame']
            self.data_before_outliers = None
        self.plot_pupil_trace()
        
        
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

    def updateFrameSlider(self):
        self.timeLabel.setEnabled(True)
        self.frameSlider.setEnabled(True)

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
                process.init_fit_area(self)
                process.preprocess(self,\
                                gaussian_smoothing=float(self.smoothBox.text()),
                                saturation=self.sl.value())
                
                self.pPupilimg.setImage(self.img)
                self.pPupilimg.setLevels([self.img.min(), self.img.max()])

        if self.scatter is not None:
            self.p1.removeItem(self.scatter)
        if self.fit is not None:
            self.fit.remove(self)
            
        if self.data is not None:
            
            self.iframe = np.arange(len(self.data['frame']))[self.data['frame']>=self.cframe][0]
            self.scatter.setData(self.data['frame'][self.iframe]*np.ones(1),
                                 self.data['sx'][self.iframe]*np.ones(1),
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

        if len(self.bROI)>0:
            data['blanks'] = [r.extract_props() for r in self.bROI]
        if len(self.reflectors)>0:
            data['reflectors'] = [r.extract_props() for r in self.reflectors]
        if self.ROI is not None:
            data['ROIellipse'] = self.ROI.extract_props()
        if self.pupil is not None:
            data['ROIpupil'] = self.pupil.extract_props()
        data['ROIsaturation'] = self.sl.value()

        boundaries = process.extract_boundaries_from_ellipse(\
                                    data['ROIellipse'], self.Lx, self.Ly)
        for key in boundaries:
            data[key]=boundaries[key]
        
        
    def save(self):
        """ """
        self.extract_ROI(self.data)
        self.save_pupil_data()
        
    def build_cmd(self):
        return '%s %s -df %s' % (python_path, self.process_script, self.datafolder)
        
    def run_as_subprocess(self):
        self.save_pupil_data()
        cmd = self.build_cmd()
        p = subprocess.Popen(cmd+' --verbose', shell=True)
        print('"%s" launched as a subprocess' % cmd)

    def add_to_bash_script(self):
        self.save_pupil_data()
        cmd = self.build_cmd()
        with open(self.script, 'a') as f:
            f.write(cmd+' & \n')
        print('Command: "%s"\n successfully added to the script: "%s"' % (cmd, self.script))

    def save_pupil_data(self):
        """ """
        if self.data is not None:
            self.data['gaussian_smoothing'] = int(self.smoothBox.text())
            # self.data = process.clip_to_finite_values(self.data, ['cx', 'cy', 'sx', 'sy', 'residual', 'angle'])
            np.save(os.path.join(self.datafolder, 'pupil.npy'), self.data)
            print('Data successfully saved as "%s"' % os.path.join(self.datafolder, 'pupil.npy'))
            self.save_gui_settings()
        else:
            print('Need to pre-process data ! ')
            
        
    def process(self):
        self.process_ROIs()
        self.save_gui_settings()
        
    def process_ROIs(self):

        if (self.data is None) or ('frame' in self.data):
            self.data = {}
            self.extract_ROI(self.data)

            
        self.subsampling = int(self.samplingBox.text())

        print('processing pupil size over the whole recording [...]')
        print(' with %i frame subsampling' % self.subsampling)

        process.init_fit_area(self)
        temp = process.perform_loop(self,
                                    subsampling=self.subsampling,
                                    gaussian_smoothing=int(self.smoothBox.text()),
                                    saturation=self.sl.value(),
                                    reflectors=[r.extract_props() for r in self.reflectors],
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
            cond = np.isfinite(self.data['sx'])
            self.p1.plot(self.data['frame'][cond],
                         self.data['sx'][cond], pen=(0,255,0))
            if xrange is None:
                xrange = (0, self.data['frame'][cond][-1])
            self.p1.setRange(xRange=xrange,
                             yRange=(self.data['sx'][cond].min()-.1,
                                     self.data['sx'][cond].max()+.1),
                             padding=0.0)
            if ('blinking' in self.data) and (np.sum(self.data['blinking'])>0):
                cond = self.data['blinking']>0
                self.p1.plot(self.data['frame'][cond],
                             0*self.data['frame'][cond]+self.data['sx'][cond].min(),
                             symbolPen=pg.mkPen(color=(0, 0, 255, 255), width=0),                                      
                             symbolBrush=pg.mkBrush(0, 0, 255, 255), symbolSize=7,
                             pen=None, symbol='o')
            self.p1.show()

            

    def fit(self):
        self.fit_pupil_size()
        
    def fit_pupil_size(self, value=0, coords_only=False):
        
        if not coords_only and (self.pupil is not None):
            self.pupil.remove(self)

        coords, _, _ = process.perform_fit(self,
                                           saturation=self.sl.value(),
                                           reflectors=[r.extract_props() for r in self.reflectors])

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
        for key in ['cx', 'cy', 'sx', 'sy', 'residual', 'angle']:
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



    
