import sys, os, shutil, glob, time, subprocess, pathlib
import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
from scipy.interpolate import interp1d

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from misc.folders import FOLDERS, python_path
from misc.style import set_dark_style, set_app_icon
from misc.guiparts import NewWindow, Slider
from assembling.tools import load_FaceCamera_data
from facemotion import roi, process

class MainWindow(NewWindow):
    
    def __init__(self, app,
                 args=None,
                 parent=None,
                 spatial_subsampling=1,
                 time_subsampling=1):
        """
        FaceMotion GUI
        """
        self.app = app
        
        super(MainWindow, self).__init__(i=2,
                                         title='Face-motion/Whisking tracking')

        ##############################
        ##### keyboard shortcuts #####
        ##############################

        self.refc1 = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+1'), self)
        self.refc1.activated.connect(self.set_cursor_1)
        self.refc2 = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+2'), self)
        self.refc2.activated.connect(self.set_cursor_2)
        self.refc3 = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+3'), self)
        self.refc3.activated.connect(self.interpolate_data)

        #############################
        ##### module quantities #####
        #############################
        
        # initializing to defaults/None:
        self.saturation = 255
        self.ROI, self.pupil = None, None
        self.times, self.imgfolder, self.nframes, self.FILES = None, None, None, None
        self.data, self.scatter, self.fit= None, None, None # the pupil size contour
        self.spatial_subsampling = spatial_subsampling
        self.time_subsampling = time_subsampling
        self.process_script = os.path.join(str(pathlib.Path(__file__).resolve().parents[0]),
                                           'process.py') # for batch processing -> execute this script
        
        ########################
        ##### building GUI #####
        ########################
        
        self.minView = False
        self.showwindow()

        # central widget
        self.cwidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.cwidget)
        # layout
        self.l0 = QtWidgets.QGridLayout()
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
        self.pFace.setMenuEnabled(False)
        self.pFaceimg = pg.ImageItem(None)
        self.pFace.addItem(self.pFaceimg)

        # saturation sliders
        self.sl = Slider(0, self)
        self.l0.addWidget(self.sl,1,6,1,7)
        qlabel= QtWidgets.QLabel('saturation')
        qlabel.setStyleSheet('color: white;')
        self.l0.addWidget(qlabel, 0,8,1,3)

        # reset
        self.reset_btn = QtWidgets.QPushButton('reset')
        self.l0.addWidget(self.reset_btn, 1, 11+6, 1, 1)
        self.reset_btn.clicked.connect(self.reset)
        self.reset_btn.setEnabled(True)

        # debug
        self.debugBtn = QtWidgets.QPushButton('- Debug -')
        self.l0.addWidget(self.debugBtn, 2, 11+6, 1, 1)
        self.debugBtn.setEnabled(True)
        self.debugBtn.clicked.connect(self.debug)

        
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
        self.movieLabel = QtWidgets.QLabel("No datafile chosen")
        self.movieLabel.setStyleSheet("color: white;")
        self.l0.addWidget(self.movieLabel,0,1,1,5)

        # create frame slider
        self.timeLabel = QtWidgets.QLabel("time : ")
        self.timeLabel.setStyleSheet("color: white;")
        # self.timeLabel.setFixedWidth(300)
        self.currentTime = QtWidgets.QLineEdit()
        self.currentTime.setText('0 s')
        self.currentTime.setFixedWidth(70)
        # self.currentTime.returnPressed.connect(self.set_precise_time)
        
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
        
        self.processBtn = QtWidgets.QPushButton('process data')
        self.processBtn.clicked.connect(self.process)

        self.interpolate = QtGui.QPushButton('interpolate')
        self.interpolate.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.interpolate.clicked.connect(self.interpolate_data)

        self.motionCheckBox = QtWidgets.QCheckBox("display motion frames")
        self.motionCheckBox.setStyleSheet("color: gray;")
        
        self.runAsSubprocess = QtWidgets.QPushButton('run as subprocess')
        self.runAsSubprocess.clicked.connect(self.run_as_subprocess)

        self.load = QtWidgets.QPushButton('  load data [Ctrl+O]  \u2b07')
        self.load.clicked.connect(self.open_file)

        sampLabel1 = QtWidgets.QLabel("spatial subsampling")
        sampLabel1.setStyleSheet("color: gray;")
        sampLabel1.setFixedWidth(220)
        self.SsamplingBox = QtWidgets.QLineEdit()
        self.SsamplingBox.setText(str(self.spatial_subsampling))
        self.SsamplingBox.setFixedWidth(30)

        sampLabel2 = QtWidgets.QLabel("temporal subsampling")
        sampLabel2.setStyleSheet("color: gray;")
        sampLabel2.setFixedWidth(220)
        self.TsamplingBox = QtWidgets.QLineEdit()
        self.TsamplingBox.setText(str(self.time_subsampling))
        self.TsamplingBox.setFixedWidth(30)
        
        self.addROI = QtWidgets.QPushButton("set ROI")
        self.addROI.clicked.connect(self.add_ROI)

        self.saveData = QtWidgets.QPushButton('save data')
        self.saveData.clicked.connect(self.save_data)

        sampLabel3 = QtWidgets.QLabel("grooming threshold")
        sampLabel3.setStyleSheet("color: gray;")
        sampLabel3.setFixedWidth(220)
        self.groomingBox = QtWidgets.QLineEdit()
        self.groomingBox.setText('-1')
        self.groomingBox.setFixedWidth(40)
        self.groomingBox.returnPressed.connect(self.update_grooming_threshold)

        self.processGrooming = QtWidgets.QPushButton("process grooming")
        self.processGrooming.clicked.connect(self.process_grooming)
        
        for x in [self.processBtn, self.motionCheckBox, self.runAsSubprocess,
                  self.load, self.addROI, self.saveData, self.TsamplingBox,
                  self.SsamplingBox, self.groomingBox, self.processGrooming]:
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

        for wdg, loc in zip([self.load,self.folderB,
                             sampLabel1, sampLabel2,
                             self.addROI, self.saveData,
                             sampLabel3, self.processGrooming,
                             self.motionCheckBox, self.processBtn, self.runAsSubprocess,
                             self.timeLabel],
                            [1,2,
                             8,9,
                             12,14,
                             19,20,
                             23,24,26,
                             istretch+13]):
            
            self.l0.addWidget(wdg,loc,0,1,3)

        self.l0.addWidget(self.SsamplingBox, 8, 2, 1, 3)
        self.l0.addWidget(self.TsamplingBox, 9, 2, 1, 3)
        self.l0.addWidget(self.groomingBox, 19, 2, 1, 3)
        
        self.l0.addWidget(QtWidgets.QLabel(''),istretch,0,1,3)
        self.l0.setRowStretch(istretch,1)
        self.l0.addWidget(self.currentTime, istretch+13,1,1,3)
        self.l0.addWidget(self.frameSlider, istretch+17,3,1,15)

        # self.l0.addWidget(QtWidgets.QLabel(''),17,2,1,1)
        # self.l0.setRowStretch(16,2)
        # # self.l0.addWidget(ll, istretch+3+k+1,0,1,4)
        self.timeLabel.setEnabled(True)
        self.frameSlider.setEnabled(True)
        
        self.nframes = 0
        self.cframe = 0
        self.grooming_threshold = -1

        self.show()


    def add_ROI(self):
        self.ROI = roi.faceROI(moveable=True, parent=self)

    def open_file(self):

        self.cframe = 0
        
        folder = QtWidgets.QFileDialog.getExistingDirectory(self,\
                                    "Choose datafolder",
                                    FOLDERS[self.folderB.currentText()])
        # folder = '/home/yann/DATA/14-10-48/'

        if folder!='':
            
            self.datafolder = folder
            
            if os.path.isdir(os.path.join(folder, 'FaceCamera-imgs')):
                
                self.reset()
                self.imgfolder = os.path.join(self.datafolder, 'FaceCamera-imgs')
                process.load_folder(self) # in init: self.times, _, self.nframes, ...
                
            else:
                self.times, self.imgfolder, self.nframes, self.FILES = None, None, None, None
                print(' /!\ no raw FaceCamera data found ...')

            if os.path.isfile(os.path.join(self.datafolder, 'facemotion.npy')):
                
                self.data = np.load(os.path.join(self.datafolder, 'facemotion.npy'),
                                    allow_pickle=True).item()
                
                if (self.nframes is None) and ('frame' in self.data):
                    self.nframes = self.data['frame'].max()

                if 'ROI' in self.data:
                    self.ROI = roi.faceROI(moveable=True, parent=self,
                                           pos=self.data['ROI'])
                    

                if 'ROIsaturation' in self.data:
                    self.sl.setValue(int(self.data['ROIsaturation']))

                if 'grooming_threshold' in self.data:
                    self.grooming_threshold = self.data['grooming_threshold']
                else:
                    self.grooming_threshold = int(self.data['motion'].max())+1
                    
                self.groomingBox.setText(str(self.grooming_threshold))
                    
                if 'frame' in self.data:
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

        self.data['grooming_threshold'] = self.grooming_threshold
        
        np.save(os.path.join(self.datafolder, 'facemotion.npy'), self.data)
        
        print('data saved as: "%s"' % os.path.join(self.datafolder, 'facemotion.npy'))
        
        
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
            
        if (self.data is not None) and ('frame' in self.data):

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
        self.data = {'frame':frames, 't':self.times[frames],
                     'motion':motion, 'grooming':0*frames}
        if self.grooming_threshold==-1:
            self.grooming_threshold = int(self.data['motion'].max())+1
            
        self.plot_motion_trace()


    def plot_motion_trace(self, xrange=None):
        self.p1.clear()
        self.p1.plot(self.data['frame'],
                     self.data['motion'], pen=(0,0,255))

        if xrange is None:
            xrange = (0, self.nframes)

        self.line = pg.InfiniteLine(pos=self.grooming_threshold, angle=0, movable=True)
        self.p1.addItem(self.line)
        
        self.p1.setRange(xRange=xrange,
                         yRange=(self.data['motion'].min()-.1,
                                 np.max([self.grooming_threshold, self.data['motion'].max()])),
                         padding=0.0)
        self.p1.show()
    

    def build_cmd(self):
        return '%s %s -df %s' % (python_path, self.process_script, self.datafolder)
    
    def run_as_subprocess(self):
        self.save_data()
        cmd = self.build_cmd()
        p = subprocess.Popen(cmd, shell=True)
        print('"%s" launched as a subprocess' % cmd)

    def add_to_bash_script(self):
        self.save_data()
        cmd = self.build_cmd()
        with open(self.script, 'a') as f:
            f.write(cmd+' & \n')
        print('Command: "%s"\n successfully added to the script: "%s"' % (cmd, self.script))


    def set_cursor_1(self):
        self.cframe1 = self.cframe
        print('cursor 1 set to: %i' % self.cframe1)

        
    def set_cursor_2(self):
        self.cframe2 = self.cframe
        print('cursor 2 set to: %i' % self.cframe2)

    def process_grooming(self):
        
        if not 'motion_before_grooming' in self.data:
            self.data['motion_before_grooming'] = self.data['motion'].copy()

        self.grooming_threshold = int(self.line.value())
        up_cond = self.data['motion_before_grooming']>self.grooming_threshold
        self.data['motion'][up_cond] = self.grooming_threshold
        self.data['motion'][~up_cond] = self.data['motion_before_grooming'][~up_cond]

        if 'grooming' not in self.data:
            self.data['grooming'] = 0*self.data['motion']
            
        self.data['grooming'][up_cond] = 1
        self.data['grooming'][~up_cond] = 0

        self.plot_motion_trace()

    def update_line(self):
        self.groomingBox.setText('%i' % self.line.value())
        
    def update_grooming_threshold(self):
        self.grooming_threshold = int(self.groomingBox.text())
        self.plot_motion_trace()
        
    def interpolate_data(self, with_blinking_flag=False):
        
        if self.data is not None and (self.cframe1!=0) and (self.cframe2!=0):
            
            i1 = np.arange(len(self.data['frame']))[self.data['frame']>=self.cframe1][0]
            i2 = np.arange(len(self.data['frame']))[self.data['frame']>=self.cframe2][0]
            self.data['motion'][i1:i2] = 0
            if i1>0:
                new_i1 = i1-1
            else:
                new_i1 = i2
            if i2<len(self.data['frame'])-1:
                new_i2 = i2+1
            else:
                new_i2 = i1

            for key in ['motion']:
                I = np.arange(i1, i2)
                self.data[key][i1:i2] = self.data[key][new_i1]+(I-i1)/(i2-i1)*(self.data[key][
                    new_i2]-self.data[key][new_i1])

            self.plot_motion_trace(xrange=self.xaxis.range)
            self.cframe1, self.cframe2 = 0, 0
        else:
            print('cursors at: ', self.cframe1, self.cframe2)
        
        
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



    
