import sys, os, shutil, glob, time
import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
from pyqtgraph import GraphicsScene
from scipy.stats import zscore, skew
from matplotlib import cm
import pathlib
from analyz.IO.npz import load_dict
from analyz.workflow.shell import printProgressBar

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from pupil import guiparts, process, roi
from misc.style import set_dark_style, set_app_icon
from assembling.saving import from_folder_to_datetime, check_datafolder

class MainWindow(QtWidgets.QMainWindow):
    
    def __init__(self, parent=None, savedir=None,
                 sampling_rate=0.,
                 gaussian_smoothing=2,
                 compressed_version=False,
                 slider_nframes=200):
        """
        sampling in Hz
        """
        super(MainWindow, self).__init__()
        self.sampling_rate = sampling_rate

        # adding a "quit" keyboard shortcut
        self.quitSc = QtWidgets.QShortcut(QtGui.QKeySequence('Q'), self) # or 'Ctrl+Q'
        self.quitSc.activated.connect(self.quit)

        pg.setConfigOptions(imageAxisOrder='row-major')
        self.setGeometry(30,30,1300,700)
        
        self.setWindowTitle('Pupil-size tracking software')
        
        self.gaussian_smoothing = gaussian_smoothing
        self.slider_nframes = slider_nframes
        
        # menus.mainmenu(self)
        self.online_mode=False
        #menus.onlinemenu(self)

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
        self.p0 = self.win.addViewBox(lockAspect=True,row=0,col=0,invertY=True,border=[100,100,100])
        # self.p0 = pg.ViewBox(lockAspect=False,name='plot1',border=[100,100,100],invertY=True)
        self.p0.setMouseEnabled(x=False,y=False)
        self.p0.setMenuEnabled(False)
        self.pimg = pg.ImageItem()
        self.p0.setAspectLocked()
        self.p0.addItem(self.pimg)

        # image ROI
        self.pROI = self.win.addViewBox(lockAspect=True,row=0,col=1,invertY=True, border=[100,100,100])
        self.pROI.setAspectLocked()
        #self.p0.setMouseEnabled(x=False,y=False)
        self.pROI.setMenuEnabled(False)
        self.pROIimg = pg.ImageItem(None)
        self.pROI.addItem(self.pROIimg)

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
        qlabel = QtGui.QLabel('saturation')
        qlabel.setStyleSheet('color: white;')
        self.l0.addWidget(qlabel,0,8,1,3)

        # adding blanks ("corneal reflections, ...")
        self.reflector = QtGui.QPushButton('add blank')
        self.l0.addWidget(self.reflector, 1, 8+6, 1, 1)
        self.reflector.setEnabled(False)
        self.reflector.clicked.connect(self.add_reflectROI)
        # fit pupil
        self.fit_pupil = QtGui.QPushButton('fit Pupil')
        self.l0.addWidget(self.fit_pupil, 1, 9+6, 1, 1)
        self.fit_pupil.setEnabled(False)
        self.fit_pupil.clicked.connect(self.fit_pupil_size)
        # choose pupil shape
        self.pupil_shape = QtGui.QComboBox(self)
        self.pupil_shape.addItem("Circle fit")
        self.pupil_shape.addItem("Ellipse fit")
        self.l0.addWidget(self.pupil_shape, 1, 10+6, 1, 1)
        # reset
        self.reset_btn = QtGui.QPushButton('reset')
        self.l0.addWidget(self.reset_btn, 1, 11+6, 1, 1)
        self.reset_btn.clicked.connect(self.reset)
        self.reset_btn.setEnabled(True)
        # draw pupil
        self.pupil_draw = QtGui.QPushButton('draw Pupil')
        self.l0.addWidget(self.pupil_draw, 2, 10+6, 1, 1)
        self.pupil_draw.setEnabled(False)
        self.pupil_draw.clicked.connect(self.draw_pupil)
        self.pupil_draw_save = QtGui.QPushButton('- Debug -')
        self.l0.addWidget(self.pupil_draw_save, 2, 11+6, 1, 1)
        # self.pupil_draw_save.setEnabled(False)
        self.pupil_draw_save.setEnabled(True)
        self.pupil_draw_save.clicked.connect(self.debug)

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
        self.p1.setLabel('bottom', 'time (s)')
        self.xaxis = self.p1.getAxis('bottom')
        # self.p1.autoRange(padding=0.01)
        
        self.win.ci.layout.setRowStretchFactor(0,5)
        self.movieLabel = QtGui.QLabel("No movie chosen")
        self.movieLabel.setStyleSheet("color: white;")
        self.l0.addWidget(self.movieLabel,0,1,1,5)
        self.nframes = 0
        self.cframe = 0
        
        self.make_buttons()

        #self.updateButtons()
        self.updateTimer = QtCore.QTimer()
        # self.updateTimer.timeout.connect(self.next_frame)
        self.cframe = 0
        self.loaded = False
        self.Floaded = False
        self.wraw = False
        # self.win.scene().sigMouseClicked.connect(self.plot_clicked)
        self.win.show()
        self.show()
        self.processed = False
        
        if savedir is not None:
            self.save_path = savedir
            self.savelabel.setText(savedir)

    def load_data_batch(self):

        self.batch = True
        self.reset()
        
        file_dialog = QtGui.QFileDialog()
        file_dialog.setFileMode(QtGui.QFileDialog.DirectoryOnly)
        file_dialog.setOption(QtGui.QFileDialog.DontUseNativeDialog, True)
        file_view = file_dialog.findChild(QtGui.QListView, 'listView')

        # to make it possible to select multiple directories:
        if file_view:
            file_view.setSelectionMode(QtGui.QAbstractItemView.MultiSelection)
        f_tree_view = file_dialog.findChild(QtGui.QTreeView)
        if f_tree_view:
            f_tree_view.setSelectionMode(QtGui.QAbstractItemView.MultiSelection)

        if file_dialog.exec():
            paths = file_dialog.selectedFiles()


        self.folders = []
        for path in paths:
            check = check_datafolder(path)
            if check['FaceCamera']:
                self.folders.append(path)
            else:
                print('\n Problem with "%s"' % path)
                print(' ----> The datafolder did not pass the sanity check ! ')
                
        s = ''
        for path in self.folders:
            date, time = from_folder_to_datetime(path)
            s += time+', '
        self.movieLabel.setText("%s => [%s]" % (date, s[:-2]))
        
        if len(self.folders)>0:
            # concatenate datafiles
            process.build_temporal_subsampling(self, folders=self.folders)
            self.addROI.setEnabled(True)
            self.genscript.setEnabled(True)

            self.currentTime.setValidator(QtGui.QDoubleValidator(0, self.times[-1], 2))
            # initialize to first available image
            self.cframe = 0
            self.fullimg = self.Pupil.grab_frame(self.times[self.cframe])
            
            #
            self.Lx, self.Ly = self.fullimg.shape

            self.p1.clear()
            self.p1.plot(self.times, 0*self.times, pen=(0,255,0))
            if self.nframes > 0:
                self.timeLabel.setEnabled(True)
                self.frameSlider.setEnabled(True)
                self.updateFrameSlider()
                self.addROI.setEnabled(True)
            if os.path.isfile(os.path.join(self.folders[0], 'pupil-ROIs.npy')):
                self.load_ROI(datafolder=self.folders[0])
            self.show_fullframe()
            self.plot_pupil_trace()

            
    def load_data(self):

        self.batch = False
        
        self.datafolder = '/home/yann/DATA/2020_10_08/16-02-19/'
        # self.datafolder = QtGui.QFileDialog.getExistingDirectory(self,
        #                                                          "Choose data folder",
        #                                       os.path.join(os.path.expanduser('~'), 'DATA'))


        process.load_data(self,
                          lazy_loading=True,
                          compressed_version=compressed_version)

        if self.Face is not None:
            self.reset()

            if self.Pupil.processed is not None:
                self.data = self.Pupil.processed

            self.sampling_rate = self.Face.sampling_rate
            self.rateBox.setText(str(round(self.sampling_rate)))

            self.nframes = len(self.times)
            
            # update time limits
            self.currentTime.setValidator(QtGui.QDoubleValidator(0, self.times[-1], 2))
            self.time = 0 # initialize to first available image
            self.jump_to_frame()

            self.reset()
            self.Lx, self.Ly = self.fullimg.shape

            self.p1.clear()
            if self.data is not None:
                self.p1.plot(self.data['times'], self.data['diameter'], pen=(0,255,0))
            else:
                self.p1.plot(self.times, 0*self.times, pen=(0,255,0))
                
            # self.movieLabel.setText(os.path.dirname(self.datafolder))
            self.movieLabel.setText("%s => %s" % from_folder_to_datetime(self.datafolder))
            if len(self.times)>0:
                self.timeLabel.setEnabled(True)
                self.frameSlider.setEnabled(True)
                self.updateFrameSlider()
                self.updateButtons()
            self.loaded = True
            self.processed = False

            if os.path.isfile(os.path.join(self.datafolder, 'pupil-ROIs.npy')):
                self.load_ROI()

            self.show_fullframe()
            self.plot_pupil_trace()
            self.genscript.setEnabled(True)
        else:
            print("ERROR: provide a valid data folder !")

            
    def make_buttons(self):
        
        # create frame slider
        self.timeLabel = QtGui.QLabel("Current time (seconds):")
        self.timeLabel.setStyleSheet("color: white;")
        self.currentTime = QtGui.QLineEdit()
        self.currentTime.setText('0.00')
        self.currentTime.setValidator(QtGui.QDoubleValidator(0, 100000, 2))
        self.currentTime.setFixedWidth(50)
        self.currentTime.returnPressed.connect(self.set_precise_time)
        
        self.frameSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.frameSlider.setMinimum(0)
        self.frameSlider.setMaximum(self.slider_nframes)
        self.frameSlider.setTickInterval(1)
        self.frameSlider.setTracking(False)
        self.frameSlider.valueChanged.connect(self.go_to_frame)

        istretch = 23
        iplay = istretch+15
        iconSize = QtCore.QSize(20, 20)

        self.process = QtGui.QPushButton('process data')
        self.process.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.process.clicked.connect(self.process_ROIs)
        self.process.setEnabled(False)
        self.savedata = QtGui.QPushButton('save data')
        self.savedata.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.savedata.clicked.connect(self.save_pupil_data)
        self.savedata.setEnabled(False)
        self.genscript = QtGui.QPushButton('add to bash script')
        self.genscript.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.genscript.clicked.connect(self.gen_bash_script)
        self.genscript.setEnabled(False)

        self.load = QtGui.QPushButton('  load data  \u2b07')
        self.load.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.load.clicked.connect(self.load_data)
        self.load.setEnabled(True)

        self.load_batch = QtGui.QPushButton('  load batch \u2b07')
        self.load_batch.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.load_batch.clicked.connect(self.load_data_batch)
        self.load_batch.setEnabled(True)

        sampLabel = QtGui.QLabel("Sampling rate           (Hz)")
        sampLabel.setStyleSheet("color: gray;")
        self.rateBox = QtGui.QLineEdit()
        self.rateBox.setText(str(self.sampling_rate))
        self.rateBox.setFixedWidth(35)

        smoothLabel = QtGui.QLabel("Smoothing              (px)")
        smoothLabel.setStyleSheet("color: gray;")
        self.smoothBox = QtGui.QLineEdit()
        self.smoothBox.setText(str(self.gaussian_smoothing))
        self.smoothBox.setFixedWidth(25)
        
        self.addROI = QtGui.QPushButton("add Pupil-ROI")
        self.addROI.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.addROI.clicked.connect(self.add_ROI)
        self.addROI.setEnabled(False)

        self.saverois = QtGui.QPushButton('save ROIs')
        self.saverois.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.saverois.clicked.connect(self.save_ROIs)
        self.saverois.setEnabled(False)

        self.batchlist=[]
        self.batchname=[]
        for k in range(6):
            self.batchname.append(QtGui.QLabel(''))
            self.batchname[-1].setStyleSheet("color: white;")
            self.l0.addWidget(self.batchname[-1],18+k,0,1,4)

        iconSize = QtCore.QSize(30, 30)
        self.playButton = QtGui.QToolButton()
        self.playButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPlay))
        self.playButton.setIconSize(iconSize)
        self.playButton.setToolTip("Play")
        self.playButton.setCheckable(True)
        # self.playButton.clicked.connect(self.start)
        self.pauseButton = QtGui.QToolButton()
        self.pauseButton.setCheckable(True)
        self.pauseButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPause))
        self.pauseButton.setIconSize(iconSize)
        self.pauseButton.setToolTip("Pause")
        # self.pauseButton.clicked.connect(self.pause)

        btns = QtGui.QButtonGroup(self)
        btns.addButton(self.playButton,0)
        btns.addButton(self.pauseButton,1)
        btns.setExclusive(True)

        self.l0.addWidget(self.load,2,0,1,3)
        self.l0.addWidget(self.load_batch,3,0,1,3)
        self.l0.addWidget(sampLabel, 8, 0, 1, 3)
        self.l0.addWidget(self.rateBox, 8, 2, 1, 3)
        self.l0.addWidget(smoothLabel, 9, 0, 1, 3)
        self.l0.addWidget(self.smoothBox, 9, 2, 1, 3)
        self.l0.addWidget(self.addROI,14,0,1,3)
        self.l0.addWidget(self.saverois, 16, 0, 1, 3)
        self.l0.addWidget(self.process, 22, 0, 1, 3)
        self.l0.addWidget(self.savedata, 23, 0, 1, 3)
        self.l0.addWidget(self.genscript, 24, 0, 1, 3)
        # self.l0.addWidget(self.processbatch, 21, 0, 1, 3)
        self.l0.addWidget(self.playButton,iplay,0,1,1)
        self.l0.addWidget(self.pauseButton,iplay,1,1,1)

        self.playButton.setEnabled(False)
        self.pauseButton.setEnabled(False)
        self.pauseButton.setChecked(True)

        self.l0.addWidget(QtGui.QLabel(''),istretch,0,1,3)
        self.l0.setRowStretch(istretch,1)
        self.l0.addWidget(self.timeLabel, istretch+13,0,1,3)
        self.l0.addWidget(self.currentTime, istretch+14,0,1,3)
        self.l0.addWidget(self.frameSlider, istretch+15,3,1,15)

        self.l0.addWidget(QtGui.QLabel(''),17,2,1,1)
        self.l0.setRowStretch(16,2)
        ll = QtGui.QLabel('play/pause [SPACE]')
        ll.setStyleSheet("color: gray;")
        self.l0.addWidget(ll, istretch+3+k+1,0,1,4)
        self.updateFrameSlider()

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

    def add_reflectROI(self):
        self.rROI.append(roi.reflectROI(len(self.rROI), moveable=True, parent=self))

    def draw_pupil(self):
        self.pupil = roi.pupilROI(moveable=True, parent=self)
        
    def debug(self):
        np.savez('pupil.npz',
                 **{'img':self.img,
                    'ximg':self.ximg,
                    'yimg':self.yimg,
                    'reflectors':[r.extract_props() for r in self.rROI],
                    'ROIpupil':self.pupil.extract_props(),
                    'ROIellipse':self.ROI.extract_props()})
        
    def add_ROI(self):

        if self.ROI is not None:
            self.ROI.remove(self)
        for r in self.rROI:
            r.remove(self)
        self.ROI = roi.sROI(parent=self)
        self.rROI = []
        self.reflectors = []
        # self.pupil_fit.setEnabled(False)
        self.pupil_draw.setEnabled(True)
        self.fit_pupil.setEnabled(True)
        self.saverois.setEnabled(True)
        
        return

    def load_ROI(self, datafolder=None):

        if datafolder is None:
            datafolder = self.datafolder
        data = np.load(os.path.join(datafolder, 'pupil-ROIs.npy'), allow_pickle=True).item()
        self.saturation = data['ROIsaturation']
        self.sl.setValue(int(self.saturation))
        self.ROI = roi.sROI(parent=self,
                            pos = roi.ellipse_props_to_ROI(data['ROIellipse']))
        self.ROI.plot(self)
        self.rROI = []
        self.reflectors = []
        if 'reflectors' in data:
            for r in data['reflectors']:
                self.rROI.append(roi.reflectROI(len(self.rROI),
                                                pos = roi.ellipse_props_to_ROI(r),
                                                moveable=True, parent=self))
            
        self.pupil_draw.setEnabled(True)
        self.fit_pupil.setEnabled(True)
        self.saverois.setEnabled(True)

    def set_precise_time(self):
        self.time = float(self.currentTime.text())
        t1, t2 = self.xaxis.range
        frac_value = (self.time-t1)/(t2-t1)
        self.frameSlider.setValue(int(self.slider_nframes*frac_value))
        self.jump_to_frame()
        
    def go_to_frame(self):

        t1, t2 = self.xaxis.range
        self.time = t1+(t2-t1)*float(self.frameSlider.value())/self.slider_nframes
        self.jump_to_frame()

    def fitToWindow(self):
        self.movieLabel.setScaledContents(self.fitCheckBox.isChecked())

    def updateFrameSlider(self):
        self.timeLabel.setEnabled(True)
        self.frameSlider.setEnabled(True)

    def updateButtons(self):
        self.playButton.setEnabled(True)
        self.pauseButton.setEnabled(False)
        self.addROI.setEnabled(True)
        self.pauseButton.setChecked(True)
        self.process.setEnabled(True)
        self.saverois.setEnabled(True)

    def jump_to_frame(self):

        self.time, self.fullimg = self.Pupil.grab_frame(self.time, with_time=True)
        self.currentTime.setText('%.2f' % float(self.time))
        
        self.pimg.setImage(self.fullimg)
        self.currentTime.setText('%.2f' % float(self.time))
        if self.ROI is not None:
            self.ROI.plot(self)
        if self.scatter is not None:
            self.p1.removeItem(self.scatter)
        if self.data is not None:
            i0 = np.argmin((self.data['times']-self.time)**2)
            print(self.data['times'][i0], self.time)
            self.scatter.setData(self.data['times'][i0]*np.ones(1),
                                 self.data['diameter'][i0]*np.ones(1),
                                 size=10, brush=pg.mkBrush(255,255,255))
            self.p1.addItem(self.scatter)
            if self.fit is not None:
                self.fit.remove(self)
            coords = []
            for key1, key2 in zip(['cx', 'cy'], ['xmin', 'ymin']):
                coords.append(self.data[key1][i0]-self.data[key2])
            for key in ['sx', 'sy']:
                coords.append(self.data[key][i0])
            self.fit = roi.pupilROI(moveable=True,
                                    parent=self,
                                    color=(0, 200, 0),
                                    pos = roi.ellipse_props_to_ROI(coords))
        self.win.show()
        self.show()
            

    def show_fullframe(self):

        self.pimg.setImage(self.fullimg)
        self.pimg.setLevels([0,255])
        self.currentTime.setText('%.2f' % float(self.times[self.cframe]))
        self.win.show()
        self.show()

    def save_ROIs(self):
        """ """
        
        # format data
        data = {}
        if len(self.rROI)>0:
            data['reflectors'] = [r.extract_props() for r in self.rROI]
        if self.ROI is not None:
            data['ROIellipse'] = self.ROI.extract_props()
        if self.pupil is not None:
            data['ROIpupil'] = self.pupil.extract_props()
        data['ROIsaturation'] = self.saturation
        
        # save in data-folder
        if not self.batch:
            np.save(os.path.join(self.datafolder, 'pupil-ROIs.npy'), data)
            print('successfully save the ROIs as:', os.path.join(self.datafolder, 'pupil-ROIs.npy'))
        else:
            # loop over datafiles !
            for datafolder in self.folders:
                np.save(os.path.join(datafolder, 'pupil-ROIs.npy'), data)
                print('successfully save the ROIs as:', os.path.join(datafolder, 'pupil-ROIs.npy'))
        self.genscript.setEnabled(True)

    def gen_bash_script(self):
        if self.batch:
            fs = self.folders
        else:
            fs = [self.datafolder]
            
        with open('./script.sh', 'a') as f:
            for fn in fs:
                f.write('python pupil/process.py -df %s &\n' % fn)

        print('Script successfully written in "%s"' % str(os.path.abspath('./script.sh')))


    def save_pupil_data(self):
        """ """
        if not self.batch:
            self.data = replace_outliers(self.data)
            np.save(os.path.join(self.datafolder, 'pupil-data.npy'), self.data)
        else:
            # loop over datafiles !
            pass
            
    def process_ROIs(self):

        for x in [self.playButton, self.pauseButton, self.addROI, self.process,
                  self.reflector, self.load_batch, self.load,
                  self.saverois, self.reset_btn, self.pupil_shape, self.pupil_draw, self.fit_pupil]:
            x.setEnabled(False)
        self.win.show()
        self.show()

        print('processing pupil size over the whole recording [...]')
        # process.build_temporal_subsampling(self) # we re-build the sampling
        printProgressBar(0, self.nframes)
        for self.cframe in range(self.nframes):
            # preprocess image
            process.preprocess(self)
            coords = self.fit_pupil_size(None, coords_only=True)
            self.PD[self.cframe] = np.pi*coords[2]*coords[3]
            self.Pr1[self.cframe], self.Pr2[self.cframe] = coords[2], coords[3]
            printProgressBar(self.cframe, self.nframes)
        printProgressBar(self.nframes, self.nframes)
        print('Pupil size calculation over !')

        self.savedata.setEnabled(True)
        self.plot_pupil_trace()
            
        for x in [self.playButton, self.pauseButton, self.addROI, self.process,
                  self.reflector, self.load_batch, self.load,
                  self.saverois, self.reset_btn, self.pupil_shape, self.pupil_draw, self.fit_pupil]:
            x.setEnabled(True)
            
        self.win.show()
        self.show()

    def plot_processed(self):
        pass
    
    def plot_trace(self, wplot, proctype, wroi, color):
        pass
        
    def plot_pupil_trace(self):
            self.p1.clear()
            self.p1.plot(self.data['times'], self.data['diameter'], pen=(0,255,0))
            self.p1.setRange(xRange=(self.times[0],self.times[-1]),
                             yRange=(self.data['diameter'].min()-.1, self.data['diameter'].max()+.1),
                             padding=0.0)
            self.p1.show()

                
    def button_status(self, status):
        self.playButton.setEnabled(status)
        self.pauseButton.setEnabled(status)
        self.frameSlider.setEnabled(status)
        self.process.setEnabled(status)
        self.saverois.setEnabled(status)

    def fit_pupil_size(self, value, coords_only=False):
        
        if not coords_only and (self.pupil is not None):
            self.pupil.remove(self)

        if self.pupil_shape.currentText()=='Ellipse fit':
            coords, shape, res = process.fit_pupil_size(self, shape='ellipse')
        else:
            coords, shape, res = process.fit_pupil_size(self, shape='circle')
            coords = list(coords)+[coords[-1]] # form circle to ellipse

        if not coords_only:
            self.pupil = roi.pupilROI(moveable=True,
                                      pos = roi.ellipse_props_to_ROI(coords),
                                      parent=self)
            
        return coords
            
        
    def quit(self):
        QtWidgets.QApplication.quit()


def run(app, parent=None, compressed_version=False):
    set_dark_style(app)
    set_app_icon(app)
    return MainWindow(app,
                      compressed_version=compressed_version)
    
if __name__=='__main__':
    if sys.argv[-1]=='raw':
        compressed_version = False
    else:
        compressed_version = True
    app = QtWidgets.QApplication(sys.argv)
    main = run(app,
               compressed_version=compressed_version)
    sys.exit(app.exec_())
        
