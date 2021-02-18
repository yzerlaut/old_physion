import sys, os, shutil, glob, time, pynwb
import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
from pyqtgraph import GraphicsScene
from scipy.stats import zscore, skew
from matplotlib import cm
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from pupil import guiparts, process, roi
from misc.style import set_dark_style, set_app_icon
from assembling.saving import from_folder_to_datetime, check_datafolder

from dataviz.plots import convert_index_to_time

class MainWindow(QtWidgets.QMainWindow):
    
    def __init__(self, app,
                 args=None,
                 parent=None,
                 gaussian_smoothing=2,
                 subsampling=1000):
        """
        sampling in Hz
        """
        self.app = app
        
        super(MainWindow, self).__init__()

        
        self.setGeometry(100,100,400,400)
        
        self.compressed_version=False

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
        self.minView = False
        self.showwindow()
        
        pg.setConfigOptions(imageAxisOrder='row-major')
        
        self.setWindowTitle('Pupil-size tracking module -- Physion')
        
        self.gaussian_smoothing = gaussian_smoothing
        self.subsampling = subsampling
        
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
        self.p0 = self.win.addViewBox(lockAspect=False,row=0,col=0,invertY=True,border=[100,100,100])
        # self.p0 = pg.ViewBox(lockAspect=False,name='plot1',border=[100,100,100],invertY=True)
        self.p0.setMouseEnabled(x=False,y=False)
        self.p0.setMenuEnabled(False)
        self.pimg = pg.ImageItem()
        self.p0.setAspectLocked()
        self.p0.addItem(self.pimg)

        # image ROI
        self.pPupil = self.win.addViewBox(lockAspect=False,row=0,col=1,invertY=True, border=[100,100,100])
        self.pPupil.setAspectLocked()
        #self.p0.setMouseEnabled(x=False,y=False)
        self.pPupil.setMenuEnabled(False)
        self.pPupilimg = pg.ImageItem(None)
        self.pPupil.addItem(self.pPupilimg)

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
        self.reflector.setEnabled(True)
        self.reflector.clicked.connect(self.add_reflectROI)
        # fit pupil
        self.fit_pupil = QtGui.QPushButton('fit Pupil')
        self.l0.addWidget(self.fit_pupil, 1, 9+6, 1, 1)
        self.fit_pupil.setEnabled(True)
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
        self.reset_btn.setEnabled(True)
        # draw pupil
        self.pupil_draw = QtGui.QPushButton('draw Pupil')
        self.l0.addWidget(self.pupil_draw, 2, 10+6, 1, 1)
        self.pupil_draw.setEnabled(True)
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
        self.p1.setLabel('bottom', 'time (frame #)')
        self.xaxis = self.p1.getAxis('bottom')
        self.p1.autoRange(padding=0.01)
        
        self.win.ci.layout.setRowStretchFactor(0,5)
        self.movieLabel = QtGui.QLabel("No datafile chosen")
        self.movieLabel.setStyleSheet("color: white;")
        self.l0.addWidget(self.movieLabel,0,1,1,5)
        self.nframes = 0
        self.cframe = 0
        self.make_buttons()

        self.updateTimer = QtCore.QTimer()
        self.cframe = 0
        
        self.win.show()
        self.show()
        self.processed = False

        self.datafile = args.datafile

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

        self.batch = False
        self.cframe = 0
        
        filename = os.path.join('C:\\Users\\yann.zerlaut\\DATA\\2021_02_16\\15-41-13', 'metadata.npy') # a default for debugging
        
        # filename, _ = QtGui.QFileDialog.getOpenFileName(self,
        #              "Open Pupil Data(through metadata file or NWB file) ",
        #                 os.path.join(os.path.expanduser('~'),'DATA'),
        #                             filter="*.nwb, metadata.npy")

        
        if filename.endswith('.nwb'):
            self.reset()
            self.datafile = filename
            nwbfile = pynwb.NWBHDF5IO(args.datafile, 'r').read()
            self.FaceCamera = nwbfile.acquisition['FaceCamera']
            self.nframes, self.Lx, self.Ly = self.FaceCamera.data.shape
            self.FILES, self.times, self.imgfolder = None, None, None
            self.datafolder = os.path.abspath(filename)
        elif os.path.isdir(filename.replace('metadata.npy', 'FaceCamera-imgs')):
            self.reset()
            self.datafolder, self.FaceCamera = os.path.dirname(filename), None
            self.imgfolder = os.path.join(self.datafolder, 'FaceCamera-imgs')
            times = np.array([float(f.replace('.npy', '')) for f in os.listdir(self.imgfolder) if f.endswith('.npy')])
            self.times = times[np.argsort(times)]
            self.FILES = np.array([f for f in os.listdir(self.imgfolder) if f.endswith('.npy')])[np.argsort(times)]
            self.nframes = len(self.times)
            self.Lx, self.Ly = np.load(os.path.join(self.imgfolder, self.FILES[0])).shape
            print('Sampling frequency: %.1f Hz' % (1./np.diff(self.times).mean()))

        else:
            print(' /!\ no FaceCamera data found ...')
            
        if os.path.isfile(os.path.join(os.path.dirname(filename), 'pupil.npy')):
            self.data = np.load(os.path.join(os.path.dirname(filename), 'pupil.npy'),
                                allow_pickle=True).item()
            self.smoothBox.setText('%i' % self.data['gaussian_smoothing'])
            process.load_ROI(self)
            self.plot_pupil_trace()
            
        self.jump_to_frame()
            
        self.timeLabel.setEnabled(True)
        self.frameSlider.setEnabled(True)
        self.updateFrameSlider()
        self.updateButtons()
        self.currentTime.setValidator(\
            QtGui.QDoubleValidator(0, self.nframes, 2))

        self.movieLabel.setText(filename)

        self.addROI.setEnabled(True)
        self.saverois.setEnabled(True)
        
        

            
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
        self.frameSlider.setMaximum(200)
        self.frameSlider.setTickInterval(1)
        self.frameSlider.setTracking(False)
        self.frameSlider.valueChanged.connect(self.go_to_frame)

        istretch = 23
        iplay = istretch+15
        iconSize = QtCore.QSize(20, 20)

        self.process = QtGui.QPushButton('process data')
        self.process.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.process.clicked.connect(self.process_ROIs)
        self.process.setEnabled(True)
        
        self.savedata = QtGui.QPushButton('save data')
        self.savedata.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.savedata.clicked.connect(self.save_pupil_data)
        self.savedata.setEnabled(True)
        
        self.genscript = QtGui.QPushButton('add to bash script')
        self.genscript.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.genscript.clicked.connect(self.gen_bash_script)
        self.genscript.setEnabled(True)

        self.load = QtGui.QPushButton('  load data  \u2b07')
        self.load.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.load.clicked.connect(self.load_data)
        self.load.setEnabled(True)

        self.load_batch = QtGui.QPushButton('  load batch \u2b07')
        self.load_batch.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.load_batch.clicked.connect(self.load_data_batch)
        # self.load_batch.setEnabled(True)

        sampLabel = QtGui.QLabel("Subsampling          (frame)")
        sampLabel.setStyleSheet("color: gray;")
        self.samplingBox = QtGui.QLineEdit()
        self.samplingBox.setText(str(self.subsampling))
        self.samplingBox.setFixedWidth(35)

        smoothLabel = QtGui.QLabel("Smoothing              (px)")
        smoothLabel.setStyleSheet("color: gray;")
        self.smoothBox = QtGui.QLineEdit()
        self.smoothBox.setText(str(self.gaussian_smoothing))
        self.smoothBox.setFixedWidth(25)
        
        self.addROI = QtGui.QPushButton("add Pupil-ROI")
        self.addROI.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.addROI.clicked.connect(self.add_ROI)
        self.addROI.setEnabled(True)

        self.saverois = QtGui.QPushButton('save ROIs')
        self.saverois.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.saverois.clicked.connect(self.save_ROIs)
        self.saverois.setEnabled(True)

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
        self.l0.addWidget(self.samplingBox, 8, 2, 1, 3)
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
        

    # def load_ROI(self, with_plot=True):

    #     self.saturation = self.data['ROIsaturation']
    #     self.sl.setValue(int(self.saturation))
    #     self.ROI = roi.sROI(parent=self,
    #                         pos = roi.ellipse_props_to_ROI(self.data['ROIellipse']))
    #     if with_plot:
    #         self.ROI.plot(self)
    #     self.rROI = []
    #     self.reflectors = []
    #     if 'reflectors' in self.data:
    #         for r in self.data['reflectors']:
    #             self.rROI.append(roi.reflectROI(len(self.rROI),
    #                                             pos = roi.ellipse_props_to_ROI(r),
    #                                             moveable=True, parent=self))
            

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

    def updateButtons(self):
        self.playButton.setEnabled(True)
        self.pauseButton.setEnabled(False)
        self.addROI.setEnabled(True)
        self.pauseButton.setChecked(True)
        self.process.setEnabled(True)
        self.saverois.setEnabled(True)

    def jump_to_frame(self):

        # full image 
        if self.FaceCamera is not None:
            self.fullimg = self.FaceCamera.data[self.cframe,:,:]
        else:
            self.fullimg = np.load(os.path.join(self.imgfolder, self.FILES[self.cframe]))
        self.pimg.setImage(self.fullimg)

        # zoomed image
        if self.ROI is not None:
            process.init_fit_area(self)
            process.preprocess(self,\
                    gaussian_smoothing=float(self.smoothBox.text()))
            self.pPupilimg.setImage(self.img)
            self.pPupilimg.setLevels([self.img.min(), self.img.max()])
        
            self.reflector.setEnabled(False)
            self.reflector.setEnabled(True)
            
        if self.scatter is not None:
            self.p1.removeItem(self.scatter)
        if self.fit is not None:
            self.fit.remove(self)
            
        if self.data is not None:
            self.scatter.setData(self.cframe*np.ones(1),
                                 self.data['diameter'][self.cframe]*np.ones(1),
                                 size=10, brush=pg.mkBrush(255,255,255))
            self.p1.addItem(self.scatter)
            self.p1.show()
            coords = []
            if 'sx-corrected' in self.data:
                for key in ['cx-corrected', 'cy-corrected',
                            'sx-corrected', 'sy-corrected']:
                    coords.append(self.data[key][self.cframe])
            else:
                for key in ['cx', 'cy', 'sx', 'sy']:
                    coords.append(self.data[key][self.cframe])
            self.fit = roi.pupilROI(moveable=True,
                                    parent=self,
                                    color=(0, 200, 0),
                                    pos = roi.ellipse_props_to_ROI(coords))
            
        self.win.show()
        self.show()
            

    def show_fullframe(self):

        self.pimg.setImage(self.fullimg)
        self.pimg.setLevels([0,255])
        # self.currentTime.setText('%.2f' % float(self.times[self.cframe]))
        self.win.show()
        self.show()

    def extract_ROI(self, data):

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
        
        
    def build_data(self):

        data = {}
        self.extract_ROI(data)
        
        data['diameter'] = np.ones(self.nframes)
        data['cx'] = np.ones(self.nframes)*.5*(data['xmax']-data['xmin'])
        data['cy'] = np.ones(self.nframes)*.5*(data['ymax']-data['ymin'])
        data['sx'] = 10*np.ones(self.nframes)
        data['sy'] = 10*np.ones(self.nframes)
        
        return data
        
    def save_ROIs(self):
        """ """
        self.save_pupil_data()
        # data = self.build_data()
        # fn = os.path.join(self.datafolder, 'pupil.npy')
        # np.save(fn, data)
        # print('successfully save the ROIs as: "%s" ' % fn)
        

    def gen_bash_script(self):
            
        with open('./script.sh', 'a') as f:
            for fn in fs:
                f.write('python pupil/process.py -df %s &\n' % fn)
        print('Script successfully written in "%s"' % str(os.path.abspath('./script.sh')))


    def save_pupil_data(self):
        """ """
        if self.pupil_shape.currentText()=='Ellipse fit':
            self.data['shape'] = 'ellipse'
        else:
            self.data['shape'] = 'circle'
        self.data['gaussian_smoothing'] = int(self.smoothBox.text())
        np.save(os.path.join(self.datafolder, 'pupil.npy'), self.data)
        print('Data successfully saved as "%s"' % os.path.join(self.datafolder, 'pupil.npy'))
        
            
    def process_ROIs(self):

        self.data = self.build_data()

        self.subsampling = int(self.samplingBox.text())
        if self.pupil_shape.currentText()=='Ellipse fit':
            self.Pshape = 'ellipse'
        else:
            self.Pshape = 'circle'

        print('processing pupil size over the whole recording [...]')
        print(' with %i frame subsampling' % self.subsampling)

        temp = process.perform_loop(self,
                                    subsampling=self.subsampling,
                                    shape=self.Pshape,
                                    gaussian_smoothing=int(self.smoothBox.text()),
                                    with_ProgressBar=True)

        if self.subsampling==1: # we just take those data
            for key in ['cx', 'cy', 'sx', 'sy', 'diameter', 'residual']:
                self.data[key] = temp[key]
        else:
            from scipy.interpolate import interp1d
            # we interpolate the full data
            for key in ['cx', 'cy', 'sx', 'sy', 'diameter', 'residual']:
                func = interp1d(temp['frame'], temp[key], kind='linear')
                self.data[key] = func(np.arange(self.nframes))

        # removing outliers in the fit
        self.data = process.replace_outliers(self.data)
                
        self.plot_pupil_trace()
            
        self.win.show()
        self.show()

    def plot_pupil_trace(self):
        self.p1.clear()
        if self.data is not None:
            self.p1.plot(np.arange(len(self.data['cx'])), self.data['diameter'], pen=(0,255,0))
            self.p1.setRange(xRange=(0, len(self.data['cx'])),
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
            coords, _, _ = process.perform_fit(self, shape='ellipse')
        else:
            coords, _, _ = process.perform_fit(self, shape='circle')
            coords = list(coords)+[coords[-1]] # form circle to ellipse

        if not coords_only:
            self.pupil = roi.pupilROI(moveable=True,
                                      pos = roi.ellipse_props_to_ROI(coords),
                                      parent=self)

        return coords
            
        
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



    
