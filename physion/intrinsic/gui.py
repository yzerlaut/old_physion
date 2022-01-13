import sys, os, shutil, glob, time, subprocess, pathlib, json, tempfile, datetime
import numpy as np
import pynwb, PIL
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
from scipy.interpolate import interp1d
try:
    from pycromanager import Bridge
except ModuleNotFoundError:
    print('camera support not available !')

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from misc.folders import FOLDERS
from misc.guiparts import NewWindow
from assembling.saving import generate_filename_path, day_folder, last_datafolder_in_dayfolder
from visual_stim.psychopy_code.stimuli import visual_stim, visual
import multiprocessing # for the camera streams !!
from intrinsic import analysis

subjects_path = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'exp', 'subjects')

phase_color_map = pg.ColorMap(pos=np.linspace(0.0, 1.0, 3),
                              color=[(255, 0, 0),
                                     (100, 200, 100),
                                     (0, 0, 255)]).getLookupTable(0.0, 1.0, 256)

power_color_map = pg.ColorMap(pos=np.linspace(0.0, 1.0, 3),
                              color=[(0, 0, 0),
                                     (100, 100, 100),
                                     (255, 200, 200)]).getLookupTable(0.0, 1.0, 256)

signal_color_map = pg.ColorMap(pos=np.linspace(0.0, 1.0, 3),
                               color=[(0, 0, 0),
                                      (100, 100, 100),
                                      (255, 255, 255)]).getLookupTable(0.0, 1.0, 256)

class df:
    def __init__(self):
        pass
    def get(self):
        return tempfile.gettempdir()

class dummy_parent:
    def __init__(self):
        self.stop_flag = False
        self.datafolder = df()

class MainWindow(NewWindow):
    
    def __init__(self, app,
                 args=None,
                 parent=None,
                 spatial_subsampling=4,
                 time_subsampling=1):
        """
        Intrinsic Imaging GUI
        """
        self.app = app
        
        super(MainWindow, self).__init__(i=1,
                                         title='intrinsic imaging')

        # some initialisation
        self.running, self.stim = False, None
        self.datafolder = ''
        
        self.t0, self.period = 0, 1
        
        ### trying the camera
        try:
            # we initialize the camera
            self.bridge = Bridge()
            self.core = self.bridge.get_core()
            self.exposure = self.core.get_exposure()
            self.demo = False
            auto_shutter = self.core.get_property('Core', 'AutoShutter')
            self.core.set_property('Core', 'AutoShutter', 0)
        except BaseException as be:
            print(be)
            print('')
            print(' /!\ Problem with the Camera /!\ ')
            print('        --> no camera found ')
            print('')
            self.exposure = -1 # flag for no camera
            self.demo = True
        
        ########################
        ##### building GUI #####
        ########################
        
        self.minView = False
        self.showwindow()

        # layout (from NewWindow class)
        self.init_basic_widget_grid(wdgt_length=3,
                                    Ncol_wdgt=20, Nrow_wdgt=20)
        
        # -- A plot area (ViewBox + axes) for displaying the image ---
        self.view = self.graphics_layout.addViewBox(lockAspect=True, invertY=True)
        self.view.setMenuEnabled(False)
        self.view.setAspectLocked()
        self.pimg = pg.ImageItem()
        
        # ---  setting subject information ---
        self.add_widget(QtWidgets.QLabel('subjects file:'))
        self.subjectFileBox = QtWidgets.QComboBox(self)
        self.subjectFileBox.addItems([f for f in os.listdir(subjects_path)[::-1] if f.endswith('.json')])
        self.subjectFileBox.activated.connect(self.get_subject_list)
        self.add_widget(self.subjectFileBox)

        self.add_widget(QtWidgets.QLabel('subject:'))
        self.subjectBox = QtWidgets.QComboBox(self)
        self.get_subject_list()
        self.add_widget(self.subjectBox)

        self.add_widget(QtWidgets.QLabel(20*' - '))
        self.vascButton = QtWidgets.QPushButton(" - = save Vasculature Picture = - ", self)
        self.vascButton.clicked.connect(self.take_vasculature_picture)
        self.add_widget(self.vascButton)
        
        self.add_widget(QtWidgets.QLabel(20*' - '))
        
        # ---  data acquisition properties ---
        self.add_widget(QtWidgets.QLabel('data folder:'), spec='small-left')
        self.folderB = QtWidgets.QComboBox(self)
        self.folderB.addItems(FOLDERS.keys())
        self.add_widget(self.folderB, spec='large-right')

        self.add_widget(QtWidgets.QLabel('  - protocol:'),
                        spec='small-left')
        self.protocolBox = QtWidgets.QComboBox(self)
        self.protocolBox.addItems(['ALL', 'up', 'down', 'left', 'right'])
        self.add_widget(self.protocolBox,
                        spec='large-right')
        self.add_widget(QtWidgets.QLabel('  - exposure: %.0f ms (from Micro-Manager)' % self.exposure))

        self.add_widget(QtWidgets.QLabel('  - Nrepeat :'),
                        spec='large-left')
        self.repeatBox = QtWidgets.QLineEdit()
        self.repeatBox.setText('10')
        self.add_widget(self.repeatBox, spec='small-right')

        self.add_widget(QtWidgets.QLabel('  - stim. period (s):'),
                        spec='large-left')
        self.periodBox = QtWidgets.QLineEdit()
        self.periodBox.setText('10')
        self.add_widget(self.periodBox, spec='small-right')
        
        self.add_widget(QtWidgets.QLabel('  - bar size (degree):'),
                        spec='large-left')
        self.barBox = QtWidgets.QLineEdit()
        self.barBox.setText('4')
        self.add_widget(self.barBox, spec='small-right')

        self.add_widget(QtWidgets.QLabel('  - spatial sub-sampling (px):'),
                        spec='large-left')
        self.spatialBox = QtWidgets.QLineEdit()
        self.spatialBox.setText(str(spatial_subsampling))
        self.add_widget(self.spatialBox, spec='small-right')

        self.add_widget(QtWidgets.QLabel('  - acq. freq. (Hz):'),
                        spec='large-left')
        self.freqBox = QtWidgets.QLineEdit()
        self.freqBox.setText('1')
        self.add_widget(self.freqBox, spec='small-right')

        self.add_widget(QtWidgets.QLabel('  - flick. freq. (Hz) /!\ > acq:'),
                        spec='large-left')
        self.flickBox = QtWidgets.QLineEdit()
        self.flickBox.setText('20')
        self.add_widget(self.flickBox, spec='small-right')
        
        self.demoBox = QtWidgets.QCheckBox("demo mode")
        self.demoBox.setStyleSheet("color: gray;")
        self.add_widget(self.demoBox, spec='large-left')
        self.demoBox.setChecked(self.demo)

        self.camBox = QtWidgets.QCheckBox("cam.")
        self.camBox.setStyleSheet("color: gray;")
        self.add_widget(self.camBox, spec='small-right')
        self.camBox.setChecked(True)
        
        # ---  launching acquisition ---
        self.liveButton = QtWidgets.QPushButton("--   live view    -- ", self)
        self.liveButton.clicked.connect(self.live_view)
        self.add_widget(self.liveButton)
        
        # ---  launching acquisition ---
        self.acqButton = QtWidgets.QPushButton("-- RUN PROTOCOL -- ", self)
        self.acqButton.clicked.connect(self.launch_protocol)
        self.add_widget(self.acqButton, spec='large-left')
        self.stopButton = QtWidgets.QPushButton(" STOP ", self)
        self.stopButton.clicked.connect(self.stop_protocol)
        self.add_widget(self.stopButton, spec='small-right')

        # ---  launching analysis ---
        self.add_widget(QtWidgets.QLabel(20*' - '))
        self.analysisButton = QtWidgets.QPushButton(" - = Analysis GUI = - ", self)
        self.analysisButton.clicked.connect(self.open_analysis)
        self.add_widget(self.analysisButton, spec='large-left')

        self.pimg.setImage(0*self.get_frame())
        self.view.addItem(self.pimg)
        self.view.autoRange(padding=0.001)
        self.analysisWindow = None

    def take_vasculature_picture(self):

        filename = generate_filename_path(FOLDERS[self.folderB.currentText()],
                            filename='vasculature-%s' % self.subjectBox.currentText(),
                            extension='.npy')
        
        img = self.get_frame(force_HQ=True)
        self.pimg.setImage(img) # show on display
        np.save(filename, img)

        img = np.array(255*(img-img.min())/(img.max()-img.min()), dtype=np.uint8)
        im = PIL.Image.fromarray(img)
        im.save(filename.replace('npy', 'tif'))
        
        print('vasculature image, saved as:')
        print(filename)
        

    
    def open_analysis(self):

        self.analysisWindow =  runAnalysis(self.app,
                                           parent=self)

        
    def get_subject_list(self):
        with open(os.path.join(subjects_path, self.subjectFileBox.currentText())) as f:
            self.subjects = json.load(f)
        self.subjectBox.clear()
        self.subjectBox.addItems(self.subjects.keys())

        
    def init_visual_stim(self, demo=True):

        with open(os.path.join(pathlib.Path(__file__).resolve().parents[1], 'intrinsic', 'vis_stim', 'up.json'), 'r') as fp:
            protocol = json.load(fp)

        if self.demoBox.isChecked():
            protocol['demo'] = True

        self.stim = visual_stim.build_stim(protocol)
        self.parent = dummy_parent()

        
    def get_patterns(self, protocol, angle, size,
                     Npatch=30):

        patterns = []

        if protocol in ['left', 'right']:
            z = np.linspace(-self.stim.screen['resolution'][1], self.stim.screen['resolution'][1], Npatch)
            for i in np.arange(len(z)-1)[(1 if self.flip else 0)::2]:
                patterns.append(visual.Rect(win=self.stim.win,
                                            size=(self.stim.angle_to_pix(size),
                                                  z[1]-z[0]),
                                            pos=(self.stim.angle_to_pix(angle), z[i]),
                                            units='pix', fillColor=1))
        if protocol in ['up', 'down']:
            x = np.linspace(-self.stim.screen['resolution'][0], self.stim.screen['resolution'][0], Npatch)
            for i in np.arange(len(x)-1)[(1 if self.flip else 0)::2]:
                patterns.append(visual.Rect(win=self.stim.win,
                                            size=(x[1]-x[0],
                                                  self.stim.angle_to_pix(size)),
                                            pos=(x[i], self.stim.angle_to_pix(angle)),
                                            units='pix', fillColor=1))

        return patterns

    def run(self):

        self.flip = False
        
        self.stim = visual_stim({"Screen": "Dell-2020",
                                 "presentation-prestim-screen": -1,
                                 "presentation-poststim-screen": -1}, demo=self.demoBox.isChecked())

        self.Nrepeat = int(self.repeatBox.text()) #
        self.period = float(self.periodBox.text()) # degree / second
        self.bar_size = float(self.barBox.text()) # degree / second
        self.dt_save, self.dt = 1./float(self.freqBox.text()), 1./float(self.flickBox.text())
        
        xmin, xmax = 1.1*np.min(self.stim.x), 1.1*np.max(self.stim.x)
        zmin, zmax = 1.3*np.min(self.stim.z), 1.3*np.max(self.stim.z)

        self.angle_start, self.angle_max, self.protocol, self.label = 0, 0, '', ''
        self.Npoints = int(self.period/self.dt_save)

        if self.protocolBox.currentText()=='ALL':
            LABELS, self.label = ['up', 'left', 'down', 'right'], 'up'
        else:
            LABELS, self.label = [self.protocolBox.currentText()], self.protocolBox.currentText()
            
        self.STIM = {'angle_start':[zmin, xmax, zmax, xmin],
                     'angle_stop':[zmax, xmin, zmin, xmax],
                     'label': LABELS,
                     'xmin':xmin, 'xmax':xmax, 'zmin':zmin, 'zmax':zmax}

        for il, label in enumerate(self.STIM['label']):
            self.STIM[label+'-times'] = np.arange(self.Npoints*self.Nrepeat)*self.dt_save
            self.STIM[label+'-angle'] = np.concatenate([np.linspace(self.STIM['angle_start'][il],
                                                                    self.STIM['angle_stop'][il], self.Npoints) for n in range(self.Nrepeat)])

        # initialize one episode:
        self.iEp, self.iTime, self.t0_episode = 0, 0, time.time()

        self.img, self.nSave = self.new_img(), 0

        self.save_metadata()
        
        print('acquisition running [...]')
        
        self.update_dt() # while loop

    def new_img(self):
        return np.zeros(self.imgsize, dtype=np.float64)
    
    def save_img(self):
        
        if self.nSave>0:
            self.img /= self.nSave

        if True: # live display
            self.pimg.setImage(self.img)

        # NEED TO STORE DATA HERE
        self.FRAMES.append(self.img)

        # re-init time step of acquisition
        self.img, self.nSave = self.new_img(), 0
        
    def update_dt(self):

        self.tSave = time.time()

        while (time.time()-self.tSave)<=self.dt_save:

            # show image
            patterns = self.get_patterns(self.STIM['label'][self.iEp%len(self.STIM['label'])],
                                         self.STIM[self.STIM['label'][self.iEp%len(self.STIM['label'])]+'-angle'][self.iTime],
                                         self.bar_size)
            for pattern in patterns:
                pattern.draw()
            try:
                self.stim.win.flip()
            except BaseException:
                pass

            if self.camBox.isChecked():
                # # fetch image
                self.img += self.get_frame()
                self.nSave+=1.0

            time.sleep(self.dt/3.)
            self.flip = (False if self.flip else True) # flip the flag (ADJUST TO HAVE IT ONLY AT DT)
            
        if self.camBox.isChecked():
            self.save_img() # re-init image here

        self.iTime += 1
        
        # checking if not episode over
        if not (self.iTime<len(self.STIM[self.STIM['label'][self.iEp%len(self.STIM['label'])]+'-angle'])):
            if self.camBox.isChecked():
                self.write_data() # writing data when over
            self.t0_episode, self.img, self.nSave = time.time(), self.new_img(), 0
            self.FRAMES = [] # re init data
            self.iTime = 0  
            self.iEp += 1
            
        # continuing ?
        if self.running:
            QtCore.QTimer.singleShot(1, self.update_dt)


    def write_data(self):

        filename = '%s-%i.nwb' % (self.STIM['label'][self.iEp%len(self.STIM['label'])], int(self.iEp/len(self.STIM['label']))+1)
        
        nwbfile = pynwb.NWBFile('Intrinsic Imaging data following bar stimulation',
                                'intrinsic',
                                datetime.datetime.utcnow(),
                                file_create_date=datetime.datetime.utcnow())

        # Create our time series
        angles = pynwb.TimeSeries(name='angle_timeseries',
                                  data=self.STIM[self.STIM['label'][self.iEp%len(self.STIM['label'])]+'-angle'],
                                  unit='Rd',
                                  timestamps=self.STIM[self.STIM['label'][self.iEp%len(self.STIM['label'])]+'-times'])
        nwbfile.add_acquisition(angles)

        images = pynwb.image.ImageSeries(name='image_timeseries',
                                         data=np.array(self.FRAMES, dtype=np.float64),
                                         unit='a.u.',
                                         timestamps=self.STIM[self.STIM['label'][self.iEp%len(self.STIM['label'])]+'-times'])

        nwbfile.add_acquisition(images)
        
        # Write the data to file
        io = pynwb.NWBHDF5IO(os.path.join(self.datafolder, filename), 'w')
        print('writing:', filename)
        io.write(nwbfile)
        io.close()
        print(filename, ' saved !')
        

    def save_metadata(self):
        
        filename = generate_filename_path(FOLDERS[self.folderB.currentText()],
                                          filename='metadata', extension='.npy')
        metadata = {'subject':str(self.subjectBox.currentText()),
                    'exposure':self.exposure,
                    'bar-size':float(self.barBox.text()),
                    'acq-freq':float(self.freqBox.text()),
                    'period':float(self.periodBox.text()),
                    'Nrepeat':int(self.repeatBox.text()),
                    'imgsize':self.imgsize,
                    'STIM':self.STIM}
        
        np.save(filename, metadata)
        self.datafolder = os.path.dirname(filename)

        
    def launch_protocol(self):

        if not self.running:
            self.running = True

            # initialization of data
            self.FRAMES = []
            self.img = self.get_frame()
            self.imgsize = self.img.shape
            self.pimg.setImage(self.img)
            self.view.autoRange(padding=0.001)
            
            self.run()
            
        else:
            print(' /!\  --> pb in launching acquisition (either already running or missing camera)')

    def live_view(self):
        self.running, self.t0 = True, time.time()
        self.update_Image()
        
    def stop_protocol(self):
        if self.running:
            self.running = False
            if self.stim is not None:
                self.stim.close()
        else:
            print('acquisition not launched')

    def get_frame(self, force_HQ=False):
        
        if self.exposure>0:
            self.core.snap_image()
            tagged_image = self.core.get_tagged_image()
            #pixels by default come out as a 1D array. We can reshape them into an image
            img = np.reshape(tagged_image.pix,
                             newshape=[tagged_image.tags['Height'], tagged_image.tags['Width']])
        elif self.stim is not None:
            it = int((time.time()-self.t0_episode)/self.dt_save)%int(self.period/self.dt_save)
            img = np.random.randn(*self.stim.x.shape)+\
                .5*np.exp(-(self.stim.x-(40*it/self.Npoints-20))**2/2./10**2)
        else:
            img = np.random.randn(720, 1280)

        if (int(self.spatialBox.text())>1) and not force_HQ:
            return 1.0*analysis.resample_img(img, int(self.spatialBox.text()))
        else:
            return 1.0*img


        
        
    def update_Image(self):
        # plot it
        self.pimg.setImage(self.get_frame())
        new_t0 = time.time()
        print('dt=%.1f ms' % (1e3*(new_t0-self.t0)))
        self.t0 = new_t0
        if self.running:
            QtCore.QTimer.singleShot(1, self.update_Image)
                
    def hitting_space(self):
        if not self.running:
            self.launch_protocol()
        else:
            self.stop_protocol()

    def process(self):
        self.launch_analysis()
    
        
    def quit(self):
        if self.exposure>0:
            self.bridge.close()
        sys.exit()


class AnalysisWindow(NewWindow):
    
    def __init__(self, app,
                 args=None,
                 parent=None):
        """
        Intrinsic Imaging Analysis GUI
        """
        self.app = app
        
        super(AnalysisWindow, self).__init__(i=2,
                                         title='intrinsic imaging analysis')
        
        self.datafolder = args.datafile
        
        ########################
        ##### building GUI #####
        ########################
        
        self.minView = False
        self.showwindow()

        # layout (from NewWindow class)
        self.init_basic_widget_grid(wdgt_length=3,
                                    Ncol_wdgt=23,
                                    Nrow_wdgt=20)
        
        # --- ROW (Nx_wdgt), COLUMN (Ny_wdgt)
        self.add_widget(QtWidgets.QLabel('data folder:'), spec='small-left')
        self.folderB = QtWidgets.QComboBox(self)
        self.folderB.addItems(FOLDERS.keys())
        self.add_widget(self.folderB, spec='large-right')

        self.raw_trace = self.graphics_layout.addPlot(row=0, col=0, rowspan=1, colspan=23)
        
        self.spectrum_power = self.graphics_layout.addPlot(row=1, col=0, rowspan=2, colspan=9)
        self.spDot = pg.ScatterPlotItem()
        self.spectrum_power.addItem(self.spDot)
        
        self.spectrum_phase = self.graphics_layout.addPlot(row=1, col=9, rowspan=2, colspan=9)
        self.sphDot = pg.ScatterPlotItem()
        self.spectrum_phase.addItem(self.sphDot)

        # images
        self.img1B = self.graphics_layout.addViewBox(row=3, col=0, rowspan=10, colspan=10,
                                                    lockAspect=True, invertY=True)
        self.img1 = pg.ImageItem()
        self.img1B.addItem(self.img1)

        self.img2B = self.graphics_layout.addViewBox(row=3, col=10, rowspan=10, colspan=9,
                                                    lockAspect=True, invertY=True)
        self.img2 = pg.ImageItem()
        self.img2B.addItem(self.img2)

        self.pix1 = pg.ScatterPlotItem()
        self.pix2 = pg.ScatterPlotItem()
        self.img1B.addItem(self.pix1)
        self.img2B.addItem(self.pix2)
        self.img1.setImage(np.random.randn(30,30))
        self.img2.setImage(np.random.randn(30,30))

        for i in range(3):
            self.graphics_layout.ci.layout.setColumnStretchFactor(i, 1)
        self.graphics_layout.ci.layout.setColumnStretchFactor(3, 2)
        self.graphics_layout.ci.layout.setColumnStretchFactor(12, 2)
        self.graphics_layout.ci.layout.setRowStretchFactor(0, 3)
        self.graphics_layout.ci.layout.setRowStretchFactor(1, 4)
        self.graphics_layout.ci.layout.setRowStretchFactor(3, 5)
            
        self.add_widget(QtWidgets.QLabel(' '))
        self.folderButton = QtWidgets.QPushButton("load data [Ctrl+O]", self)
        self.folderButton.clicked.connect(self.open_file)
        self.add_widget(self.folderButton, spec='large-left')
        self.lastBox = QtWidgets.QCheckBox("last ")
        self.lastBox.setStyleSheet("color: gray;")
        self.add_widget(self.lastBox, spec='small-right')
        self.lastBox.setChecked((self.datafolder==''))

        self.add_widget(QtWidgets.QLabel(' '))
        self.rdButton = QtWidgets.QPushButton(" ===== show raw data ===== ", self)
        self.rdButton.clicked.connect(self.show_raw_data)
        self.add_widget(self.rdButton)
        self.add_widget(QtWidgets.QLabel('  - pixel loc. (x,y):'),
                        spec='large-left')
        self.pixBox = QtWidgets.QLineEdit()
        self.pixBox.setText('50, 50')
        self.add_widget(self.pixBox, spec='small-right')
        self.add_widget(QtWidgets.QLabel('  - protocol:'),
                        spec='small-left')
        self.protocolBox = QtWidgets.QComboBox(self)
        self.protocolBox.addItems(['up', 'down', 'left', 'right'])
        self.add_widget(self.protocolBox,
                        spec='small-middle')
        self.numBox = QtWidgets.QComboBox(self)
        self.numBox.addItems(['sum']+[str(i) for i in range(1,10)])
        self.add_widget(self.numBox,
                        spec='small-right')
        self.pmButton = QtWidgets.QPushButton(" ==== compute phase maps ==== ", self)
        self.pmButton.clicked.connect(self.compute_phase_maps)
        self.add_widget(self.pmButton)
        
        self.add_widget(QtWidgets.QLabel(' '))
        self.rmButton = QtWidgets.QPushButton(" === compute retinotopic maps === ", self)
        self.rmButton.clicked.connect(self.compute_retinotopic_maps)
        self.add_widget(self.rmButton)
        self.add_widget(QtWidgets.QLabel('  - direction:'),
                        spec='small-left')
        self.mapBox = QtWidgets.QComboBox(self)
        self.mapBox.addItems(['azimuth', 'altitude'])
        self.add_widget(self.mapBox,
                        spec='large-right')
        self.add_widget(QtWidgets.QLabel('  - spatial smoothing (px):'),
                        spec='large-left')
        self.spatialSmoothingBox = QtWidgets.QLineEdit()
        self.spatialSmoothingBox.setText('5')
        self.add_widget(self.spatialSmoothingBox, spec='small-right')

        self.add_widget(QtWidgets.QLabel('  - temporal smoothing (ms):'),
                        spec='large-left')
        self.temporalSmoothingBox = QtWidgets.QLineEdit()
        self.temporalSmoothingBox.setText('100')
        self.add_widget(self.temporalSmoothingBox, spec='small-right')

        self.add_widget(QtWidgets.QLabel(' '))
        self.pasButton = QtWidgets.QPushButton(" == perform area segmentation == ", self)
        self.pasButton.clicked.connect(self.perform_area_segmentation)
        self.add_widget(self.pasButton)
        self.add_widget(QtWidgets.QLabel('  - display:'),
                        spec='small-left')
        self.displayBox = QtWidgets.QComboBox(self)
        self.displayBox.addItems(['sign map', 'areas (+vasc.)'])
        self.add_widget(self.displayBox,
                        spec='large-right')
        
        self.show()

    def get_pixel_value(self):
        self.pix1.clear()
        self.pix2.clear()
        try:
            x, y = [int(tt) for tt in self.pixBox.text().replace(' ', '').split(',')]
            self.pix1.setData([x], [y],
                               size=10, brush=pg.mkBrush(255,0,0,255))
            self.pix2.setData([x], [y],
                               size=10, brush=pg.mkBrush(255,0,0,255))
            return y, x
        except BaseException as be:
            print(be)
            return 0, 0

    
    def refresh(self):
        self.show_raw_data()
        
    def show_raw_data(self):
        
        if self.numBox.currentText()=='sum':
            # clear previous plots when using the sum
            for plot in [self.raw_trace, self.spectrum_power, self.spectrum_phase]:
                plot.clear()

        # load data
        p, (t, data) = analysis.load_raw_data(self.get_datafolder(),
                                            self.protocolBox.currentText(),
                                            run_id=self.numBox.currentText())

        xpix, ypix = self.get_pixel_value()

        self.raw_trace.plot(t, data[:,xpix, ypix])
        
        self.img1.setLookupTable(signal_color_map)
        self.img2.setLookupTable(signal_color_map)
        self.img1.setImage(data[0, :, :])
        self.img2.setImage(data[-1, :, :])

        spectrum = np.fft.fft(data[:,xpix, ypix])
        power, phase = np.abs(spectrum), -np.angle(spectrum) #%(2.*np.pi)
        x = np.arange(len(power))
        self.spectrum_power.plot(np.log10(x[1:]), np.log10(power[1:]))
        self.spectrum_phase.plot(np.log10(x[1:]), phase[1:])
        self.spectrum_power.plot([np.log10(x[int(p['Nrepeat'])])],
                                 [np.log10(power[int(p['Nrepeat'])])],
                                 size=10, symbolPen='g',
                                 symbol='o')
        self.spectrum_phase.plot([np.log10(x[int(p['Nrepeat'])])],
                                 [phase[int(p['Nrepeat'])]],
                                 size=10, symbolPen='g',
                                 symbol='o')

    def process(self):
        self.compute_phase_maps()
        
    def compute_phase_maps(self):
        p, (t, data) = analysis.load_raw_data(self.get_datafolder(),
                                            self.protocolBox.currentText(),
                                            run_id=self.numBox.currentText())
        power_map, phase_map = analysis.perform_fft_analysis(data, p['Nrepeat'])

        xpix, ypix = self.get_pixel_value()
        print('')
        print('power @ pix', power_map[xpix, ypix])
        print('phase @ pix', phase_map[xpix, ypix])

        self.img1.setLookupTable(power_color_map)
        self.img2.setLookupTable(phase_color_map)
        self.img1.setImage(power_map)
        self.img2.setImage(phase_map)
        

    def compute_retinotopic_maps(self):

        power_map, delay_map = analysis.get_retinotopic_maps(self.get_datafolder(),
                                                             self.mapBox.currentText(),
                                                             run_id=self.numBox.currentText())

        
        self.img1.setLookupTable(power_color_map)
        self.img2.setLookupTable(phase_color_map)
        
        self.img1.setImage(power_map)
        self.img2.setImage(delay_map)
        

    def perform_area_segmentation(self):
        pass

    def get_datafolder(self):

        if self.lastBox.isChecked():
            self.datafolder = last_datafolder_in_dayfolder(day_folder(FOLDERS[self.folderB.currentText()])
                                                           ,
                                              with_NIdaq=False)
            #
        elif self.datafolder!='':
            pass
        else:
            print('need to set a proper datafolder !')

        return self.datafolder
        
    def open_file(self):

        self.lastBox.setChecked(False)
        folder = QtWidgets.QFileDialog.getExistingDirectory(self,\
                                                            "Choose datafolder",
                                                            FOLDERS[self.folderB.currentText()])
        
        if folder!='':
            self.datafolder = folder
        else:
            print('data-folder not set !')

    def launch_analysis(self):
        print('launching analysis [...]')
        if self.datafolder=='' and self.lastBox.isChecked():
            self.datafolder = last_datafolder_in_dayfolder(day_folder(os.path.join(FOLDERS[self.folderB.currentText()])),
                                                           with_NIdaq=False)
        analysis.run(self.datafolder, show=True)
        print('-> analysis done !')

    def pick_display(self):

        if self.displayBox.currentText()=='horizontal-map':
            print('show horizontal map')
        elif self.displayBox.currentText()=='vertical-map':
            print('show vertical map')
            

    
        
def run(app, args=None, parent=None):
    return MainWindow(app,
                      args=args,
                      parent=parent)

def runAnalysis(app, args=None, parent=None):
    return AnalysisWindow(app,
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
    parser.add_argument('-a', "--analysis", action="store_true")
    
    args = parser.parse_args()
    app = QtWidgets.QApplication(sys.argv)
    build_dark_palette(app)
    if args.analysis:
        main = AnalysisWindow(app,
                              args=args)
    else:
        main = MainWindow(app,
                          args=args)
    sys.exit(app.exec_())
