import sys, os, shutil, glob, time, subprocess, pathlib, json, tempfile, datetime
import numpy as np
import pynwb, PIL
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
from scipy.interpolate import interp1d

# pycromanager for the intrinsic imaging camera
try:
    from pycromanager import Bridge
except ModuleNotFoundError:
    print('camera support not available !')

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from misc.folders import FOLDERS
from misc.guiparts import NewWindow
from assembling.saving import generate_filename_path, day_folder, last_datafolder_in_dayfolder

from intrinsic import analysis

# NIDAQ API for the whisker puff stimulation
try:
    from hardware_control.NIdaq.main import Acquisition
except ModuleNotFoundError:
    print(' /!\ Problem with the NIdaq module /!\ ')

import multiprocessing # for the camera streams !!

subjects_path = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'exp', 'subjects')


class MainWindow(NewWindow):
    
    def __init__(self, app,
                 args=None,
                 parent=None,
                 spatial_subsampling=4,
                 delay=2.,
                 time_subsampling=1):
        """
        Intrinsic Imaging GUI
        """
        self.app = app
        
        super(MainWindow, self).__init__(i=1,
                                         title='intrinsic imaging')

        # some initialisation
        self.running, self.stim= False, None
        self.datafolder, self.img, self.vasculature_img = '', None, None
        
        self.t0, self.period, self.delay = 0, 1, delay
        
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
        self.add_widget(QtWidgets.QLabel('folder:'), spec='small-left')
        self.folderB = QtWidgets.QComboBox(self)
        self.folderB.addItems(FOLDERS.keys())
        self.add_widget(self.folderB, spec='large-right')

        self.add_widget(QtWidgets.QLabel('- stim:'),
                        spec='small-left')
        self.protocolBox = QtWidgets.QComboBox(self)
        self.protocolBox.addItems(['air puff - whisker'])
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
        
        self.add_widget(QtWidgets.QLabel('  - spatial sub-sampling (px):'),
                        spec='large-left')
        self.spatialBox = QtWidgets.QLineEdit()
        self.spatialBox.setText(str(spatial_subsampling))
        self.add_widget(self.spatialBox, spec='small-right')

        self.add_widget(QtWidgets.QLabel('  - acq. freq. (Hz):'),
                        spec='large-left')
        self.freqBox = QtWidgets.QLineEdit()
        self.freqBox.setText('10')
        self.add_widget(self.freqBox, spec='small-right')

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
                            extension='.tif')
        
        # save HQ image as tiff
        img = self.get_frame(force_HQ=True)
        np.save(filename.replace('.tif', '.npy'), img)
        img = np.array(255*(img-img.min())/(img.max()-img.min()), dtype=np.uint8)
        im = PIL.Image.fromarray(img)
        im.save(filename)
        print('vasculature image, saved as:')
        print(filename)

        # then keep a version to store with imaging:
        self.vasculature_img = self.get_frame()
        self.pimg.setImage(self.vasculature_img) # show on display

    
    def open_analysis(self):

        self.analysisWindow =  runAnalysis(self.app,
                                           parent=self)

        
    def get_subject_list(self):
        with open(os.path.join(subjects_path, self.subjectFileBox.currentText())) as f:
            self.subjects = json.load(f)
        self.subjectBox.clear()
        self.subjectBox.addItems(self.subjects.keys())

        
    def init_whisker_stim(self, demo=True):

        self.acq = None # by default

        output_steps = [{"channel":0, "onset":1, "duration":0.1, "value":0},
                        {"channel":1, "onset":1, "duration":0.1, "value":0}]+\
                       [{"channel":2,
		         "onset": self.delay+Dt,
		         "duration":0.5,
		         "value":5.0} for Dt in np.arange(self.Nrepeat)*self.period]

        if not self.demoBox.isChecked():
            try:
                self.acq = Acquisition(dt=1e-3, # forced to 1kHz
                                       Nchannel_analog_in=1,
                                       Nchannel_digital_in=0,
                                       max_time=self.tstop,
                                       output_steps=output_steps,
                                       filename= self.filename.replace('metadata', 'NIdaq'))
                self.acq.launch()
            except BaseException as e:
                print(e)
                print(' /!\ PB WITH NI-DAQ /!\ ')

        
    def run(self):

        
        self.Nrepeat = int(self.repeatBox.text()) #
        self.period = float(self.periodBox.text()) # degree / second
        self.tstop = 1+self.Nrepeat*self.period+1 # 1s hard-coded security around TOI

        self.dt = 1./float(self.freqBox.text())
        self.protocol, self.label = '', ''
        self.times = []

        self.save_metadata()

        # initialize one episode:
        self.iEp, self.iTime, self.t0 = 0, 0, time.time()
        self.img, self.nSave = self.new_img(), 0
        
        print('- initializing whisker stim on NIdaq [...]')
        self.init_whisker_stim()
        
        print('- acquisition running [...]')

        # while loop:
        self.update_dt()


    def new_img(self):
        return np.zeros(self.imgsize, dtype=np.float64)
    
    def save_img(self):
        
        if self.nSave>0:
            self.img /= self.nSave

        if True: # live display
            self.pimg.setImage(self.img)

        # NEED TO STORE DATA HERE
        self.FRAMES.append(self.img)
        self.times.append(time.time()-self.t0)

        # re-init time step of acquisition
        self.img, self.nSave = self.new_img(), 0
        
    def update_dt(self):

        self.tSave = time.time()

        while (time.time()-self.tSave)<=self.dt:

            self.t = time.time()

            if self.camBox.isChecked():
                # # fetch image
                self.img += self.get_frame()
                self.nSave+=1.0

            # time.sleep(max([self.dt-(time.time()-self.t), 0])) 
            # self.flip = (False if self.flip else True) # flip the flag
            
        if self.camBox.isChecked():
            self.save_img() # re-init image here

        self.iTime += 1
        
        # continuing ?
        if self.running and ((time.time()-self.t0)<self.tstop):
            QtCore.QTimer.singleShot(1, self.update_dt)
        else:
            # stop
            if self.acq is not None:
                self.acq.close()
                self.acq = None # reset acq
                # then write data
            if self.camBox.isChecked():
                self.write_data() # writing data when over
            print('- acquisition over ! \n -------------------------------------- \n ')

    def write_data(self):

        filename = self.filename.replace('metadata.npy', 'intrinsic-whisker-stim.nwb')
        
        nwbfile = pynwb.NWBFile(session_description=\
                'Intrinsic Imaging data following whisker stimulation',
                                identifier=self.filename,
                                session_start_time=\
                                datetime.datetime.now(datetime.timezone.utc))

        if os.path.isfile(self.filename.replace('metadata', 'NIdaq')):
            NIdaq_data = np.load(self.filename.replace('metadata', 'NIdaq'),
                    allow_pickle=True).item()
            puff_data = NIdaq_data['analog'][0]
        else:
            puff_data = np.zeros(int(self.tstop/1e-3)) # 

        puff = pynwb.TimeSeries(name='puff-control_timeseries',
                                data=puff_data, unit='V',
                                timestamps=1e-3*np.arange(len(puff_data)))
        nwbfile.add_acquisition(puff)

        # we resample the data to a regularly sample dataset
        func = interp1d(self.times,
                        np.array(self.FRAMES, dtype=np.float64),
                        axis=0)
        new_t = np.arange(int(self.period*self.Nrepeat/self.dt))*self.dt+\
                self.times[0]+self.dt
        images = pynwb.image.ImageSeries(name='image_timeseries',
                                         data=func(new_t),
                                         unit='a.u.',
                                         timestamps=new_t)

        nwbfile.add_acquisition(images)
        
        # Write the data to file
        io = pynwb.NWBHDF5IO(os.path.join(self.datafolder, filename), 'w')
        print('- writing:', filename, ' [...]')
        io.write(nwbfile)
        io.close()
        print('- data saved as: ', os.path.join(self.datafolder, filename))
        

    def save_metadata(self):
        
        self.filename = generate_filename_path(FOLDERS[self.folderB.currentText()],
                                               filename='metadata', extension='.npy')

        metadata = {'subject':str(self.subjectBox.currentText()),
                    'exposure':self.exposure,
                    'acq-freq':float(self.freqBox.text()),
                    'period':float(self.periodBox.text()),
                    'Nrepeat':int(self.repeatBox.text()),
                    'delay':self.delay,
                    'imgsize':self.imgsize}
        
        np.save(self.filename, metadata)
        if self.vasculature_img is not None:
            np.save(self.filename.replace('metadata', 'vasculature'),
                    self.vasculature_img)
            
        self.datafolder = os.path.dirname(self.filename)

        
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

    def refresh(self):
        # to have the "r" shortcut working
        if self.running:
            self.stop_protocol()
        else:
            self.launch_protocol()

    def get_frame(self, force_HQ=False):
        
        if self.exposure>0:
            self.core.snap_image()
            tagged_image = self.core.get_tagged_image()
            #pixels by default come out as a 1D array. We can reshape them into an image
            img = np.reshape(tagged_image.pix,
                             newshape=[tagged_image.tags['Height'], tagged_image.tags['Width']])
        else:
            img = np.random.randn(720, 1280)
            if int(1000*(time.time()-self.t0-self.delay-1))%(int(1000*self.period))<(1000*0.3):
                img[300:500,200:600] = -1

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
        self.open_analysis()
    
        
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

        self.datafolder, self.img, self.vasculature_img = '', None, None
        self.data, self.pData = None, None

        if args is not None:
            self.datafolder = args.datafile
        else:
            self.datafolder = ''
        
        ########################
        ##### building GUI #####
        ########################
        
        self.minView = False
        self.showwindow()

        # layout (from NewWindow class)
        self.init_basic_widget_grid(wdgt_length=3,
                                    Ncol_wdgt=13,
                                    Nrow_wdgt=20)
        
        # --- ROW (Nx_wdgt), COLUMN (Ny_wdgt)
        self.raw_trace = self.graphics_layout.addPlot(row=0, col=2, rowspan=5, colspan=10)

        self.mean_trace = self.graphics_layout.addPlot(row=5, col=2, rowspan=5, colspan=5)
        
        self.spectrum_power = self.graphics_layout.addPlot(row=10, col=2, rowspan=5, colspan=5)
        self.spDot = pg.ScatterPlotItem()
        self.spectrum_power.addItem(self.spDot)
        
        self.spectrum_phase = self.graphics_layout.addPlot(row=15, col=2, rowspan=5, colspan=5)
        self.sphDot = pg.ScatterPlotItem()
        self.spectrum_phase.addItem(self.sphDot)

        # # images
        self.img1B = self.graphics_layout.addViewBox(row=5, col=7, rowspan=15, colspan=5,
                                                     lockAspect=True, invertY=True)
        self.img1 = pg.ImageItem()
        self.img1B.addItem(self.img1)
        self.img = np.random.randn(30,30)
        self.img1.setImage(self.img)


        self.add_widget(QtWidgets.QLabel('folder:'), spec='small-left')
        self.folderB = QtWidgets.QComboBox(self)
        self.folderB.addItems(FOLDERS.keys())
        self.add_widget(self.folderB, spec='large-right')

        self.folderButton = QtWidgets.QPushButton("open data [O]", self)
        self.folderButton.clicked.connect(self.open_file)
        self.add_widget(self.folderButton, spec='large-left')
        self.lastBox = QtWidgets.QCheckBox("last ")
        self.lastBox.setStyleSheet("color: gray;")
        self.add_widget(self.lastBox, spec='small-right')
        self.lastBox.setChecked((self.datafolder==''))
        self.dataLabel = QtWidgets.QLabel('...')
        self.add_widget(self.dataLabel)

        self.add_widget(QtWidgets.QLabel(' '))
        # self.delayLabel = QtWidgets.QLabel(' - Delay= ... ')
        # self.add_widget(self.delayLabel)
        self.repeatLabel = QtWidgets.QLabel(' - Nrepeat')
        self.add_widget(self.repeatLabel)
        self.periodLabel = QtWidgets.QLabel(' - Period')
        self.add_widget(self.periodLabel)
        self.freqLabel = QtWidgets.QLabel(' - Acq-Freq')
        self.add_widget(self.freqLabel)
        self.add_widget(QtWidgets.QLabel(' '))

        self.vdButton = QtWidgets.QPushButton("vasculature img", self)
        self.vdButton.clicked.connect(self.show_vasc_pic)
        self.add_widget(self.vdButton)

        self.add_widget(QtWidgets.QLabel(' '))

        self.rdButton = QtWidgets.QPushButton(" === show raw data === ", self)
        self.rdButton.clicked.connect(self.show_raw_data)
        self.add_widget(self.rdButton)

        self.add_widget(QtWidgets.QLabel('  - phase conv.:'),
                        spec='large-left')
        self.twoPiBox = QtWidgets.QCheckBox("[0,2pi]")
        self.twoPiBox.setStyleSheet("color: gray;")
        self.add_widget(self.twoPiBox, spec='small-right')

        self.add_widget(QtWidgets.QLabel('  - high pass filtering:'),
                        spec='large-left')
        self.hpBox = QtWidgets.QLineEdit()
        self.hpBox.setText('0')
        self.add_widget(self.hpBox, spec='small-right')

        self.add_widget(QtWidgets.QLabel('  - spatial smoothing (px):'),
                        spec='large-left')
        self.spatialSmoothingBox = QtWidgets.QLineEdit()
        self.spatialSmoothingBox.setText('0')
        self.add_widget(self.spatialSmoothingBox, spec='small-right')

        self.add_widget(QtWidgets.QLabel('  - temporal smoothing (#):'),
                        spec='large-left')
        self.temporalSmoothingBox = QtWidgets.QLineEdit()
        self.temporalSmoothingBox.setText('0')
        self.add_widget(self.temporalSmoothingBox, spec='small-right')

        self.stdNormBox = QtWidgets.QCheckBox("std norm.[0,2pi]")
        self.stdNormBox.setStyleSheet("color: gray;")
        self.add_widget(self.stdNormBox, spec='large-right')

        self.avgButton = QtWidgets.QPushButton(" === avg std map === ", self)
        self.avgButton.clicked.connect(self.compute_avgs)
        self.add_widget(self.avgButton)

        self.mapButton = QtWidgets.QPushButton(" === power map === ", self)
        self.mapButton.clicked.connect(self.compute_maps)
        self.add_widget(self.mapButton)

        self.pixROI = pg.ROI((10, 10), size=(5,5),
                             pen=pg.mkPen((255,0,0,255), width=4),
                             rotatable=False,resizable=False)
        self.img1B.addItem(self.pixROI)
        self.pixROI.sigRegionChangeFinished.connect(self.traces_update)

        self.show()

    def set_pixROI(self, img=None):
        if self.img is not None:
            img = self.img
        self.pixROI.setSize((5,5))
        return 
    
    def get_pixel_value(self):
        y, x = int(self.pixROI.pos()[0]+2), int(self.pixROI.pos()[1]+2) # because size=(5,5)
        return x, y
        

    def show_vasc_pic(self):
        pic = os.path.join(self.get_datafolder(), 'vasculature.npy')
        if os.path.isfile(pic):
            self.img1.setImage(np.load(pic))
            
    def refresh(self):
        self.show_raw_data()

    def load_data(self):

        self.params = np.load(os.path.join(self.get_datafolder(),
                             'metadata.npy'), allow_pickle=True).item()

        self.dataLabel.setText('rec: 202%s' % self.get_datafolder().split('202')[1])

        # self.delayLabel.setText(' - delay = %.1fs' % self.params['delay'])
        self.repeatLabel.setText(' - Nrepeat = %i' % self.params['Nrepeat'])
        self.periodLabel.setText(' - period (s)= %.1f' % self.params['period'])
        self.freqLabel.setText(' - acq-freq (Hz)= %.1f' % self.params['acq-freq'])

        io = pynwb.NWBHDF5IO(os.path.join(self.get_datafolder(),
                                         'intrinsic-whisker-stim.nwb'), 'r')

        nwbfile = io.read()

        self.t, self.data = nwbfile.acquisition['image_timeseries'].timestamps[:],\
            nwbfile.acquisition['image_timeseries'].data[:,:,:]
        self.pData = None

        self.Nsamples_per_episode = int(self.data.shape[0]/self.params['Nrepeat'])

        io.close()

        print('- data loaded !')

       
    def traces_update(self):

        for plot in [self.raw_trace, self.mean_trace, self.spectrum_power, self.spectrum_phase]:
            plot.clear()

        # pixel position 
        xpix, ypix = self.get_pixel_value()

        if self.pData is None:
            new_data = self.data[:,xpix, ypix]
        else:
            new_data = self.pData[:,xpix, ypix]

        self.raw_trace.plot(self.t, new_data, pen='r')

        spectrum = np.fft.fft((new_data-new_data.mean())/new_data.mean())

        if self.twoPiBox.isChecked():
            power, phase = np.abs(spectrum), -np.angle(spectrum)%(2.*np.pi)
        else:
            power, phase = np.abs(spectrum), np.angle(spectrum)

        x = np.arange(len(power))
        # self.spectrum_power.plot(x[1:], power[1:])
        # self.spectrum_power.plot([x[int(self.params['Nrepeat'])]],
                                 # [power[int(self.params['Nrepeat'])]],
                                 # size=10, symbolPen='g',
                                 # symbol='o')
        self.spectrum_power.plot(np.log10(x[1:int(len(x)/2)]), np.log10(power[1:int(len(x)/2)]))
        self.spectrum_power.plot([np.log10(x[int(self.params['Nrepeat'])])],
                                 [np.log10(power[int(self.params['Nrepeat'])])],
                                 size=10, symbolPen='g',
                                 symbol='o')

        self.spectrum_phase.plot(np.log10(x[1:int(len(x)/2)]), phase[1:int(len(x)/2)])
        self.spectrum_phase.plot([np.log10(x[int(self.params['Nrepeat'])])],
                                 [phase[int(self.params['Nrepeat'])]],
                                 size=10, symbolPen='g',
                                 symbol='o')

        self.mean_trace.plot(np.arange(self.Nsamples_per_episode)/self.params['acq-freq'],
                new_data.reshape(self.params['Nrepeat'], self.Nsamples_per_episode).mean(axis=0), pen='b')


    def show_raw_data(self, with_raw_img=False):
        
        if self.data is None:
            self.load_data()

        if self.pData is None:
            self.img1.setImage(self.data[0, :, :])
        else:
            self.img1.setImage(self.pData[0, :, :])

        self.traces_update()

    def hitting_space(self):
        self.show_raw_data()

    def process(self):
        self.compute_maps()
        
    def compute_avgs(self):

        # clear previous plots
        for plot in [self.spectrum_power, self.spectrum_phase]:
            plot.clear()

        self.pData = analysis.preprocess_data(self.data,
                                              temporal_smoothing=float(self.temporalSmoothingBox.text()),
                                              spatial_smoothing=int(self.spatialSmoothingBox.text()),
                                              high_pass_filtering=float(self.hpBox.text()))

        std_map = self.pData.reshape(self.params['Nrepeat'], self.Nsamples_per_episode,
                        self.pData.shape[1], self.pData.shape[2]).mean(axis=0).std(axis=0)
        if self.stdNormBox.isChecked():
            self.img1.setImage(std_map/self.pData.std(axis=0))
        else:
            self.img1.setImage(std_map)


    def compute_maps(self):

        print('computing FFT [...]')
        self.pData = analysis.preprocess_data(self.data,
                                              temporal_smoothing=float(self.temporalSmoothingBox.text()),
                                              spatial_smoothing=int(self.spatialSmoothingBox.text()),
                                              high_pass_filtering=float(self.hpBox.text()))

        power_map, phase_map = analysis.perform_fft_analysis(self.pData, self.params['Nrepeat'],
                                                             zero_two_pi_convention=self.twoPiBox.isChecked())

        print('Done !')
        if self.stdNormBox.isChecked():
            self.img1.setImage(power_map/self.pData.std(axis=0))
        else:
            self.img1.setImage(power_map)

       

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

        self.load_data()

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
