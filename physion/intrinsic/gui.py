import sys, os, shutil, glob, time, subprocess, pathlib, json, tempfile, datetime
import numpy as np
import pynwb
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
from scipy.interpolate import interp1d
from skimage import measure
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
                 spatial_subsampling=1,
                 time_subsampling=1):
        """
        Intrinsic Imaging GUI
        """
        self.app = app
        
        super(MainWindow, self).__init__(i=1,
                                         title='intrinsic imaging')

        ###
        try:
            bridge = Bridge()
            core = bridge.get_core()
            self.exposure = core.get_exposure()
            self.init_camera()
            self.demo = False
        except BaseException as be:
            print(be)
            print('')
            print(' /!\ Problem with the Camera /!\ ')
            print('        --> no camera found ')
            print('')
            self.exposure = -1 # flag for no camera
            self.demo = True
        

        # some initialisation
        self.running, self.stim = False, None
        self.datafolder = ''        
        
        ########################
        ##### building GUI #####
        ########################
        
        self.minView = False
        self.showwindow()
        # central widget
        self.cwidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.cwidget)

        # layout
        Nx_wdgt, Ny_wdgt, self.wdgt_length = 20, 20, 3
        self.i_wdgt = 0
        self.l0 = QtWidgets.QGridLayout()
        self.cwidget.setLayout(self.l0)
        
        self.win = pg.GraphicsLayoutWidget()
        self.l0.addWidget(self.win,0, self.wdgt_length,
                          Nx_wdgt, Ny_wdgt-self.wdgt_length)
        
        # -- A plot area (ViewBox + axes) for displaying the image ---
        self.view = self.win.addViewBox(lockAspect=True, invertY=True)
        self.view.setMenuEnabled(False)
        self.view.setAspectLocked()
        self.pimg = pg.ImageItem()
        self.view.addItem(self.pimg)
        self.pimg.setImage(self.get_frame())
        self.view.autoRange(padding=0.001)
        
        # ---  setting subject information ---
        self.add_widget(QtWidgets.QLabel('subjects file:'))
        self.subjectFileBox = QtWidgets.QComboBox(self)
        self.subjectFileBox.addItems([f for f in os.listdir(subjects_path) if f.endswith('.json')])
        self.subjectFileBox.activated.connect(self.get_subject_list)
        self.add_widget(self.subjectFileBox)

        self.add_widget(QtWidgets.QLabel('subject:'))
        self.subjectBox = QtWidgets.QComboBox(self)
        self.get_subject_list()
        self.add_widget(self.subjectBox)

        self.add_widget(QtWidgets.QLabel(20*' - '))
        
        # ---  data acquisition properties ---
        self.add_widget(QtWidgets.QLabel('data folder:'), spec='small-left')
        self.folderB = QtWidgets.QComboBox(self)
        self.folderB.addItems(FOLDERS.keys())
        self.add_widget(self.folderB, spec='large-right')

        self.add_widget(QtWidgets.QLabel('  - exposure (ms):'),
                        spec='large-left')
        self.exposureBox = QtWidgets.QLineEdit()
        self.exposureBox.setText(str(self.exposure))
        self.add_widget(self.exposureBox, spec='small-right')

        self.add_widget(QtWidgets.QLabel('  - speed (degree/s):'),
                        spec='large-left')
        self.speedBox = QtWidgets.QLineEdit()
        self.speedBox.setText('10')
        self.add_widget(self.speedBox, spec='small-right')

        self.add_widget(QtWidgets.QLabel('  - bar size (degree):'),
                        spec='large-left')
        self.barBox = QtWidgets.QLineEdit()
        self.barBox.setText('4')
        self.add_widget(self.barBox, spec='small-right')

        self.add_widget(QtWidgets.QLabel('  - spatial sub-sampling (px):'),
                        spec='large-left')
        self.spatialBox = QtWidgets.QLineEdit()
        self.spatialBox.setText('1')
        self.add_widget(self.spatialBox, spec='small-right')

        self.add_widget(QtWidgets.QLabel('  - acq. freq. (Hz):'),
                        spec='large-left')
        self.freqBox = QtWidgets.QLineEdit()
        self.freqBox.setText('10')
        self.add_widget(self.freqBox, spec='small-right')

        self.add_widget(QtWidgets.QLabel('  - flick. freq. (Hz):'),
                        spec='large-left')
        self.flickBox = QtWidgets.QLineEdit()
        self.flickBox.setText('40')
        self.add_widget(self.flickBox, spec='small-right')
        
        self.demoBox = QtWidgets.QCheckBox("demo mode")
        self.demoBox.setStyleSheet("color: gray;")
        self.add_widget(self.demoBox, spec='large-right')
        self.demoBox.setChecked(self.demo)
        
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
        
        self.folderButton = QtWidgets.QPushButton("load data [Ctrl+O]", self)
        self.folderButton.clicked.connect(self.open_file)
        self.add_widget(self.folderButton, spec='large-left')
        self.lastBox = QtWidgets.QCheckBox("last ")
        self.lastBox.setStyleSheet("color: gray;")
        self.add_widget(self.lastBox, spec='small-right')
        self.lastBox.setChecked(True)

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
        self.analysisButton = QtWidgets.QPushButton("- RUN ANALYSIS - ", self)
        self.analysisButton.clicked.connect(self.launch_analysis)
        self.add_widget(self.analysisButton)
        
        self.add_widget(QtWidgets.QLabel(20*' - '))

    def add_widget(self, wdgt, spec='None'):
        if 'small' in spec:
            wdgt.setFixedWidth(70)
            
        if spec=='small-left':
            self.l0.addWidget(wdgt, self.i_wdgt, 0, 1, 1)
        elif spec=='large-left':
            self.l0.addWidget(wdgt, self.i_wdgt, 0, 1, self.wdgt_length-1)
        elif spec=='small-right':
            self.l0.addWidget(wdgt, self.i_wdgt, self.wdgt_length-1, 1, 1)
            self.i_wdgt += 1
        elif spec=='large-right':
            self.l0.addWidget(wdgt, self.i_wdgt, 1, 1, self.wdgt_length-1)
            self.i_wdgt += 1
        else:
            self.l0.addWidget(wdgt, self.i_wdgt, 0, 1, self.wdgt_length)
            self.i_wdgt += 1
        
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

    def init_camera(self):
        self.bridge = Bridge()
        self.core = self.bridge.get_core()
        self.core.set_exposure(int(self.exposureBox.text()))
        # SHUTTER PROPS ???
        # auto_shutter = self.core.get_property('Core', 'AutoShutter')
        # self.core.set_property('Core', 'AutoShutter', 0)

    def get_patterns(self, direction, angle, size,
                     Npatch=25):

        patterns = []
        
        if direction=='horizontal':
            x = self.stim.angle_to_pix(np.linspace(self.STIM['zmin'], self.STIM['zmax'], Npatch))
            # for i in np.random.choice(np.arange(Npatch-1), int(Npatch/2)+1):
            for i in np.arange(Npatch-1)[(1if self.flip else 0)::2]:
                patterns.append(visual.Rect(win=self.stim.win,
                                            size=(self.stim.angle_to_pix(size),
                                                  self.stim.angle_to_pix(np.abs(x[i+1]-x[i]), starting_angle=np.abs(x[i]))),
                                            pos=(self.stim.angle_to_pix(angle), self.stim.angle_to_pix(x[i])),
                                            units='pix', fillColor=1, color=-1))
        elif direction=='vertical':
            x = np.linspace(self.STIM['xmin'], self.STIM['xmax'], Npatch)
            # for i in np.random.choice(np.arange(Npatch-1), int(Npatch/2)+1):
            for i in np.arange(Npatch-1)[(1 if self.flip else 0)::2]:
                patterns.append(visual.Rect(win=self.stim.win,
                                            size=(self.stim.angle_to_pix(np.abs(x[i+1]-x[i]), starting_angle=np.abs(x[i])),
                                                  self.stim.angle_to_pix(size)),
                                            pos=(self.stim.angle_to_pix(x[i]), self.stim.angle_to_pix(angle)),
                                            units='pix', fillColor=1, color=-1))

        self.flip = (False if self.flip else True) # flip the flag
            
        return patterns

    def resample_img(self, img, Nsubsampling):
        if Nsubsampling>1:
            return measure.block_reduce(img, block_size=(Nsubsampling,
                                                         Nsubsampling), func=np.mean)
        else:
            return img
        
    def run(self):

        self.flip = False
        
        self.stim = visual_stim({"Screen": "Dell-2020",
                                 "presentation-prestim-screen": -1,
                                 "presentation-poststim-screen": -1}, demo=self.demoBox.isChecked())
        
        self.speed = float(self.speedBox.text()) # degree / second
        self.bar_size = float(self.barBox.text()) # degree / second
        self.dt_save, self.dt = 1/float(self.freqBox.text()), 1/float(self.flickBox.text())
        
        xmin, xmax = 1.2*np.min(self.stim.x), 1.2*np.max(self.stim.x)
        zmin, zmax = 1.2*np.min(self.stim.z), 1.2*np.max(self.stim.z)

        self.angle_start, self.angle_max, self.direction, self.label = 0, 0, '', ''

        self.STIM = {'angle_start':[zmin, xmax, zmax, xmin],
                     'angle_stop':[zmax, xmin, zmin, xmax],
                     'direction':['vertical', 'horizontal', 'vertical', 'horizontal'],
                     'label':['up', 'left', 'down', 'right'],
                     'xmin':xmin, 'xmax':xmax, 'zmin':zmin, 'zmax':zmax}

        for il, label in enumerate(self.STIM['label']):
            tmax = np.abs(self.STIM['angle_stop'][il]-self.STIM['angle_start'][il])/self.speed
            self.STIM[label+'-times'] = np.arange(int(tmax/self.dt))*self.dt
            self.STIM[label+'-angle'] = np.linspace(self.STIM['angle_start'][il],
                                                    self.STIM['angle_stop'][il],
                                                    int(tmax/self.dt))
  
        self.iEp, self.iTime, self.tstart, self.label = 0, 0, time.time(), 'up'
        self.tSave, self.img, self.nSave = time.time(), np.zeros(self.imgsize), 0
        
        self.update_dt() # while loop

        if self.exposure>0: # at the end we close the camera
            self.bridge.close()
        

    def save_img(self):
        
        if self.nSave>0:
            self.img /= self.nSave

        if True: # live display
            self.pimg.setImage(self.img)

        self.iTime += 1
        # NEED TO STORE DATA HERE
        self.FRAMES.append(self.img)

        # re-init time step of acquisition
        self.tSave, self.img, self.nSave = time.time(), np.zeros(self.imgsize), 0
        
    def update_dt(self):
        
        t0 = time.time()
        while (time.time()-t0)<=self.dt:

            # show image
            patterns = self.get_patterns(self.STIM['direction'][self.iEp%4],
                                         self.STIM[self.STIM['label'][self.iEp%4]+'-angle'][self.iTime],
                                         self.bar_size)
            for pattern in patterns:
                pattern.draw()
            try:
                self.stim.win.flip()
            except BaseException:
                pass

            # fetch image
            self.img += self.resample_img(self.get_frame(),
                                          int(self.spatialBox.text()))
            self.nSave+=1

            
        # continuing ?
        if self.running:

            # saving frame data
            if (time.time()-self.tSave)<=self.dt_save:
                self.save_img()
            
            # checking if not episode over
            if not (self.iTime<len(self.STIM[self.STIM['label'][self.iEp%4]+'-angle'])):
                self.write_data() # writing data when over
                self.FRAMES = [] # re init data
                self.iTime = 0  
                self.iEp += 1
                
            QtCore.QTimer.singleShot(1, self.update_dt)


    def write_data(self):

        filename = '%s-%i.nwb' % (self.STIM['label'][self.iEp%4], int(self.iEp/4)+1)
        
        nwbfile = pynwb.NWBFile('Intrinsic Imaging data following bar stimulation',
                                'intrinsic',
                                datetime.datetime.utcnow(),
                                file_create_date=datetime.datetime.utcnow())

        # Create our time series
        angles = pynwb.TimeSeries(name='angle_timeseries',
                                  data=self.STIM[self.STIM['label'][self.iEp%4]+'-angle'],
                                  unit='Rd',
                                  timestamps=self.STIM[self.STIM['label'][self.iEp%4]+'-times'])
        nwbfile.add_acquisition(angles)

        images = pynwb.image.ImageSeries(name='image_timeseries',
                                         data=np.array(self.FRAMES),
                                         unit='a.u.',
                                         timestamps=self.STIM[self.STIM['label'][self.iEp%4]+'-times'])

        nwbfile.add_acquisition(images)
        
        
        # Write the data to file
        io = pynwb.NWBHDF5IO(os.path.join(self.datafolder, filename), 'w')
        print('writing:', filename)
        io.write(nwbfile)
        io.close()
        print(filename, ' saved !')
        
        # filename = '%s-%i.npy' % (self.STIM['label'][self.iEp%4], int(self.iEp/4)+1))
        # data = {'times':self.TIMES,
        #         'angles':self.ANGLES,
        #         'frames':self.FRAMES}
        # np.save(os.path.join(self.datafolder, filename), data)
        
        
    def launch_protocol(self):

        if not self.running:
            self.running = True

            # initialization of data
            self.FRAMES = []
            self.imgsize = self.resample_img(self.get_frame(),
                                             int(self.spatialBox.text())).shape
            self.pimg.setImage(self.resample_img(self.get_frame(),
                                                 int(self.spatialBox.text())))
            self.view.autoRange(padding=0.001)
            
            # init
            filename = generate_filename_path(FOLDERS[self.folderB.currentText()],
                                              filename='metadata', extension='.npy')
            metadata = {'subject':str(self.subjectBox.currentText()),
                        'exposure':float(self.exposureBox.text()),
                        'bar-size':float(self.barBox.text()),
                        'acq-freq':float(self.freqBox.text()),
                        'speed':float(self.speedBox.text())}
            np.save(filename, metadata)
            self.datafolder = os.path.dirname(filename)

            print('acquisition running [...]')
            self.run()
            
        else:
            print(' /!\  --> pb in launching acquisition (either already running or missing camera)')

    def live_view(self):
        self.running = True
        self.update_Image()
        
    def stop_protocol(self):
        if self.running:
            self.running = False
            if self.stim is not None:
                self.stim.close()
        else:
            print('acquisition not launched')

    def get_frame(self):
        if self.exposure>0:
            self.core.snap_image()
            tagged_image = self.core.get_tagged_image()
            #pixels by default come out as a 1D array. We can reshape them into an image
            return np.reshape(tagged_image.pix,
                              newshape=[tagged_image.tags['Height'], tagged_image.tags['Width']])
        else:
            return np.random.randn(720, 1080)
        
    def update_Image(self):
        # plot it
        self.pimg.setImage(self.get_frame())
        if self.running:
            QtCore.QTimer.singleShot(1, self.update_Image)
        elif self.exposure>0: # at the end
            self.bridge.close()

    def hitting_space(self):
        if not self.running:
            self.launch_protocol()
        else:
            self.stop_protocol()

    def launch_analysis(self):
        print('launching analysis [...]')
        if self.datafolder=='' and self.lastBox.isChecked():
            self.datafolder = last_datafolder_in_dayfolder(day_folder(os.path.join(FOLDERS[self.folderB.currentText()])),
                                                           with_NIdaq=False)
        analysis.run(self.datafolder)
        print('-> analysis done !')

    def process(self):
        self.launch_analysis()
    

    def open_file(self):

        self.lastBox.setChecked(False)
        folder = QtWidgets.QFileDialog.getExistingDirectory(self,\
                                                            "Choose datafolder",
                                                            FOLDERS[self.folderB.currentText()])

        if folder!='':
            self.datafolder = folder
        else:
            print('data-folder not set !')
        
        
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
