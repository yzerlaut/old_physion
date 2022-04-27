import sys, time, tempfile, os, pathlib, datetime, string, pynwb, subprocess
import numpy as np
from PyQt5 import QtGui, QtWidgets, QtCore
import pyqtgraph as pg

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import day_folder, generate_filename_path, list_dayfolder, get_files_with_extension
from assembling.dataset import Dataset, MODALITIES
from dataviz import plots
from analysis.trial_averaging import TrialAverageWindow
from analysis.read_NWB import Data
from analysis.summary_pdf import summary_pdf_folder
from misc.folders import FOLDERS, python_path
from misc import guiparts
from visual_stim.stimuli import build_stim # we'll load it without psychopy

try:
    from analysis.make_figures import FiguresWindow
except ModuleNotFoundError:
    print('--------------------------------------------')
    print('  "datavyz" submodule not found')
    print('  -> install with "pip install ./physion/dataviz/datavyz/."')
    print('             (after a "git submodule init; git submodule update" if not already done) ')

settings = {
    'window_size':(1000,600),
    # raw data plot settings
    'increase-factor':1.5, # so "Calcium" is twice "Eletrophy", that is twice "Pupil",..  "Locomotion"
    'blank-space':0.1, # so "Calcium" is twice "Eletrophy", that is twice "Pupil",..  "Locomotion"
    'colors':{'Screen':(100, 100, 100, 255),#'grey',
              'Locomotion':(255,255,255,255),#'white',
              'FaceMotion':(255,0,255,255),#'purple',
              'Pupil':(255,0,0,255),#'red',
              'Gaze':(200,100,0,255),#'orange',
              'Electrophy':(100,100,255,255),#'blue',
              'LFP':(100,100,255,255),#'blue',
              'Vm':(100,100,100,255),#'blue',
              'CaImaging':(0,255,0,255)},#'green'},
    # general settings
    'Npoints':500}

class MainWindow(guiparts.NewWindow):
    
    def __init__(self, app,
                 args=None,
                 parent=None,
                 raw_data_visualization=False,
                 df_width = 600,
                 selector_height = 30,
                 win1_Wmax=1200, win1_Wmin=300,
                 win1_Hmax=300, win2_Wmax=500,
                 fullscreen=False):

        self.app = app
        self.settings = settings
        self.raw_data_visualization = raw_data_visualization
        self.no_subsampling = False
        
        super(MainWindow, self).__init__(i=0,
            title='Data Visualization')

        # play button
        self.updateTimer = QtCore.QTimer()
        self.updateTimer.timeout.connect(self.next_frame)
        
        self.cwidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.cwidget)

        self.statusBar.showMessage('open file [Ctrl+O],    refresh plot [Ctrl+R],    play/pause [Ctrl+Space],    initial-view [Ctrl-I],    max-window [Ctrl+M] ' )

        mainLayout = QtWidgets.QVBoxLayout()

        Layout1 = QtWidgets.QHBoxLayout()
        mainLayout.addLayout(Layout1)

        Layout11 = QtWidgets.QVBoxLayout()
        Layout1.addLayout(Layout11)
        guiparts.create_calendar(self, Layout11)
        self.cal.setMaximumHeight(150)

        # folder box
        self.fbox = QtWidgets.QComboBox(self)
        self.fbox.setFont(guiparts.smallfont)
        self.fbox.activated.connect(self.scan_folder)
        self.fbox.setMaximumHeight(selector_height)
        self.folder_default_key = '  [root datafolder]'
        self.fbox.addItem(self.folder_default_key)
        self.fbox.setCurrentIndex(0)
        Layout11.addWidget(self.fbox)

        # subject box
        self.sbox = QtWidgets.QComboBox(self)
        self.sbox.setFont(guiparts.smallfont)
        self.sbox.activated.connect(self.pick_subject) # To be written !!
        self.sbox.setMaximumHeight(selector_height)
        self.subject_default_key = '  [subject] '
        self.sbox.addItem(self.subject_default_key)

        self.sbox.setCurrentIndex(0)
        Layout11.addWidget(self.sbox)

        # notes
        self.notes = QtWidgets.QLabel('\n[exp info]'+5*'\n', self)
        self.notes.setFont(guiparts.smallfont)
        self.notes.setMaximumHeight(60)
        Layout11.addWidget(self.notes)

        self.pbox = QtWidgets.QComboBox(self)
        self.pbox.setFont(guiparts.smallfont)
        self.pbox.activated.connect(self.display_quantities)
        self.pbox.setMaximumHeight(selector_height)
        self.pbox.addItem('[visualization/analysis]')
        self.pbox.addItem('-> Show Raw Data')
        self.pbox.addItem('-> Trial-average')
        self.pbox.addItem('-> draw figures')
        self.pbox.addItem('-> build PDF summary')
        self.pbox.addItem('-> open PDF summary')
        self.pbox.setCurrentIndex(0)

        Layout11.addWidget(self.pbox)

        Layout113 = QtWidgets.QHBoxLayout()
        Layout11.addLayout(Layout113)

        add_buttons(self, Layout113)

        Layout12 = QtWidgets.QVBoxLayout()
        Layout1.addLayout(Layout12)

        self.dbox = QtWidgets.QComboBox(self)
        self.dbox.setMinimumWidth(df_width)
        self.dbox.setMaximumWidth(win1_Wmax)
        self.dbox.setMinimumHeight(selector_height)
        self.dbox.setMaximumHeight(selector_height)
        self.dbox.activated.connect(self.pick_datafile)
        Layout12.addWidget(self.dbox)

        self.win1 = pg.GraphicsLayoutWidget()
        self.win1.setMaximumHeight(int(win1_Hmax-1.5*selector_height))
        Layout12.addWidget(self.win1)

        self.winTrace = pg.GraphicsLayoutWidget()
        mainLayout.addWidget(self.winTrace)

        guiparts.build_slider(self, mainLayout)

        self.init_panels()
        
        self.plot = self.winTrace.addPlot()
        self.plot.hideAxis('left')
        self.plot.setMouseEnabled(x=True,y=False)
        # self.plot.setMenuEnabled(False)
        self.plot.setLabel('bottom', 'time (s)')
        self.xaxis = self.plot.getAxis('bottom')
        self.scatter = pg.ScatterPlotItem()
        self.plot.addItem(self.scatter)


        Layout122 = QtWidgets.QHBoxLayout()
        Layout12.addLayout(Layout122)

        self.stimSelect = QtWidgets.QCheckBox("vis. stim")
        self.stimSelect.clicked.connect(self.select_stim)
        self.stimSelect.setStyleSheet('color: grey;')

        self.pupilSelect = QtWidgets.QCheckBox("pupil")
        self.pupilSelect.setStyleSheet('color: red;')

        self.gazeSelect = QtWidgets.QCheckBox("gaze")
        self.gazeSelect.setStyleSheet('color: orange;')

        self.faceMtnSelect = QtWidgets.QCheckBox("whisk.")
        self.faceMtnSelect.setStyleSheet('color: magenta;')

        self.runSelect = QtWidgets.QCheckBox("run")
        
        self.photodiodeSelect = QtWidgets.QCheckBox("photodiode")
        self.photodiodeSelect.setStyleSheet('color: grey;')

        self.ephysSelect = QtWidgets.QCheckBox("ephys")
        self.ephysSelect.setStyleSheet('color: blue;')
        
        self.ophysSelect = QtWidgets.QCheckBox("ophys")
        self.ophysSelect.setStyleSheet('color: green;')

        for x in [self.stimSelect, self.pupilSelect,
                  self.gazeSelect, self.faceMtnSelect,
                  self.runSelect,self.photodiodeSelect,
                  self.ephysSelect, self.ophysSelect]:
            x.setFont(guiparts.smallfont)
            Layout122.addWidget(x)
        
        
        self.roiPick = QtWidgets.QLineEdit()
        self.roiPick.setText(' [...] ')
        self.roiPick.setMinimumWidth(50)
        self.roiPick.setMaximumWidth(250)
        self.roiPick.returnPressed.connect(self.select_ROI)
        self.roiPick.setFont(guiparts.smallfont)

        self.ephysPick = QtWidgets.QLineEdit()
        self.ephysPick.setText(' ')
        # self.ephysPick.returnPressed.connect(self.select_ROI)
        self.ephysPick.setFont(guiparts.smallfont)

        self.guiKeywords = QtWidgets.QLineEdit()
        self.guiKeywords.setText('     [GUI keywords] ')
        # self.guiKeywords.setFixedWidth(200)
        self.guiKeywords.returnPressed.connect(self.keyword_update)
        self.guiKeywords.setFont(guiparts.smallfont)

        Layout122.addWidget(self.guiKeywords)
        # Layout122.addWidget(self.ephysPick)
        Layout122.addWidget(self.roiPick)

        self.subsamplingSelect = QtWidgets.QCheckBox("subsampl.")
        self.subsamplingSelect.setStyleSheet('color: grey;')
        self.subsamplingSelect.setFont(guiparts.smallfont)
        Layout122.addWidget(self.subsamplingSelect)

        self.annotSelect = QtWidgets.QCheckBox("annot.")
        self.annotSelect.setStyleSheet('color: grey;')
        self.annotSelect.setFont(guiparts.smallfont)
        Layout122.addWidget(self.annotSelect)
        
        self.imgSelect = QtWidgets.QCheckBox("img")
        self.imgSelect.setStyleSheet('color: grey;')
        self.imgSelect.setFont(guiparts.smallfont)
        self.imgSelect.setChecked(True)
        self.imgSelect.clicked.connect(self.remove_img)
        Layout122.addWidget(self.imgSelect)
        
        self.cwidget.setLayout(mainLayout)
        self.show()
        
        self.fbox.addItems(FOLDERS.keys())
        self.windowTA, self.windowBM = None, None # sub-windows

        if args is not None:
            self.root_datafolder = args.root_datafolder
        else:
            self.root_datafolder = os.path.join(os.path.expanduser('~'), 'DATA')

        self.time, self.data, self.roiIndices, self.tzoom = 0, None, [], [0,50]
        self.CaImaging_bg_key, self.planeID = 'meanImg', 0
        self.CaImaging_key = 'Fluorescence'

        self.FILES_PER_DAY, self.FILES_PER_SUBJECT, self.SUBJECTS = {}, {}, {}

        self.minView = False
        self.showwindow()

        if (args is not None) and hasattr(args, 'datafile') and os.path.isfile(args.datafile):
            self.datafile=args.datafile
            self.load_file(self.datafile)
            plots.raw_data_plot(self, self.tzoom)
            

    def init_panel_imgs(self):
        
        self.pScreenimg.setImage(np.ones((10,12))*50)
        self.pFaceimg.setImage(np.ones((10,12))*50)
        self.pPupilimg.setImage(np.ones((10,12))*50)
        self.pFacemotionimg.setImage(np.ones((10,12))*50)
        self.pCaimg.setImage(np.ones((50,50))*100)
        self.pupilContour.setData([0], [0], size=1, brush=pg.mkBrush(0,0,0))
        self.faceMotionContour.setData([0], [0], size=2,
                    brush=pg.mkBrush(*settings['colors']['FaceMotion'][:3]))
        self.facePupilContour.setData([0], [0], size=2,
                    brush=pg.mkBrush(*settings['colors']['Pupil'][:3]))

    def remove_img(self):
        if not self.imgSelect.isChecked():
            self.init_panel_imgs()
            
    def init_panels(self):

        # screen panel
        self.pScreen = self.win1.addViewBox(lockAspect=True, invertY=False, border=[1, 1, 1], colspan=2)
        self.pScreenimg = pg.ImageItem(np.ones((10,12))*50)
        # FaceCamera panel
        self.pFace = self.win1.addViewBox(lockAspect=True, invertY=True, border=[1, 1, 1], colspan=2)
        self.faceMotionContour = pg.ScatterPlotItem()
        self.facePupilContour = pg.ScatterPlotItem()
        self.pFaceimg = pg.ImageItem(np.ones((10,12))*50)
        # Pupil panel
        self.pPupil=self.win1.addViewBox(lockAspect=True, invertY=True, border=[1, 1, 1])
        self.pupilContour = pg.ScatterPlotItem()
        self.pPupilimg = pg.ImageItem(np.ones((10,12))*50)
        # Facemotion panel
        self.pFacemotion=self.win1.addViewBox(lockAspect=True, invertY=True, border=[1, 1, 1])
        self.facemotionROI = pg.ScatterPlotItem()
        self.pFacemotionimg = pg.ImageItem(np.ones((10,12))*50)
        # Ca-Imaging panel
        self.pCa=self.win1.addViewBox(lockAspect=True,invertY=True, border=[1, 1, 1])
        self.pCaimg = pg.ImageItem(np.ones((50,50))*100)
        
        for x, y in zip([self.pScreen, self.pFace,self.pPupil,self.pPupil,self.pFacemotion,self.pFacemotion,self.pCa, self.pFace, self.pFace],
                        [self.pScreenimg, self.pFaceimg, self.pPupilimg, self.pupilContour, self.pFacemotionimg, self.facemotionROI, self.pCaimg, self.faceMotionContour, self.facePupilContour]):
            x.addItem(y)
            

    def open_file(self):

        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self,
                     "Open Multimodal Experimental Recording (NWB file) ",
                    (FOLDERS[self.fbox.currentText()] if self.fbox.currentText() in FOLDERS else os.path.join(os.path.expanduser('~'), 'DATA')),
                                                        filter="*.nwb")
        # filename = '/home/yann/UNPROCESSED/2021_06_17/2021_06_17-12-57-44.nwb'
        # filename = '/home/yann/DATA/CaImaging/NDNFcre_GCamp6s/Batch-2_September_2021/2021_09_10/2021_09_10-14-55-23.nwb'
        
        if filename!='':
            self.datafile=filename
            self.load_file(self.datafile)
            plots.raw_data_plot(self, self.tzoom)
        else:
            print('"%s" filename not loaded/recognized ! ' % filename)

    def save(self):
        pass

    def reset(self):
        self.windowTA, self.windowBM = None, None # sub-windows
        self.notes.clear()
        self.subsamplingSelect.setChecked(True)
        self.annotSelect.setChecked(False)
        self.stimSelect.setChecked(False)
        self.init_panel_imgs()
        self.roiIndices = None

    def select_stim(self):
        if self.stimSelect.isChecked():
            if self.data.visual_stim is None:
                self.load_VisualStim()
            self.keyword_update(string='stim')
        else:
            self.keyword_update(string='no_stim')
            self.data.visual_stim = None
        
    def select_ROI(self):
        """ see select ROI above """
        self.roiIndices = self.select_ROI_from_pick(self.data)
        plots.raw_data_plot(self, self.tzoom, with_roi=True)

            
    def load_file(self, filename):
        """ should be a minimal processing so that the loading is fast"""
        self.reset()

        self.data = Data(filename)
            
        self.tzoom = self.data.tlim
        self.notes.setText(self.data.description)

        self.cal.setSelectedDate(self.data.nwbfile.session_start_time.date())
        if self.dbox.currentIndex()<1:
            self.dbox.clear()
            self.dbox.addItem(self.data.df_name)
            self.dbox.setCurrentIndex(0)
            
        if len(self.SUBJECTS.keys())==0:
            self.sbox.clear()
            self.sbox.addItem(self.subject_default_key)
            self.sbox.addItem(self.data.nwbfile.subject.description)
            self.sbox.setCurrentIndex(1)
            
        self.pbox.setCurrentIndex(1)

        if 'ophys' in self.data.nwbfile.processing:
            self.roiPick.setText(' [select ROI: %i-%i]' % (0, len(self.data.valid_roiIndices)-1))
            self.ophysSelect.setChecked(True)

        if ('Electrophysiological-Signal' in self.data.nwbfile.acquisition) or ('Vm' in self.data.nwbfile.acquisition) or ('LFP' in self.data.nwbfile.acquisition):
            self.ephysSelect.setChecked(True)
            
        if 'Photodiode-Signal' in self.data.nwbfile.acquisition:
            self.photodiodeSelect.setChecked(True)

        if 'Running-Speed' in self.data.nwbfile.acquisition:
            self.runSelect.setChecked(True)
            self.runSelect.isChecked()

        if 'FaceMotion' in self.data.nwbfile.processing:
            coords = self.data.nwbfile.processing['FaceMotion'].description.split('facemotion ROI: (x0,dx,y0,dy)=(')[1].split(')\n')[0].split(',')
            coords = [int(c) for c in coords]
            self.faceMotionContour.setData(np.concatenate([np.linspace(x1, x2, 20) for x1, x2 in zip([coords[1], coords[1], coords[1]+coords[3], coords[1]+coords[3], coords[1]],
                                                                                                     [coords[1], coords[1]+coords[3], coords[1]+coords[3], coords[1], coords[1]])]),
                                           np.concatenate([np.linspace(y1, y2, 20) for y1, y2 in zip([coords[0], coords[0]+coords[2], coords[0]+coords[2], coords[0], coords[0]],
                                                                                                     [coords[0]+coords[2], coords[0]+coords[2], coords[0], coords[0], coords[0]])]))
            self.faceMtnSelect.setChecked(True)
            
        if 'Pupil' in self.data.nwbfile.processing:
            self.pupil_mm_to_pix = 1./float(self.data.nwbfile.processing['Pupil'].description.split('pix_to_mm=')[1].split('\n')[0])
            coords = self.data.nwbfile.processing['Pupil'].description.split('pupil ROI: (xmin,xmax,ymin,ymax)=(')[1].split(')\n')[0].split(',')
            if len(coords)==3: # bug (fixed), typo in previous datafiles
                coords.append(coords[2][3:])
                coords[2] = coords[2][:3]
            coords = [int(c) for c in coords]
            self.facePupilContour.setData(np.concatenate([np.linspace(x1, x2, 10) for x1, x2 in zip([coords[2], coords[2], coords[3], coords[3]],
                                                                                                    [coords[2], coords[3], coords[3], coords[2]])]),
                                           np.concatenate([np.linspace(y1, y2, 10) for y1, y2 in zip([coords[0], coords[1], coords[1], coords[0]],
                                                                                                     [coords[1], coords[1], coords[0], coords[0]])]))
            self.pupilSelect.setChecked(True)

        if 'Pupil' in self.data.nwbfile.processing:
            self.gaze_center = [np.mean(self.data.nwbfile.processing['Pupil'].data_interfaces['cx'].data[:]),
                                np.mean(self.data.nwbfile.processing['Pupil'].data_interfaces['cy'].data[:])]
            self.gazeSelect.setChecked(True)

            
    def load_VisualStim(self):

        # load visual stimulation
        if self.data.metadata['VisualStim']:
            self.data.init_visual_stim()
        else:
            self.data.visual_stim = None
            print(' /!\ No stimulation in this recording /!\  ')


    def scan_folder(self):

        print('inspecting the folder "%s" [...]' % FOLDERS[self.fbox.currentText()])

        FILES0 = get_files_with_extension(FOLDERS[self.fbox.currentText()],
                                          extension='.nwb', recursive=True)

        TIMES, DATES, FILES = [], [], []
        for f in FILES0:
            Time = f.split(os.path.sep)[-1].replace('.nwb', '').split('-')
            if len(Time)>=4:
                TIMES.append(3600*int(Time[0])+60*int(Time[1])+int(Time[2]))
                DATES.append(f.split(os.path.sep)[-1].split('-')[0])
                FILES.append(f)
                
        TIMES, DATES, FILES = np.array(TIMES), np.array(DATES), np.array(FILES)
        NDATES = np.array([datetime.date(*[int(dd) for dd in date.split('_')]).toordinal() for date in DATES])
        self.FILES_PER_DAY = {}
        
        guiparts.reinit_calendar(self,
                                 min_date= tuple(int(dd) for dd in DATES[np.argmin(NDATES)].split('_')),
                                 max_date= tuple(int(dd) for dd in DATES[np.argmax(NDATES)].split('_')))
        for d in np.unique(DATES):
            try:
                self.cal.setDateTextFormat(QtCore.QDate(datetime.date(*[int(dd) for dd in d.split('_')])),
                                           self.highlight_format)
                day_cond = (DATES==d)
                time_sorted = np.argsort(TIMES[day_cond])
                self.FILES_PER_DAY[d] = [os.path.join(FOLDERS[self.fbox.currentText()], f)\
                                         for f in np.array(FILES)[day_cond][time_sorted]]
            except BaseException as be:
                print(be)
                print('error for date %s' % d)
            
        print(' -> found n=%i datafiles ' % len(FILES))
        

    def compute_subjects(self):

        FILES = get_files_with_extension(FOLDERS[self.fbox.currentText()],
                                         extension='.nwb', recursive=True)
        print(' looping over n=%i datafiles to fetch "subjects" metadata [...]' % len(FILES))
        DATES = np.array([f.split(os.path.sep)[-1].split('-')[0] for f in FILES])

        SUBJECTS, DISPLAY_NAMES, SDATES, NDATES = [], [], [], []
        for fn, date in zip(FILES, DATES):
            infos = self.preload_datafolder(fn)
            SDATES.append(date)
            SUBJECTS.append(infos['subject'])
            DISPLAY_NAMES.append(infos['display_name'])
            NDATES.append(datetime.date(*[int(dd) for dd in date.split('_')]).toordinal())

        self.SUBJECTS = {}
        for s in np.unique(SUBJECTS):
            cond = (np.array(SUBJECTS)==s)
            self.SUBJECTS[s] = {'display_names':np.array(DISPLAY_NAMES)[cond],
                                'datafiles':np.array(FILES)[cond],
                                'dates':np.array(SDATES)[cond],
                                'dates_num':np.array(NDATES)[cond]}

        print(' -> found n=%i subjects ' % len(self.SUBJECTS.keys()))
        self.sbox.clear()
        self.sbox.addItems([self.subject_default_key]+\
                           list(self.SUBJECTS.keys()))
        self.sbox.setCurrentIndex(0)
        
                                
    def pick_subject(self):
        self.plot.clear()
        if self.sbox.currentText()==self.subject_default_key:
            self.compute_subjects()
        elif self.sbox.currentText() in self.SUBJECTS:
            self.FILES_PER_DAY = {} # re-init
            date_min = self.SUBJECTS[self.sbox.currentText()]['dates'][np.argmin(self.SUBJECTS[self.sbox.currentText()]['dates_num'])]
            date_max = self.SUBJECTS[self.sbox.currentText()]['dates'][np.argmax(self.SUBJECTS[self.sbox.currentText()]['dates_num'])]
            guiparts.reinit_calendar(self,
                                     min_date= tuple(int(dd) for dd in date_min.split('_')),
                                     max_date= tuple(int(dd) for dd in date_max.split('_')))
            for d in np.unique(self.SUBJECTS[self.sbox.currentText()]['dates']):
                self.cal.setDateTextFormat(QtCore.QDate(datetime.date(*[int(dd) for dd in d.split('_')])),
                                           self.highlight_format)
                self.FILES_PER_DAY[d] = [f for f in np.array(self.SUBJECTS[self.sbox.currentText()]['datafiles'])[self.SUBJECTS[self.sbox.currentText()]['dates']==d]]
        else:
            print(' /!\ subject not recognized /!\  ')
                                
    def pick_date(self):
        date = self.cal.selectedDate()
        self.day = '%s_%02d_%02d' % (date.year(), date.month(), date.day())
        for i in string.digits:
            self.day = self.day.replace('_%s_' % i, '_0%s_' % i)

        if self.day in self.FILES_PER_DAY:
            self.list_protocol = self.FILES_PER_DAY[self.day]
            self.update_df_names()

    def pick_datafile(self):
        self.plot.clear()
        i = self.dbox.currentIndex()
        if i>0:
            self.datafile=self.list_protocol[i-1]
            self.load_file(self.datafile)
            plots.raw_data_plot(self, self.tzoom)
        else:
            # self.data.metadata = None
            self.notes.setText(20*'-'+5*'\n')
        

    def update_df_names(self):
        self.dbox.clear()
        # self.pbox.clear()
        # self.sbox.clear()
        self.plot.clear()
        self.pScreenimg.setImage(np.ones((10,12))*50)
        self.pFaceimg.setImage(np.ones((10,12))*50)
        self.pPupilimg.setImage(np.ones((10,12))*50)
        self.pCaimg.setImage(np.ones((50,50))*100)
        if len(self.list_protocol)>0:
            self.dbox.addItem(' ...' +70*' '+'(select a data-folder) ')
            for fn in self.list_protocol:
                self.dbox.addItem(self.preload_datafolder(fn)['display_name'])
                
    def preload_datafolder(self, fn):
        data = Data(fn, metadata_only=True, with_tlim=False)
        if data.nwbfile is not None:
            return {'display_name' : data.df_name,
                     'subject': data.nwbfile.subject.description}
        else:
            return {'display_name': '', 'subject':''}

    def add_datafolder_annotation(self):
        info = 20*'-'+'\n'

        if self.data.metadata['protocol']=='None':
            self.notes.setText('\nNo visual stimulation')
        else:
            for key in self.data.metadata:
                if (key[:2]=='N-') and (key!='N-repeat') and (self.data.metadata[key]>1): # meaning it was varied
                    info += '%s=%i (%.1f to %.1f)\n' % (key, self.data.metadata[key], self.data.metadata[key[2:]+'-1'], self.data.metadata[key[2:]+'-2'])
            info += '%s=%i' % ('N-repeat', self.data.metadata['N-repeat'])
            self.notes.setText(info)

    def display_quantities(self,
                           force=False,
                           plot_update=True,
                           with_images=False,
                           with_scatter=False):
        """
        # IMPLEMENT OTHER ANALYSIS HERE
        """

        if self.pbox.currentText()=='-> Trial-average':
            self.windowTA = TrialAverageWindow(parent=self)
            self.windowTA.show()
        elif self.pbox.currentText()=='-> draw figures':
            self.windowFG = FiguresWindow(self.datafile, parent=self)
            self.windowFG.show()
        elif self.pbox.currentText()=='-> build PDF summary':
            cmd = '%s %s %s --verbose' % (python_path,
                            os.path.join(str(pathlib.Path(__file__).resolve().parents[1]),
                                     'analysis', 'summary_pdf.py'), self.datafile)
            p = subprocess.Popen(cmd, shell=True)
            print('"%s" launched as a subprocess' % cmd)
        elif self.pbox.currentText()=='-> open PDF summary':
            pdf_folder = summary_pdf_folder(self.datafile)
            if os.path.isdir(pdf_folder):
                PDFS = os.listdir(pdf_folder)
                for pdf in PDFS:
                    print(' - opening: "%s"' % pdf)
                    os.system('$(basename $(xdg-mime query default application/pdf) .desktop) %s & ' % os.path.join(pdf_folder, pdf))
            else:
                print('no PDF summary files found !')

        else:
            self.plot.clear()
            plots.raw_data_plot(self, self.tzoom,
                                plot_update=plot_update,
                                with_images=with_images,
                                with_scatter=with_scatter)
        self.pbox.setCurrentIndex(1)


    def back_to_initial_view(self):
        self.time = 0
        self.tzoom = self.data.tlim
        self.display_quantities(force=True)

    def hitting_space(self):
        if self.pauseButton.isEnabled():
            self.pause()
        else:
            self.play()

    def launch_movie(self,
                     movie_refresh_time=0.2, # all in seconds
                     time_for_full_window=25):

        self.tzoom = self.plot.getAxis('bottom').range
        self.view = (self.tzoom[1]-self.tzoom[0])/2.*np.array([-1,1])
        # forcing time at 10% of the window
        self.time = self.tzoom[0]+.1*(self.tzoom[1]-self.tzoom[0])
        self.display_quantities()
        # then preparing the update rule
        N_updates = int(time_for_full_window/movie_refresh_time)+1
        self.dt_movie = (self.tzoom[1]-self.tzoom[0])/N_updates
        self.updateTimer.start(int(1e3*movie_refresh_time))

    def play(self):
        print('playing')
        self.pauseButton.setEnabled(True)
        self.playButton.setEnabled(False)
        self.playButton.setChecked(True)
        self.launch_movie()

    def next_frame(self):
        print(self.time, self.dt_movie)
        self.time = self.time+self.dt_movie
        self.prev_time = self.time
        if self.time>self.tzoom[0]+.8*(self.tzoom[1]-self.tzoom[0]):
            print('block')
            self.tzoom = [self.view[0]+self.time, self.view[1]+self.time]
        if self.time==self.prev_time:
            self.time = self.time+2.*self.dt_movie
            self.display_quantities()
    
    def pause(self):
        print('stopping')
        self.updateTimer.stop()
        self.playButton.setEnabled(True)
        self.pauseButton.setEnabled(False)
        self.pauseButton.setChecked(True)

    def refresh(self):
        self.plot.clear()
        self.tzoom = self.plot.getAxis('bottom').range
        self.display_quantities()
        
    def update_frame(self, from_slider=True):
        
        self.tzoom = self.plot.getAxis('bottom').range
        if from_slider:
            # update time based on slider
            self.time = self.tzoom[0]+(self.tzoom[1]-self.tzoom[0])*\
                float(self.frameSlider.value())/self.settings['Npoints']
        else:
            self.time = self.xaxis.range[0]

        plots.raw_data_plot(self, self.tzoom,
                            with_images=True,
                            with_scatter=True,
                            plot_update=False)



        

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

    def see_metadata(self):
        for key, val in self.data.metadata.items():
            print('- %s : ' % key, val)
            
    def change_settings(self):
        pass
    
    def quit(self):
        if self.data is not None:
            self.data.io.close()
        sys.exit()


def add_buttons(self, Layout):

    self.styleUnpressed = ("QPushButton {Text-align: left; "
                           "background-color: rgb(200, 200, 200); "
                           "color:white;}")
    self.stylePressed = ("QPushButton {Text-align: left; "
                         "background-color: rgb(100,50,100); "
                         "color:white;}")
    self.styleInactive = ("QPushButton {Text-align: left; "
                          "background-color: rgb(200, 200, 200); "
                          "color:gray;}")

    iconSize = QtCore.QSize(20, 20)

    self.playButton = QtWidgets.QToolButton()
    self.playButton.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
    self.playButton.setIconSize(iconSize)
    self.playButton.setToolTip("Play   -> [Space]")
    self.playButton.setCheckable(True)
    self.playButton.setEnabled(True)
    self.playButton.clicked.connect(self.play)

    self.pauseButton = QtWidgets.QToolButton()
    self.pauseButton.setCheckable(True)
    self.pauseButton.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPause))
    self.pauseButton.setIconSize(iconSize)
    self.pauseButton.setToolTip("Pause   -> [Space]")
    self.pauseButton.clicked.connect(self.pause)

    btns = QtWidgets.QButtonGroup(self)
    btns.addButton(self.playButton,0)
    btns.addButton(self.pauseButton,1)
    btns.setExclusive(True)

    self.playButton.setEnabled(False)
    self.pauseButton.setEnabled(True)
    self.pauseButton.setChecked(True)

    
    self.refreshButton = QtWidgets.QToolButton()
    self.refreshButton.setCheckable(True)
    self.refreshButton.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_BrowserReload))
    self.refreshButton.setIconSize(iconSize)
    self.refreshButton.setToolTip("Refresh   -> [r]")
    self.refreshButton.clicked.connect(self.refresh)

    self.quitButton = QtWidgets.QToolButton()
    # self.quitButton.setCheckable(True)
    self.quitButton.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogCloseButton))
    self.quitButton.setIconSize(iconSize)
    self.quitButton.setToolTip("Quit")
    self.quitButton.clicked.connect(self.quit)
    
    self.backButton = QtWidgets.QToolButton()
    # self.backButton.setCheckable(True)
    self.backButton.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogBack))
    self.backButton.setIconSize(iconSize)
    self.backButton.setToolTip("Back to initial view   -> [i]")
    self.backButton.clicked.connect(self.back_to_initial_view)

    self.settingsButton = QtWidgets.QToolButton()
    # self.settingsButton.setCheckable(True)
    self.settingsButton.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogDetailedView))
    self.settingsButton.setIconSize(iconSize)
    # self.settingsButton.setToolTip("Settings")
    # self.settingsButton.clicked.connect(self.change_settings)
    self.settingsButton.setToolTip("Metadata")
    self.settingsButton.clicked.connect(self.see_metadata)
    
    Layout.addWidget(self.quitButton)
    Layout.addWidget(self.playButton)
    Layout.addWidget(self.pauseButton)
    Layout.addWidget(self.refreshButton)
    Layout.addWidget(self.backButton)
    Layout.addWidget(self.settingsButton)
    

        

def run(app, args=None, parent=None,
        raw_data_visualization=False):
    return MainWindow(app,
                      args=args,
                      raw_data_visualization=raw_data_visualization,
                      parent=parent)
    
if __name__=='__main__':
    
    from misc.colors import build_dark_palette
    import tempfile, argparse, os
    parser=argparse.ArgumentParser(description="Experiment interface",
                       formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-rf', "--root_datafolder", type=str,
                        default=os.path.join(os.path.expanduser('~'), 'DATA'))
    parser.add_argument('-df', "--datafile", type=str, default='')
    parser.add_argument('-v', "--visualization", action="store_true")
    args = parser.parse_args()
    app = QtWidgets.QApplication(sys.argv)
    build_dark_palette(app)
    main = MainWindow(app,
                      args=args,
                      raw_data_visualization=args.visualization)
    sys.exit(app.exec_())

