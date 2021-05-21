import datetime, numpy, os, sys
from PyQt5 import QtGui, QtWidgets, QtCore
import pyqtgraph as pg
import numpy as np

smallfont = QtGui.QFont()
smallfont.setPointSize(7)
verysmallfont = QtGui.QFont()
verysmallfont.setPointSize(5)

def remove_size_props(o, fcol=False, button=False, image=False):
    o.setMinimumHeight(0)
    o.setMinimumWidth(0)
    o.setMaximumHeight(int(1e5))
    if fcol:
        o.setMaximumWidth(250) # just a max width for the first column
    elif button:
        o.setMaximumWidth(250/5) # just a max width for the first column
    elif image:
        # o.setMaximumWidth(500) # just a max width for the first column
        o.setMaximumHeight(500) # just a max width for the first column
    else:
        o.setMaximumWidth(int(1e5))


def create_calendar(self, Layout, min_date=(2020, 8, 1)):
    
    self.cal = QtWidgets.QCalendarWidget(self)
    self.cal.setFont(verysmallfont)
    self.cal.setMinimumHeight(160)
    self.cal.setMaximumHeight(160)
    self.cal.setMinimumWidth(265)
    self.cal.setMaximumWidth(265)
    self.cal.setMinimumDate(QtCore.QDate(datetime.date(*min_date)))
    self.cal.setMaximumDate(QtCore.QDate(datetime.date.today()+datetime.timedelta(1)))
    self.cal.adjustSize()
    self.cal.clicked.connect(self.pick_date)
    Layout.addWidget(self.cal)

    # setting an highlight format
    self.highlight_format = QtGui.QTextCharFormat()
    self.highlight_format.setBackground(self.cal.palette().brush(QtGui.QPalette.Link))
    self.highlight_format.setForeground(self.cal.palette().color(QtGui.QPalette.BrightText))

    self.cal.setSelectedDate(datetime.date.today())
    
    

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

    self.playButton = QtGui.QToolButton()
    self.playButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPlay))
    self.playButton.setIconSize(iconSize)
    self.playButton.setToolTip("Play   -> [Space]")
    self.playButton.setCheckable(True)
    self.playButton.setEnabled(True)
    self.playButton.clicked.connect(self.play)

    self.pauseButton = QtGui.QToolButton()
    self.pauseButton.setCheckable(True)
    self.pauseButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPause))
    self.pauseButton.setIconSize(iconSize)
    self.pauseButton.setToolTip("Pause   -> [Space]")
    self.pauseButton.clicked.connect(self.pause)

    btns = QtGui.QButtonGroup(self)
    btns.addButton(self.playButton,0)
    btns.addButton(self.pauseButton,1)
    btns.setExclusive(True)

    self.playButton.setEnabled(False)
    self.pauseButton.setEnabled(True)
    self.pauseButton.setChecked(True)

    
    self.refreshButton = QtGui.QToolButton()
    self.refreshButton.setCheckable(True)
    self.refreshButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_BrowserReload))
    self.refreshButton.setIconSize(iconSize)
    self.refreshButton.setToolTip("Refresh   -> [r]")
    self.refreshButton.clicked.connect(self.refresh)

    self.quitButton = QtGui.QToolButton()
    # self.quitButton.setCheckable(True)
    self.quitButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_DialogCloseButton))
    self.quitButton.setIconSize(iconSize)
    self.quitButton.setToolTip("Quit")
    self.quitButton.clicked.connect(self.quit)
    
    self.backButton = QtGui.QToolButton()
    # self.backButton.setCheckable(True)
    self.backButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_FileDialogBack))
    self.backButton.setIconSize(iconSize)
    self.backButton.setToolTip("Back to initial view   -> [i]")
    self.backButton.clicked.connect(self.back_to_initial_view)

    self.settingsButton = QtGui.QToolButton()
    # self.settingsButton.setCheckable(True)
    self.settingsButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_FileDialogDetailedView))
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
    

def build_slider(self, Layout):
    self.frameSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
    self.frameSlider.setMinimum(0)
    self.frameSlider.setMaximum(self.settings['Npoints'])
    self.frameSlider.setTickInterval(1)
    self.frameSlider.setTracking(False)
    self.frameSlider.valueChanged.connect(self.update_frame)
    self.frameSlider.setMaximumHeight(20)
    Layout.addWidget(self.frameSlider)

        
def load_config1(self,
                 df_width = 600,
                 selector_height = 30,
                 win1_Wmax=1200, win1_Wmin=300,
                 win1_Hmax=500, win2_Wmax=500):

    self.cwidget = QtGui.QWidget(self)
    self.setCentralWidget(self.cwidget)

    self.statusBar.showMessage('open file [Ctrl+O],    refresh plot [Ctrl+R],    play/pause [Ctrl+Space],    initial-view [Ctrl-I],    max-window [Ctrl+M] ' )
        
    mainLayout = QtWidgets.QVBoxLayout()

    Layout1 = QtWidgets.QHBoxLayout()
    mainLayout.addLayout(Layout1)

    Layout11 = QtWidgets.QVBoxLayout()
    Layout1.addLayout(Layout11)
    create_calendar(self, Layout11)
    self.cal.setMaximumHeight(150)

    # folder box
    self.fbox = QtWidgets.QComboBox(self)
    self.fbox.setFont(smallfont)
    self.fbox.activated.connect(self.scan_folder)
    self.fbox.setMaximumHeight(selector_height)
    self.folder_default_key = '  [root datafolder]'
    self.fbox.addItem(self.folder_default_key)
    self.fbox.setCurrentIndex(0)
    Layout11.addWidget(self.fbox)
    
    # subject box
    self.sbox = QtWidgets.QComboBox(self)
    self.sbox.setFont(smallfont)
    self.sbox.activated.connect(self.pick_subject) # To be written !!
    self.sbox.setMaximumHeight(selector_height)
    self.subject_default_key = '  [subject] '
    self.sbox.addItem(self.subject_default_key)
    
    self.sbox.setCurrentIndex(0)
    Layout11.addWidget(self.sbox)
    
    # notes
    self.notes = QtWidgets.QLabel('\n[exp info]'+5*'\n', self)
    self.notes.setFont(smallfont)
    Layout11.addWidget(self.notes)

    self.pbox = QtWidgets.QComboBox(self)
    self.pbox.setFont(smallfont)
    self.pbox.activated.connect(self.display_quantities)
    self.pbox.setMaximumHeight(selector_height)
    self.pbox.addItem('[visualization/analysis]')
    self.pbox.addItem('-> Show Raw Data')
    self.pbox.addItem('-> Trial-average')
    self.pbox.addItem('-> Behavioral-modulation')
    self.pbox.addItem('-> Make-figures')
    self.pbox.addItem('-> Open PDF summary')
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
    self.win1.setMaximumWidth(win1_Wmax)
    self.win1.setMaximumHeight(win1_Hmax-1.5*selector_height)
    Layout12.addWidget(self.win1)

    self.win2 = pg.GraphicsLayoutWidget()
    self.win2.setMaximumWidth(win2_Wmax)
    self.win2.setMaximumHeight(win1_Hmax)
    Layout1.addWidget(self.win2)

    self.winTrace = pg.GraphicsLayoutWidget()
    mainLayout.addWidget(self.winTrace)

    build_slider(self, mainLayout)

    self.pScreen = self.win1.addViewBox(lockAspect=True, invertY=True, border=[1, 1, 1], colspan=2)
    self.pScreenimg = pg.ImageItem(numpy.ones((10,12))*50)
    self.pScreenimg.setLevels([0,255])
    self.pFace = self.win1.addViewBox(lockAspect=True, invertY=True, border=[1, 1, 1], colspan=2)
    self.pFaceimg = pg.ImageItem(numpy.ones((10,12))*50)
    self.pFaceimg.setLevels([0,255])
    self.pPupil=self.win1.addViewBox(lockAspect=True, invertY=True, border=[1, 1, 1])
    self.pPupilimg = pg.ImageItem(numpy.ones((10,12))*50)
    self.pPupilimg.setLevels([0,255])
    self.pCa=self.win2.addViewBox(lockAspect=True,invertY=True, border=[1, 1, 1])
    self.pCaimg = pg.ImageItem(numpy.ones((50,50))*100)
    self.pCaimg.setLevels([0,255])
    for x, y in zip([self.pScreen, self.pFace,self.pPupil,self.pCa],
                    [self.pScreenimg, self.pFaceimg, self.pPupilimg, self.pCaimg]):
        x.setAspectLocked()
        x.addItem(y)
        x.show()

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
    
    self.roiPick = QtGui.QLineEdit()
    self.roiPick.setText(' [...] ')
    self.roiPick.setMinimumWidth(150)
    self.roiPick.setMaximumWidth(350)
    self.roiPick.returnPressed.connect(self.select_ROI)
    self.roiPick.setFont(smallfont)

    self.ephysPick = QtGui.QLineEdit()
    self.ephysPick.setText(' ')
    # self.ephysPick.returnPressed.connect(self.select_ROI)
    self.ephysPick.setFont(smallfont)

    self.guiKeywords = QtGui.QLineEdit()
    self.guiKeywords.setText('     [GUI keywords] ')
    self.guiKeywords.setFixedWidth(200)
    self.guiKeywords.returnPressed.connect(self.keyword_update)
    self.guiKeywords.setFont(smallfont)
    
    Layout122.addWidget(self.guiKeywords)
    Layout122.addWidget(self.ephysPick)
    Layout122.addWidget(self.roiPick)
    
    self.cwidget.setLayout(mainLayout)
    self.show()

class NewWindow(QtWidgets.QMainWindow):
    
    def __init__(self,
                 parent=None, i=0,
                 title='New Window'):

        super(NewWindow, self).__init__()

        self.setGeometry(600+20*i, 250+20*i, 1000, 600)
        pg.setConfigOptions(imageAxisOrder='row-major')

        # adding a few keyboard shortcut
        self.openSc = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+O'), self)
        self.openSc.activated.connect(self.open_file)
        self.openSc = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+Space'), self)
        self.openSc.activated.connect(self.hitting_space)
        self.quitSc = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+Q'), self)
        self.quitSc.activated.connect(self.quit)
        self.refreshSc = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+R'), self)
        self.refreshSc.activated.connect(self.refresh)
        self.homeSc = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+I'), self)
        self.homeSc.activated.connect(self.back_to_initial_view)
        self.maxSc = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+M'), self)
        self.maxSc.activated.connect(self.showwindow)
        
        self.setWindowTitle(title)
        
        # adding a "quit" keyboard shortcut
        self.quitSc = QtWidgets.QShortcut(QtGui.QKeySequence('Q'), self) # or 'Ctrl+Q'
        self.quitSc.activated.connect(self.quit)
        self.closeW = QtWidgets.QShortcut(QtGui.QKeySequence('C'), self) # or 'Ctrl+Q'
        self.closeW.activated.connect(self.close)
        # adding a refresh shortcut
        self.refreshSc = QtWidgets.QShortcut(QtGui.QKeySequence('R'), self) # or 'Ctrl+Q'
        self.refreshSc.activated.connect(self.refresh)
        self.maxSc = QtWidgets.QShortcut(QtGui.QKeySequence('M'), self)
        self.maxSc.activated.connect(self.showwindow)

        self.setWindowTitle(title)
        self.minView = False

        self.cwidget = QtGui.QWidget(self)
        self.setCentralWidget(self.cwidget)
        
        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.setFont(smallfont)

        self.showwindow()

    def quit(self):
        sys.exit()
        
    def refresh(self):
        pass # to be implemented in the child class !

    def open_file(self):
        pass # to be implemented in the child class !

    def hitting_space(self):
        pass # to be implemented in the child class !
    
    def back_to_initial_view(self):
        pass # to be implemented in the child class !
    
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
    
    def select_ROI_from_pick(self, cls=None):

        if cls is None:
            cls = self # so that 

        if cls.roiPick.text() in ['sum', 'all']:
            roiIndices = np.arange(np.sum(self.iscell))
        elif len(cls.roiPick.text().split('-'))>1:
            try:
                roiIndices = np.arange(int(cls.roiPick.text().split('-')[0]), int(cls.roiPick.text().split('-')[1]))
            except BaseException as be:
                print(be)
                roiIndices = None
        elif len(cls.roiPick.text().split(','))>1:
            try:
                roiIndices = np.array([int(ii) for ii in cls.roiPick.text().split(',')])
            except BaseException as be:
                print(be)
                roiIndices = None
        else:
            try:
                i0 = int(cls.roiPick.text())
                if (i0<0) or (i0>=len(self.validROI_indices)):
                    roiIndices = [0]
                    self.statusBar.showMessage(' "%i" not a valid ROI index'  % i0)
                else:
                    roiIndices = [i0]

            except BaseException as be:
                print(be)
                roiIndices = []
                self.statusBar.showMessage(' /!\ Problem in setting indices /!\ ')
        return roiIndices

    def keyword_update(self, string=None, parent=None):

        if string is None:
            string = self.guiKeywords.text()

        cls = (parent if parent is not None else self)
        
        if string in ['Stim', 'stim', 'VisualStim', 'Stimulation', 'stimulation']:
            cls.load_VisualStim()
        elif string in ['no_stim', 'no_VisualStim']:
            cls.visual_stim = None
        elif string in ['scan_folder', 'scanF', 'scan']:
            cls.scan_folder()
        elif string in ['meanImg', 'meanImgE', 'Vcorr', 'max_proj']:
            cls.CaImaging_bg_key = string
        elif string=='no_subsampling':
            cls.no_subsampling = True
        elif string in ['F', 'Fluorescence', 'Neuropil', 'Deconvolved', 'Fneu', 'dF/F', 'dFoF'] or ('F-' in string):
            if string=='F':
                cls.CaImaging_key = 'Fluorescence'
            elif string=='Fneu':
                cls.CaImaging_key = 'Neuropil'
            else:
                cls.CaImaging_key = string
        elif string=='subsampling':
            cls.no_subsampling = False
        elif string=='subjects':
            cls.compute_subjects()
        else:
            self.statusBar.showMessage('  /!\ keyword "%s" not recognized /!\ ' % string)

    # Layout11 = QtWidgets.QVBoxLayout()
    # Layout1.addLayout(Layout11)
    # create_calendar(self, Layout11)
    # self.notes = QtWidgets.QLabel(63*'-'+5*'\n', self)
    # self.notes.setMinimumHeight(70)
    # self.notes.setMaximumHeight(70)
    # Layout11.addWidget(self.notes)

    # self.pbox = QtWidgets.QComboBox(self)
    # self.pbox.activated.connect(self.display_quantities)
    # self.pbox.setMaximumHeight(selector_height)
    # if self.raw_data_visualization:
    #     self.pbox.addItem('')
    #     self.pbox.addItem('-> Show Raw Data')
    #     self.pbox.setCurrentIndex(1)
    
    
#     def __init__(self, parent=None,
#                  fullscreen=False):

#         super(TrialAverageWindow, self).__init__()

#         # adding a "quit" keyboard shortcut
#         self.quitSc = QtWidgets.QShortcut(QtGui.QKeySequence('Q'), self) # or 'Ctrl+Q'
#         self.quitSc.activated.connect(self.close)
#         self.refreshSc = QtWidgets.QShortcut(QtGui.QKeySequence('R'), self) # or 'Ctrl+Q'
#         self.refreshSc.activated.connect(self.refresh)
#         self.maxSc = QtWidgets.QShortcut(QtGui.QKeySequence('M'), self)
#         self.maxSc.activated.connect(self.showwindow)

#         ####################################################
#         # BASIC style config
#         self.setWindowTitle('Analysis Program -- Physiology of Visual Circuits')

#     def close(self):
#         pass

    
    
if __name__=='__main__':

    pass
    # app = QtWidgets.QApplication(sys.argv)
    # # build_dark_palette(app)
    # # window = TrialAverageWindow(app)
    # # window.show()
    # sys.exit(app.exec_())
