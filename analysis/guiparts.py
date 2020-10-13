import datetime, numpy, os
from PyQt5 import QtGui, QtWidgets, QtCore
import pyqtgraph as pg


df_width = 600
selector_height = 40


def build_dark_palette(app):

    app.setStyle("Fusion")

    # Now use a palette to switch to dark colors:
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(53, 53, 53))
    # palette.setColor(QtGui.QPalette.Background, QtGui.QColor(53, 53, 53))
    # palette.setColor(QtGui.QPalette.PlaceholderText, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    # palette.setColor(QtGui.QPalette.Foreground, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(25, 25, 25))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.black)
    palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
    # palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.Button, QtCore.Qt.black)
    palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
    # palette.setColor(QtGui.QPalette.Link, QtGui.QColor(42, 130, 218))
    # palette.setColor(QtGui.QPalette.Link, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Link, QtGui.QColor(200, 200, 200))
    # palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(150, 150, 150))
    palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)
    app.setPalette(palette)

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
    self.cal.setMinimumHeight(160)
    self.cal.setMaximumHeight(160)
    self.cal.setMinimumWidth(265)
    self.cal.setMaximumWidth(265)
    self.cal.setMinimumDate(QtCore.QDate(datetime.date(*min_date)))
    self.cal.setMaximumDate(QtCore.QDate(datetime.date.today()+datetime.timedelta(1)))
    self.cal.clicked.connect(self.pick_date)
    Layout.addWidget(self.cal)

def add_buttons(self, Layout):

    # self.setStyleSheet("QMainWindow {background: 'black';}")
    self.styleUnpressed = ("QPushButton {Text-align: left; "
                           "background-color: rgb(50,50,50); "
                           "color:white;}")
    self.stylePressed = ("QPushButton {Text-align: left; "
                         "background-color: rgb(100,50,100); "
                         "color:white;}")
    self.styleInactive = ("QPushButton {Text-align: left; "
                          "background-color: rgb(50,50,50); "
                          "color:gray;}")

    iconSize = QtCore.QSize(20, 20)

    self.playButton = QtGui.QToolButton()
    self.playButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPlay))
    self.playButton.setIconSize(iconSize)
    self.playButton.setToolTip("Play")
    self.playButton.setCheckable(True)
    self.playButton.clicked.connect(self.play)

    self.pauseButton = QtGui.QToolButton()
    self.pauseButton.setCheckable(True)
    self.pauseButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPause))
    self.pauseButton.setIconSize(iconSize)
    self.pauseButton.setToolTip("Pause")
    self.pauseButton.clicked.connect(self.pause)

    self.refreshButton = QtGui.QToolButton()
    self.refreshButton.setCheckable(True)
    self.refreshButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_BrowserReload))
    self.refreshButton.setIconSize(iconSize)
    self.refreshButton.setToolTip("Refresh")
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
    self.backButton.setToolTip("Back to initial view")
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
    Layout.addWidget(self.frameSlider)

        
def load_config1(self, win1_Wmax=800, win1_Wmin=300, win1_Hmax=300):

    self.cwidget = QtGui.QWidget(self)
    self.setCentralWidget(self.cwidget)
    self.PupilROI = None
    
    mainLayout = QtWidgets.QVBoxLayout()

    Layout1 = QtWidgets.QHBoxLayout()
    mainLayout.addLayout(Layout1)

    Layout11 = QtWidgets.QVBoxLayout()
    Layout1.addLayout(Layout11)
    create_calendar(self, Layout11)
    self.notes = QtWidgets.QLabel(63*'-'+5*'\n', self)
    self.notes.setMinimumHeight(70)
    self.notes.setMaximumHeight(70)
    Layout11.addWidget(self.notes)

    self.pbox = QtWidgets.QComboBox(self)
    self.pbox.activated.connect(self.display_quantities)
    self.pbox.setMaximumHeight(selector_height)
    if self.raw_data_visualization:
        self.pbox.addItem('-> Show Raw Data')
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
    self.dbox.activated.connect(self.pick_datafolder)
    Layout12.addWidget(self.dbox)

    self.win1 = pg.GraphicsLayoutWidget()
    self.win1.setMaximumWidth(win1_Wmax)
    self.win1.setMaximumHeight(win1_Hmax-1.5*selector_height)
    Layout12.addWidget(self.win1)

    self.win2 = pg.GraphicsLayoutWidget()
    self.win2.setMaximumHeight(win1_Hmax)
    Layout1.addWidget(self.win2)

    self.winTrace = pg.GraphicsLayoutWidget()
    mainLayout.addWidget(self.winTrace)

    build_slider(self, mainLayout)

    self.cwidget.setLayout(mainLayout)
    self.show()
    
    self.pScreen = self.win1.addViewBox(lockAspect=True,row=0,col=0,invertY=True,border=[20,20,20])
    self.pScreenimg = pg.ImageItem(numpy.ones((10,12))*50)
    self.pScreenimg.setLevels([0,255])
    self.pFace = self.win1.addViewBox(lockAspect=True,row=0,col=1,invertY=True,border=[20,20,20])
    self.pFaceimg = pg.ImageItem(numpy.ones((10,12))*50)
    self.pFaceimg.setLevels([0,255])
    self.pPupil=self.win1.addViewBox(lockAspect=True,row=0,col=2,invertY=True, border=[20, 20, 20])
    self.pPupilimg = pg.ImageItem(numpy.ones((10,12))*50)
    self.pPupilimg.setLevels([0,255])
    self.pCa=self.win2.addViewBox(lockAspect=True,invertY=True, border=[20, 20, 20])
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

    label = pg.LabelItem()
    txt = """
    <span style='font-size: 12pt'>
    <span style='color: grey'> <b>Screen</b> </span> <br/>
    <span style='color: white'> <b>Locomotion</b> </span> <br/>
    <span style='color: red'> <b>Pupil</b> </span> <br/>
    <span style='color: blue'> <b>Electrophy</b> </span> <br/>
    <span style='color: green'> <b> Calcium </b> </span>"""
    label.setText(txt)
    self.win2.addItem(label)
    
def load_config2(self, win1_Wmax=800, win1_Hmax=300):

    self.cwidget = QtGui.QWidget(self)
    self.setCentralWidget(self.cwidget)

    mainLayout = QtWidgets.QVBoxLayout()

    Layout1 = QtWidgets.QVBoxLayout()
    mainLayout.addLayout(Layout1)
    
