import datetime, numpy
from PyQt5 import QtGui, QtWidgets, QtCore
import pyqtgraph as pg


df_width = 500
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
    palette.setColor(QtGui.QPalette.Link, QtCore.Qt.white)
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
    self.cal.setMinimumWidth(265)
    self.cal.setMaximumHeight(160)
    self.cal.setMaximumWidth(300)
    self.cal.setMinimumDate(QtCore.QDate(datetime.date(*min_date)))
    self.cal.setMaximumDate(QtCore.QDate.currentDate())
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
    self.quitButton.setCheckable(True)
    self.quitButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_BrowserStop))
    self.quitButton.setIconSize(iconSize)
    self.quitButton.setToolTip("Quit")
    self.quitButton.clicked.connect(self.quit)
    
    Layout.addWidget(self.quitButton)
    Layout.addWidget(self.playButton)
    Layout.addWidget(self.pauseButton)
    Layout.addWidget(self.refreshButton)
    

def build_slider(self, Layout):
    self.frameSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
    self.frameSlider.setMinimum(0)
    self.frameSlider.setMaximum(100)
    self.frameSlider.setTickInterval(0.5)
    self.frameSlider.setTracking(False)
    self.frameSlider.valueChanged.connect(self.update_frame)
    Layout.addWidget(self.frameSlider)

        
def load_config1(self, win1_Wmax=800, win1_Hmax=300):

    self.cwidget = QtGui.QWidget(self)
    self.setCentralWidget(self.cwidget)

    mainLayout = QtWidgets.QVBoxLayout()

    Layout1 = QtWidgets.QHBoxLayout()
    mainLayout.addLayout(Layout1)

    Layout11 = QtWidgets.QVBoxLayout()
    Layout1.addLayout(Layout11)
    create_calendar(self, Layout11)
    self.notes = QtWidgets.QLabel('...', self)
    self.notes.setMinimumHeight(40)
    self.notes.setMaximumHeight(60)
    Layout11.addWidget(self.notes)

    self.pbox = QtWidgets.QComboBox(self)
    self.pbox.activated.connect(self.display_quantities)
    self.pbox.setMaximumHeight(selector_height)
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
    
    self.pFace = self.win1.addViewBox(lockAspect=True,row=0,col=0,invertY=True,border=[20,20,20])
    self.pFaceimg = pg.ImageItem(None)
    self.pPupil=self.win1.addViewBox(lockAspect=True,row=0,col=1,invertY=True, border=[20, 20, 20])
    self.pPupilimg = pg.ImageItem(None)
    self.pCa=self.win2.addViewBox(lockAspect=True,invertY=True, border=[20, 20, 20])
    self.pCaimg = pg.ImageItem(None)
    for x, y in zip([self.pFace,self.pPupil,self.pCa],
                    [self.pFaceimg, self.pPupilimg, self.pCaimg]):
        x.setAspectLocked()
        x.addItem(y)
        x.show()
        y.setImage(numpy.random.randn(10,13))

    self.p1 = self.winTrace.addPlot(row=0, col=0, rowspan=2, title='Calcium traces')
    self.p2 = self.winTrace.addPlot(row=2, col=0, rowspan=1, title='Pupil diameter')
    self.p3 = self.winTrace.addPlot(row=3, col=0, rowspan=1, title='Locomotion')
    self.p4 = self.winTrace.addPlot(row=4, col=0, rowspan=1, title='Electrophysiology')
    for p in [self.p1, self.p2, self.p3, self.p4]:
        p.hideAxis('left')
        p.setMouseEnabled(x=True,y=False)
        p.setMenuEnabled(False)
        p.plot(numpy.arange(200), numpy.random.randn(200))
        p.setRange(xRange=(0,200), padding=0.0)
        

def load_config2(self):

    init_layout(self)
    
    self.win1 = pg.GraphicsLayoutWidget()
    self.win2 = pg.GraphicsLayoutWidget()
    self.winTrace = pg.GraphicsLayoutWidget()

    self.grid.addWidget(self.winTrace, 0, self.Csplit, 1, self.Ncol-self.Csplit)
    self.grid.addWidget(self.win2, self.Rsplit, 0, self.Nrow-self.Rsplit-1, self.CalendarSize[1])
    self.grid.addWidget(self.win1, 1, self.CalendarSize[0], self.Nrow-2,self.Ncol-self.CalendarSize[1])

    # self.layout.setColumnStretch(0, )
    
    remove_size_props(self.win1)
    remove_size_props(self.win3)
    remove_size_props(self.win2, fcol=True)

    draw_layout(self)
