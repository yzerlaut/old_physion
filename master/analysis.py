import sys, time, tempfile, os, pathlib, json, subprocess
import numpy as np
from PyQt5 import QtGui, QtWidgets, QtCore
import pyqtgraph as pg

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import day_folder, generate_filename_path, save_dict, load_dict
from master import guiparts

## NASTY workaround to the error:
# ** OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized. **

class MasterWindow(QtWidgets.QMainWindow):
    
    def __init__(self, app,
                 saturation=100,
                 fullscreen=False):

        guiparts.build_dark_palette(app)
        
        super(MasterWindow, self).__init__()



        # adding a "quit" keyboard shortcut
        self.quitSc = QtWidgets.QShortcut(QtGui.QKeySequence('Q'), self) # or 'Ctrl+Q'
        self.quitSc.activated.connect(self.quit)

        ####################################################
        # BASIC style config
        self.setWindowTitle('Analysis Program -- Physiology of Visual Circuits')
        # pg.setConfigOptions(imageAxisOrder='row-major')
        # self.setStyleSheet("QMainWindow {background: 'black';}")
        # self.styleUnpressed = ("QPushButton {Text-align: left; "
        #                        "background-color: rgb(50,50,50); "
        #                        "color:white;}")
        # self.stylePressed = ("QPushButton {Text-align: left; "
        #                      "background-color: rgb(100,50,100); "
        #                      "color:white;}")
        # self.styleInactive = ("QPushButton {Text-align: left; "
        #                       "background-color: rgb(50,50,50); "
        #                       "color:gray;}")
        if fullscreen:
            self.showFullScreen()
        else:
            self.setGeometry(200,200,1300,700)

        Nw, Nh = 10, 10
        Layout = {'image':(0,1,Nw-1,int(Nh/3)),
                  'calendar':(0,0,1,1),
                  'play':(Nh-1,0,1,1),
                  'frameSlider':(Nw-1,1,1,Nh-1)}
        iconSize = QtCore.QSize(100, 100)
                  
        ####################################################
        # Widget elements
        self.cwidget = QtGui.QWidget(self)
        self.setCentralWidget(self.cwidget)
        self.grid = QtGui.QGridLayout()

        # a big window with the different images
        self.win = pg.GraphicsLayoutWidget()
        self.grid.addWidget(self.win,*Layout['image'])

        self.cal = QtWidgets.QCalendarWidget(self)
        self.cal.move(70, 120)
        self.cal.setMinimumWidth(350)
        self.cal.setMinimumHeight(220)
        # self.cal.setMinimumDate(QtCore.QDate(init_date))
        self.cal.setMaximumDate(QtCore.QDate.currentDate())
        # self.cal.clicked.connect(self.pick_date)
        self.grid.addWidget(self.cal,*Layout['calendar'])
        
        self.pFace = self.win.addViewBox(lockAspect=True,row=0,col=0,invertY=True,border=[50,50,50])
        self.pFaceimg = pg.ImageItem(None)
        self.pPupil=self.win.addViewBox(lockAspect=True,row=0,col=1,invertY=True, border=[50,50,50])
        self.pPupilimg = pg.ImageItem(None)
        self.pCa=self.win.addViewBox(lockAspect=True,row=0,col=2,invertY=True, border=[50,50,50])
        self.pCaimg = pg.ImageItem(None)
        for x, y in zip([self.pFace,self.pPupil,self.pCa],
                        [self.pFaceimg, self.pPupilimg, self.pCaimg]):
            x.setAspectLocked()
            x.addItem(y)
            x.show()
            

        self.frameSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.frameSlider.setMinimum(0)
        # self.frameSlider.setMaximum(self.slider_nframes)
        self.frameSlider.setTickInterval(1)
        self.frameSlider.setTracking(False)
        self.frameSlider.valueChanged.connect(self.quit)
        self.grid.addWidget(self.frameSlider, *Layout['frameSlider'])
        

        self.playButton = QtGui.QToolButton()
        self.playButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPlay))
        # self.playButton.setIconSize(iconSize)
        self.playButton.setToolTip("Play")
        self.playButton.setCheckable(True)
        self.grid.addWidget(self.playButton,*Layout['play'])
        
        self.cwidget.setLayout(self.grid)
        self.win.show()
        self.show()


    def quit(self):
        try:
            self.camera.cam.stop()
        except Exception:
            pass
        sys.exit()

app = QtWidgets.QApplication(sys.argv)
main = MasterWindow(app)
sys.exit(app.exec_())
