import datetime, numpy, os, sys, pathlib
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
    
def reinit_calendar(self, min_date=(2020, 8, 1), max_date=None):
    
    day, i = datetime.date(*min_date), 0
    while day!=datetime.date.today():
        self.cal.setDateTextFormat(QtCore.QDate(day),
                                   QtGui.QTextCharFormat())
        day = day+datetime.timedelta(1)
    day = day+datetime.timedelta(1)
    self.cal.setDateTextFormat(QtCore.QDate(day),
                               QtGui.QTextCharFormat())
    self.cal.setMinimumDate(QtCore.QDate(datetime.date(*min_date)))
    
    if max_date is not None:
        self.cal.setMaximumDate(QtCore.QDate(datetime.date(*max_date)+datetime.timedelta(1)))
        self.cal.setSelectedDate(datetime.date(*max_date)+datetime.timedelta(1))
    else:
        self.cal.setMaximumDate(QtCore.QDate(datetime.date.today()+datetime.timedelta(1)))
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

    self.playButton = QtWidgets.QToolButton()
    self.playButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPlay))
    self.playButton.setIconSize(iconSize)
    self.playButton.setToolTip("Play   -> [Space]")
    self.playButton.setCheckable(True)
    self.playButton.setEnabled(True)
    self.playButton.clicked.connect(self.play)

    self.pauseButton = QtWidgets.QToolButton()
    self.pauseButton.setCheckable(True)
    self.pauseButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPause))
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
    self.refreshButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_BrowserReload))
    self.refreshButton.setIconSize(iconSize)
    self.refreshButton.setToolTip("Refresh   -> [r]")
    self.refreshButton.clicked.connect(self.refresh)

    self.quitButton = QtWidgets.QToolButton()
    # self.quitButton.setCheckable(True)
    self.quitButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_DialogCloseButton))
    self.quitButton.setIconSize(iconSize)
    self.quitButton.setToolTip("Quit")
    self.quitButton.clicked.connect(self.quit)
    
    self.backButton = QtWidgets.QToolButton()
    # self.backButton.setCheckable(True)
    self.backButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_FileDialogBack))
    self.backButton.setIconSize(iconSize)
    self.backButton.setToolTip("Back to initial view   -> [i]")
    self.backButton.clicked.connect(self.back_to_initial_view)

    self.settingsButton = QtWidgets.QToolButton()
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
    self.frameSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    self.frameSlider.setMinimum(0)
    self.frameSlider.setMaximum(self.settings['Npoints'])
    self.frameSlider.setTickInterval(1)
    self.frameSlider.setTracking(False)
    self.frameSlider.valueChanged.connect(self.update_frame)
    self.frameSlider.setMaximumHeight(20)
    Layout.addWidget(self.frameSlider)


class NewWindow(QtWidgets.QMainWindow):
    
    def __init__(self,
                 parent=None, i=0,
                 title='New Window',
                 size=(1000,600)):

        super(NewWindow, self).__init__()

        self.script = os.path.join(str(pathlib.Path(__file__).resolve().parents[1]),
                                   'script.sh') # for batch processing
        
        self.setGeometry(600+20*i, 250+20*i, size[0], size[1])
        
        ##############################
        ##### keyboard shortcuts #####
        ##############################

        # adding a few general keyboard shortcut
        self.openSc = QtWidgets.QShortcut(QtGui.QKeySequence('O'), self)
        self.openSc.activated.connect(self.open_file)
        
        self.spaceSc = QtWidgets.QShortcut(QtGui.QKeySequence('Space'), self)
        self.spaceSc.activated.connect(self.hitting_space)

        self.saveSc = QtWidgets.QShortcut(QtGui.QKeySequence('S'), self)
        self.saveSc.activated.connect(self.save)
        
        self.add2Bash = QtWidgets.QShortcut(QtGui.QKeySequence('B'), self)
        self.add2Bash.activated.connect(self.add_to_bash_script)
        
        self.quitSc = QtWidgets.QShortcut(QtGui.QKeySequence('Q'), self)
        self.quitSc.activated.connect(self.quit)
        
        self.refreshSc = QtWidgets.QShortcut(QtGui.QKeySequence('R'), self)
        self.refreshSc.activated.connect(self.refresh)
        
        self.homeSc = QtWidgets.QShortcut(QtGui.QKeySequence('I'), self)
        self.homeSc.activated.connect(self.back_to_initial_view)
        
        self.maxSc = QtWidgets.QShortcut(QtGui.QKeySequence('M'), self)
        self.maxSc.activated.connect(self.showwindow)
        
        self.processSc = QtWidgets.QShortcut(QtGui.QKeySequence('P'), self)
        self.processSc.activated.connect(self.process)

        self.fitSc = QtWidgets.QShortcut(QtGui.QKeySequence('F'), self)
        self.fitSc.activated.connect(self.fit)
        
        ########################
        ##### building GUI #####
        ########################
        
        pg.setConfigOptions(imageAxisOrder='row-major')
        
        self.setWindowTitle(title)
        self.minView = False

        self.cwidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.cwidget)
        
        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.setFont(smallfont)

        self.showwindow()

    def quit(self):
        sys.exit()

    def print_datafile(self):
        if hasattr(self, 'datafile'):
            print('current datafile:\n', self.datafile)
            
    def process(self):
        print(' "process" function not implemented')
        print(' --> should be implemented in child class !')
        self.print_datafile()

    def fit(self):
        print(' "fit" function not implemented')
        print(' --> should be implemented in child class !')
        self.print_datafile()
        
    def add_to_bash_script(self):
        print(' "add_to_bash_script" function not implemented')
        print(' --> should be implemented in child class !')
        
    def refresh(self):
        print(' "refresh" function not implemented')
        print(' --> should be implemented in child class !')

    def open_file(self):
        print(' "open_file" function not implemented')
        print(' --> should be implemented in child class !')

    def hitting_space(self):
        print(' "hitting_space" function not implemented')
        print(' --> should be implemented in child class !')
    
    def back_to_initial_view(self):
        print(' "back_to_initial_view" function not implemented')
        print(' --> should be implemented in child class !')
    
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

    def save(self):
        print(' "save" function not implemented')
        print(' --> should be implemented in child class !')

    ###########################################
    ################ Widget tools #############
    ###########################################

    def init_basic_widget_grid(self,
                               wdgt_length=3,
                               Ncol_wdgt=20,
                               Nrow_wdgt=20):
        
        self.i_wdgt = 0 # initialize widget counter
        
        self.wdgt_length = wdgt_length
        
        self.cwidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.cwidget)
        
        # grid layout
        self.layout = QtWidgets.QGridLayout()
        self.cwidget.setLayout(self.layout)

        self.graphics_layout = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.graphics_layout, 0, self.wdgt_length,
                              Nrow_wdgt, Ncol_wdgt)
        
    
    def add_widget(self, wdgt, spec='None'):
        if 'small' in spec:
            wdgt.setFixedWidth(70)

        # if spec=='shift-right':
        #     self.layout.addWidget(wdgt, self.i_wdgt-1, self.wdgt_length,
        #                           1, self.wdgt_length+1)
        if spec=='small-left':
            self.layout.addWidget(wdgt, self.i_wdgt, 0, 1, 1)
        elif spec=='small-middle':
            self.layout.addWidget(wdgt, self.i_wdgt, 1, 1, 1)
        elif spec=='large-left':
            self.layout.addWidget(wdgt, self.i_wdgt, 0, 1, self.wdgt_length-1)
        elif spec=='small-right':
            self.layout.addWidget(wdgt, self.i_wdgt, self.wdgt_length-1, 1, 1)
            self.i_wdgt += 1
        elif spec=='large-right':
            self.layout.addWidget(wdgt, self.i_wdgt, 1, 1, self.wdgt_length-1)
            self.i_wdgt += 1
        else:
            self.layout.addWidget(wdgt, self.i_wdgt, 0, 1, self.wdgt_length)
            self.i_wdgt += 1
        
    ###########################################
    ########## Data-specific tools ############
    ###########################################
    
    def select_ROI_from_pick(self, data):

        if self.roiPick.text() in ['sum', 'all']:
            roiIndices = np.arange(data.nROIs)

        elif len(self.roiPick.text().split('-'))>1:
            try:
                roiIndices = np.arange(int(self.roiPick.text().split('-')[0]), int(self.roiPick.text().split('-')[1]))
            except BaseException as be:
                print(be)
                roiIndices = None

        elif len(self.roiPick.text().split(','))>1:
            try:
                roiIndices = np.array([int(ii) for ii in self.roiPick.text().split(',')])
            except BaseException as be:
                print(be)
                roiIndices = None
        else:
            try:
                i0 = int(self.roiPick.text())
                if (i0<0) or (i0>=data.nROIs):
                    roiIndices = [0]
                    self.statusBar.showMessage(' "%i" not a valid ROI index, roiIndices set to [0]'  % i0)
                else:
                    roiIndices = [i0]

            except BaseException as be:
                print(be)
                roiIndices = [0]
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
        elif string in ['meanImg', 'meanImg_chan2', 'meanImgE', 'Vcorr', 'max_proj']:
            cls.CaImaging_bg_key = string
        elif 'plane' in string:
            cls.planeID = int(string.split('plane')[1])
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
#         self.quitSc = QtWidgets.QShortcut(QtGui.QKeySequence('Q'), self) # or 'Q'
#         self.quitSc.activated.connect(self.close)
#         self.refreshSc = QtWidgets.QShortcut(QtGui.QKeySequence('R'), self) # or 'Q'
#         self.refreshSc.activated.connect(self.refresh)
#         self.maxSc = QtWidgets.QShortcut(QtGui.QKeySequence('M'), self)
#         self.maxSc.activated.connect(self.showwindow)

#         ####################################################
#         # BASIC style config
#         self.setWindowTitle('Analysis Program -- Physiology of Visual Circuits')

#     def close(self):
#         pass

class Slider(QtWidgets.QSlider):
    def __init__(self, bid, parent=None):
        super(self.__class__, self).__init__()
        self.bid = bid
        self.setOrientation(QtCore.Qt.Horizontal)
        self.setMinimum(0)
        self.setMaximum(255)
        self.setValue(255)
        self.setTickInterval(1)
        self.valueChanged.connect(lambda: self.level_change(parent,bid))
        self.setTracking(False)

    def level_change(self, parent, bid):
        parent.saturation = float(self.value())
        if parent.ROI is not None:
            parent.ROI.plot(parent)
        parent.win.show()

# ### custom QDialog which makes a list of items you can include/exclude
# class ListChooser(QtWidgets.QDialog):
#     def __init__(self, title, parent):
#         super(ListChooser, self).__init__(parent)
#         self.setGeometry(300,300,320,200)
#         self.setWindowTitle(title)
#         self.win = QtWidgets.QWidget(self)
#         layout = QtWidgets.QGridLayout()
#         self.win.setLayout(layout)
#         #self.setCentralWidget(self.win)
#         layout.addWidget(QtWidgets.QLabel('click to select videos (none selected => all used)'),0,0,1,1)
#         self.list = QtWidgets.QListWidget(parent)
#         for f in parent.filelist:
#             self.list.addItem(f)
#         layout.addWidget(self.list,1,0,7,4)
#         #self.list.resize(450,250)
#         self.list.setSelectionMode(QtGui.QAbstractItemView.MultiSelection)
#         done = QtGui.QPushButton('done')
#         done.clicked.connect(lambda: self.exit_list(parent))
#         layout.addWidget(done,8,0,1,1)

#     def exit_list(self, parent):
#         parent.filelist = []
#         items = self.list.selectedItems()
#         for i in range(len(items)):
#             parent.filelist.append(str(self.list.selectedItems()[i].text()))
#         self.accept()

# class TextChooser(QtWidgets.QDialog):
#     def __init__(self,parent=None):
#         super(TextChooser, self).__init__(parent)
#         self.setGeometry(300,300,350,100)
#         self.setWindowTitle('folder path')
#         self.win = QtWidgets.QWidget(self)
#         layout = QtWidgets.QGridLayout()
#         self.win.setLayout(layout)
#         self.qedit = QtWidgets.QLineEdit('')
#         layout.addWidget(QtWidgets.QLabel('folder name (does not have to exist yet)'),0,0,1,3)
#         layout.addWidget(self.qedit,1,0,1,3)
#         done = QtGui.QPushButton('OK')
#         done.clicked.connect(self.exit)
#         layout.addWidget(done,2,1,1,1)

#     def exit(self):
#         self.folder = self.qedit.text()
#         self.accept()


# class RGBRadioButtons(QtGui.QButtonGroup):
#     def __init__(self, parent=None, row=0, col=0):
#         super(RGBRadioButtons, self).__init__()
#         parent.color = 0
#         self.parent = parent
#         self.bstr = ["image", "flowsX", "flowsY", "flowsZ", "cellprob"]
#         #self.buttons = QtGui.QButtonGroup()
#         self.dropdown = []
#         for b in range(len(self.bstr)):
#             button = QtGui.QRadioButton(self.bstr[b])
#             button.setStyleSheet('color: white;')
#             if b==0:
#                 button.setChecked(True)
#             self.addButton(button, b)
#             button.toggled.connect(lambda: self.btnpress(parent))
#             self.parent.l0.addWidget(button, row+b,col,1,1)
#         self.setExclusive(True)
#         #self.buttons.

#     def btnpress(self, parent):
#        b = self.checkedId()
#        self.parent.view = b
#        if self.parent.loaded:
#            self.parent.update_plot()


# class ViewBoxNoRightDrag(pg.ViewBox):
#     def __init__(self, parent=None, border=None, lockAspect=False, enableMouse=True, invertY=False, enableMenu=True, name=None, invertX=False):
#         pg.ViewBox.__init__(self, parent, border, lockAspect, enableMouse,
#                             invertY, enableMenu, name, invertX)

#     def mouseDragEvent(self, ev, axis=None):
#         ## if axis is specified, event will only affect that axis.
#         ev.accept()  ## we accept all buttons

#         pos = ev.pos()
#         lastPos = ev.lastPos()
#         dif = pos - lastPos
#         dif = dif * -1

#         ## Ignore axes if mouse is disabled
#         mouseEnabled = np.array(self.state['mouseEnabled'], dtype=np.float)
#         mask = mouseEnabled.copy()
#         if axis is not None:
#             mask[1-axis] = 0.0

#         ## Scale or translate based on mouse button
#         if ev.button() & (QtCore.Qt.LeftButton | QtCore.Qt.MidButton):
#             if self.state['mouseMode'] == pg.ViewBox.RectMode:
#                 if ev.isFinish():  ## This is the final move in the drag; change the view scale now
#                     #print "finish"
#                     self.rbScaleBox.hide()
#                     ax = QtCore.QRectF(Point(ev.buttonDownPos(ev.button())), Point(pos))
#                     ax = self.childGroup.mapRectFromParent(ax)
#                     self.showAxRect(ax)
#                     self.axHistoryPointer += 1
#                     self.axHistory = self.axHistory[:self.axHistoryPointer] + [ax]
#                 else:
#                     ## update shape of scale box
#                     self.updateScaleBox(ev.buttonDownPos(), ev.pos())
#             else:
#                 tr = dif*mask
#                 tr = self.mapToView(tr) - self.mapToView(Point(0,0))
#                 x = tr.x() if mask[0] == 1 else None
#                 y = tr.y() if mask[1] == 1 else None

#                 self._resetTarget()
#                 if x is not None or y is not None:
#                     self.translateBy(x=x, y=y)
#                 self.sigRangeChangedManually.emit(self.state['mouseEnabled'])

# class ImageDraw(pg.ImageItem):
#     """
#     **Bases:** :class:`GraphicsObject <pyqtgraph.GraphicsObject>`
#     GraphicsObject displaying an image. Optimized for rapid update (ie video display).
#     This item displays either a 2D numpy array (height, width) or
#     a 3D array (height, width, RGBa). This array is optionally scaled (see
#     :func:`setLevels <pyqtgraph.ImageItem.setLevels>`) and/or colored
#     with a lookup table (see :func:`setLookupTable <pyqtgraph.ImageItem.setLookupTable>`)
#     before being displayed.
#     ImageItem is frequently used in conjunction with
#     :class:`HistogramLUTItem <pyqtgraph.HistogramLUTItem>` or
#     :class:`HistogramLUTWidget <pyqtgraph.HistogramLUTWidget>` to provide a GUI
#     for controlling the levels and lookup table used to display the image.
#     """

#     sigImageChanged = QtCore.Signal()

#     def __init__(self, image=None, viewbox=None, parent=None, **kargs):
#         super(ImageDraw, self).__init__()
#         #self.image=None
#         #self.viewbox=viewbox
#         self.levels = np.array([0,255])
#         self.lut = None
#         self.autoDownsample = False
#         self.axisOrder = 'row-major'
#         self.removable = False

#         self.parent = parent
#         #kernel[1,1] = 1
#         self.setDrawKernel(kernel_size=self.parent.brush_size)
#         self.parent.current_stroke = []
#         self.parent.in_stroke = False

#     def mouseClickEvent(self, ev):
#         if self.parent.masksOn:
#             if ev.button()==QtCore.Qt.RightButton and self.parent.loaded and self.parent.nmasks < 2:
#                 if not self.parent.in_stroke:
#                     ev.accept()
#                     self.parent.in_stroke = True
#                     self.create_start(ev.pos())
#                     self.parent.stroke_appended = False
#                     self.drawAt(ev.pos(), ev)
#                 else:
#                     ev.accept()
#                     self.end_stroke()
#                     self.parent.in_stroke = False
#             else:
#                 ev.ignore()
#                 return
#         else:
#             ev.ignore()
#             return


#     def mouseDragEvent(self, ev):
#         ev.ignore()
#         return

#     def hoverEvent(self, ev):
#         QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CrossCursor)
#         if self.parent.in_stroke:
#             if self.parent.in_stroke:
#                 # continue stroke if not at start
#                 self.drawAt(ev.pos())
#                 if self.is_at_start(ev.pos()):
#                     self.end_stroke()
#                     self.parent.in_stroke = False
#         else:
#             ev.acceptClicks(QtCore.Qt.RightButton)
#             #ev.acceptClicks(QtCore.Qt.LeftButton)

#     def create_start(self, pos):
#         self.scatter = pg.ScatterPlotItem([pos.x()], [pos.y()], pxMode=False,
#                                         pen=pg.mkPen(color=(255,0,0), width=self.parent.brush_size),
#                                         size=max(3*2, self.parent.brush_size*1.8*2), brush=None)
#         self.parent.p0.addItem(self.scatter)

#     def is_at_start(self, pos):
#         thresh_out = max(6, self.parent.brush_size*3)
#         thresh_in = max(3, self.parent.brush_size*1.8)
#         # first check if you ever left the start
#         if len(self.parent.current_stroke) > 3:
#             stroke = np.array(self.parent.current_stroke)
#             dist = (((stroke[1:] - stroke[:1][np.newaxis,:,:])**2).sum(axis=-1))**0.5
#             dist = dist.flatten()
#             #print(dist)
#             has_left = (dist > thresh_out).nonzero()[0]
#             if len(has_left) > 0:
#                 first_left = np.sort(has_left)[0]
#                 has_returned = (dist[max(4,first_left+1):] < thresh_in).sum()
#                 if has_returned > 0:
#                     return True
#                 else:
#                     return False
#             else:
#                 return False

#     def end_stroke(self):
#         self.parent.p0.removeItem(self.scatter)
#         if not self.parent.stroke_appended:
#             self.parent.stroke = np.array(self.parent.current_stroke)
#             self.parent.current_stroke = []
#             self.parent.stroke_appended = True
#             ioutline = self.parent.stroke[:,-1]==1
#             self.parent.point_set = list(self.parent.stroke[ioutline])
#             self.parent.add_set()

#     def tabletEvent(self, ev):
#         pass
#         #print(ev.device())
#         #print(ev.pointerType())
#         #print(ev.pressure())

#     def drawAt(self, pos, ev=None):
#         mask = self.greenmask
#         set = self.parent.current_point_set
#         stroke = self.parent.current_stroke
#         pos = [int(pos.y()), int(pos.x())]
#         dk = self.drawKernel
#         kc = self.drawKernelCenter
#         sx = [0,dk.shape[0]]
#         sy = [0,dk.shape[1]]
#         tx = [pos[0] - kc[0], pos[0] - kc[0]+ dk.shape[0]]
#         ty = [pos[1] - kc[1], pos[1] - kc[1]+ dk.shape[1]]
#         kcent = kc.copy()
#         if tx[0]<=0:
#             sx[0] = 0
#             sx[1] = kc[0] + 1
#             tx    = sx
#             kcent[0] = 0
#         if ty[0]<=0:
#             sy[0] = 0
#             sy[1] = kc[1] + 1
#             ty    = sy
#             kcent[1] = 0
#         if tx[1] >= self.parent.Ly-1:
#             sx[0] = dk.shape[0] - kc[0] - 1
#             sx[1] = dk.shape[0]
#             tx[0] = self.parent.Ly - kc[0] - 1
#             tx[1] = self.parent.Ly
#             kcent[0] = tx[1]-tx[0]-1
#         if ty[1] >= self.parent.Lx-1:
#             sy[0] = dk.shape[1] - kc[1] - 1
#             sy[1] = dk.shape[1]
#             ty[0] = self.parent.Lx - kc[1] - 1
#             ty[1] = self.parent.Lx
#             kcent[1] = ty[1]-ty[0]-1


#         ts = (slice(tx[0],tx[1]), slice(ty[0],ty[1]))
#         ss = (slice(sx[0],sx[1]), slice(sy[0],sy[1]))
#         self.image[ts] = mask[ss]

#         for ky,y in enumerate(np.arange(ty[0], ty[1], 1, int)):
#             for kx,x in enumerate(np.arange(tx[0], tx[1], 1, int)):
#                 iscent = np.logical_and(kx==kcent[0], ky==kcent[1])
#                 stroke.append([x, y, iscent])
#         self.updateImage()

#     def setDrawKernel(self, kernel_size=3):
#         bs = kernel_size
#         kernel = np.ones((bs,bs), np.uint8)
#         self.drawKernel = kernel
#         self.drawKernelCenter = [int(np.floor(kernel.shape[0]/2)),
#                                  int(np.floor(kernel.shape[1]/2))]
#         onmask = 255 * kernel[:,:,np.newaxis]
#         offmask = np.zeros((bs,bs,1))
#         opamask = 100 * kernel[:,:,np.newaxis]
#         self.redmask = np.concatenate((onmask,offmask,offmask,onmask), axis=-1)
#         self.greenmask = np.concatenate((onmask,offmask,onmask,opamask), axis=-1)


# class RangeSlider(QtWidgets.QSlider):
#     """ A slider for ranges.

#         This class provides a dual-slider for ranges, where there is a defined
#         maximum and minimum, as is a normal slider, but instead of having a
#         single slider value, there are 2 slider values.

#         This class emits the same signals as the QSlider base class, with the
#         exception of valueChanged

#         Found this slider here: https://www.mail-archive.com/pyqt@riverbankcomputing.com/msg22889.html
#         and modified it
#     """
#     def __init__(self, parent=None, *args):
#         super(RangeSlider, self).__init__(*args)

#         self._low = self.minimum()
#         self._high = self.maximum()

#         self.pressed_control = QtGui.QStyle.SC_None
#         self.hover_control = QtGui.QStyle.SC_None
#         self.click_offset = 0

#         self.setOrientation(QtCore.Qt.Vertical)
#         self.setTickPosition(QtWidgets.QSlider.TicksRight)
#         self.setStyleSheet(\
#                 "QSlider::handle:horizontal {\
#                 background-color: white;\
#                 border: 1px solid #5c5c5c;\
#                 border-radius: 0px;\
#                 border-color: black;\
#                 height: 8px;\
#                 width: 6px;\
#                 margin: -8px 2; \
#                 }")


#         #self.opt = QtGui.QStyleOptionSlider()
#         #self.opt.orientation=QtCore.Qt.Vertical
#         #self.initStyleOption(self.opt)
#         # 0 for the low, 1 for the high, -1 for both
#         self.active_slider = 0
#         self.parent = parent


#     def level_change(self):
#         if self.parent is not None:
#             if self.parent.loaded:
#                 self.parent.saturation = [self._low, self._high]
#                 self.parent.update_plot()

#     def low(self):
#         return self._low

#     def setLow(self, low):
#         self._low = low
#         self.update()

#     def high(self):
#         return self._high

#     def setHigh(self, high):
#         self._high = high
#         self.update()

#     def paintEvent(self, event):
#         # based on http://qt.gitorious.org/qt/qt/blobs/master/src/gui/widgets/qslider.cpp
#         painter = QtGui.QPainter(self)
#         style = QtGui.QApplication.style()

#         for i, value in enumerate([self._low, self._high]):
#             opt = QtGui.QStyleOptionSlider()
#             self.initStyleOption(opt)

#             # Only draw the groove for the first slider so it doesn't get drawn
#             # on top of the existing ones every time
#             if i == 0:
#                 opt.subControls = QtGui.QStyle.SC_SliderHandle#QtGui.QStyle.SC_SliderGroove | QtGui.QStyle.SC_SliderHandle
#             else:
#                 opt.subControls = QtGui.QStyle.SC_SliderHandle

#             if self.tickPosition() != self.NoTicks:
#                 opt.subControls |= QtGui.QStyle.SC_SliderTickmarks

#             if self.pressed_control:
#                 opt.activeSubControls = self.pressed_control
#                 opt.state |= QtGui.QStyle.State_Sunken
#             else:
#                 opt.activeSubControls = self.hover_control

#             opt.sliderPosition = value
#             opt.sliderValue = value
#             style.drawComplexControl(QtGui.QStyle.CC_Slider, opt, painter, self)


#     def mousePressEvent(self, event):
#         event.accept()

#         style = QtGui.QApplication.style()
#         button = event.button()
#         # In a normal slider control, when the user clicks on a point in the
#         # slider's total range, but not on the slider part of the control the
#         # control would jump the slider value to where the user clicked.
#         # For this control, clicks which are not direct hits will slide both
#         # slider parts
#         if button:
#             opt = QtGui.QStyleOptionSlider()
#             self.initStyleOption(opt)

#             self.active_slider = -1

#             for i, value in enumerate([self._low, self._high]):
#                 opt.sliderPosition = value
#                 hit = style.hitTestComplexControl(style.CC_Slider, opt, event.pos(), self)
#                 if hit == style.SC_SliderHandle:
#                     self.active_slider = i
#                     self.pressed_control = hit

#                     self.triggerAction(self.SliderMove)
#                     self.setRepeatAction(self.SliderNoAction)
#                     self.setSliderDown(True)

#                     break

#             if self.active_slider < 0:
#                 self.pressed_control = QtGui.QStyle.SC_SliderHandle
#                 self.click_offset = self.__pixelPosToRangeValue(self.__pick(event.pos()))
#                 self.triggerAction(self.SliderMove)
#                 self.setRepeatAction(self.SliderNoAction)
#         else:
#             event.ignore()

#     def mouseMoveEvent(self, event):
#         if self.pressed_control != QtGui.QStyle.SC_SliderHandle:
#             event.ignore()
#             return

#         event.accept()
#         new_pos = self.__pixelPosToRangeValue(self.__pick(event.pos()))
#         opt = QtGui.QStyleOptionSlider()
#         self.initStyleOption(opt)

#         if self.active_slider < 0:
#             offset = new_pos - self.click_offset
#             self._high += offset
#             self._low += offset
#             if self._low < self.minimum():
#                 diff = self.minimum() - self._low
#                 self._low += diff
#                 self._high += diff
#             if self._high > self.maximum():
#                 diff = self.maximum() - self._high
#                 self._low += diff
#                 self._high += diff
#         elif self.active_slider == 0:
#             if new_pos >= self._high:
#                 new_pos = self._high - 1
#             self._low = new_pos
#         else:
#             if new_pos <= self._low:
#                 new_pos = self._low + 1
#             self._high = new_pos

#         self.click_offset = new_pos
#         self.update()

#     def mouseReleaseEvent(self, event):
#         self.level_change()

#     def __pick(self, pt):
#         if self.orientation() == QtCore.Qt.Horizontal:
#             return pt.x()
#         else:
#             return pt.y()


#     def __pixelPosToRangeValue(self, pos):
#         opt = QtGui.QStyleOptionSlider()
#         self.initStyleOption(opt)
#         style = QtGui.QApplication.style()

#         gr = style.subControlRect(style.CC_Slider, opt, style.SC_SliderGroove, self)
#         sr = style.subControlRect(style.CC_Slider, opt, style.SC_SliderHandle, self)

#         if self.orientation() == QtCore.Qt.Horizontal:
#             slider_length = sr.width()
#             slider_min = gr.x()
#             slider_max = gr.right() - slider_length + 1
#         else:
#             slider_length = sr.height()
#             slider_min = gr.y()
#             slider_max = gr.bottom() - slider_length + 1

#         return style.sliderValueFromPosition(self.minimum(), self.maximum(),
#                                              pos-slider_min, slider_max-slider_min,
#                                              opt.upsideDown)
    
# if __name__=='__main__':

#     # app = QtWidgets.QApplication(sys.argv)
#     # # build_dark_palette(app)
#     # # window = TrialAverageWindow(app)
#     # # window.show()
#     # sys.exit(app.exec_())
