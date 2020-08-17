import sys, time, tempfile, os, pathlib, json, subprocess, string, datetime
import threading # for the camera stream
import numpy as np
from PyQt5 import QtGui, QtWidgets, QtCore

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import *
from assembling.analysis import quick_data_view, analyze_data, last_datafile

def propose_preprocessing(folder):
    preprocessing_list = []
    lDir = os.listdir(folder)
    if 'FaceCamera-imgs' in lDir:
        preprocessing_list.append('FaceCamera -- Binary to Video files     [ ]')
        preprocessing_list.append('FaceCamera -- Pupil-ROI determination  [ ]')
        preprocessing_list.append('FaceCamera -- Pupil-Size Fluctuations  [ ]')
    return preprocessing_list
        
init_date = QtCore.QDate(2020, 8, 1) # experiments started after 1st of August 2020
init_date = datetime.date(2020, 8, 1) # experiments started after 1st of August 2020

class MasterWindow(QtWidgets.QMainWindow):
    
    def __init__(self, app,
                 parent=None):
        
        super(MasterWindow, self).__init__(parent)
        
        self.data_folder = get_data_folder()
        self.init_date = init_date
        
        self.setWindowTitle('Preprocessing Program - Physiology of Visual Circuits')
        self.setGeometry(150, 150, 480, 500)

        # buttons and functions
        LABELS = ["v) View data", "r) Launch Preprocessing", "q) Quit"]
        FUNCTIONS = [self.view_data, self.run, self.quit]
        
        mainMenu = self.menuBar()
        self.fileMenu = mainMenu.addMenu('')

        for func, label, bl, shift in zip(FUNCTIONS, LABELS,\
                                          [100, 180, 100], [30, 140, 330]):
            btn = QtWidgets.QPushButton(label, self)
            btn.clicked.connect(func)
            btn.setMinimumWidth(bl)
            btn.move(shift+30, 20)
            action = QtWidgets.QAction(label, self)
            action.setShortcut(label.split(')')[0])
            action.triggered.connect(func)
            self.fileMenu.addAction(action)

            
        self.dfl = QtWidgets.QLabel('Data-Folder (root): "%s"' % str(self.data_folder), self)
        self.dfl.setMinimumWidth(300)
        self.dfl.move(30, 70)
        dfb = QtWidgets.QPushButton('Set folder', self)
        dfb.clicked.connect(self.choose_data_folder)
        dfb.move(330, 70)

        self.cal = QtWidgets.QCalendarWidget(self)
        self.cal.move(70, 120)
        self.cal.setMinimumWidth(350)
        self.cal.setMinimumHeight(220)
        self.cal.setMinimumDate(QtCore.QDate(init_date))
        self.cal.setMaximumDate(QtCore.QDate.currentDate())
        self.cal.clicked.connect(self.pick_date)
        
        QtWidgets.QLabel('Protocol:', self).move(30, 380)
        self.pbox = QtWidgets.QComboBox(self)
        self.pbox.move(100, 380)
        self.pbox.setMinimumWidth(300)
        self.pbox.activated.connect(self.update_preprocessing)
        
        QtWidgets.QLabel('Preprocessing:', self).move(30, 420)
        self.ppbox = QtWidgets.QComboBox(self)
        self.ppbox.move(150, 420)
        self.ppbox.setMinimumWidth(300)
        
        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)

        self.check_data_folder()
        
        self.statusBar.showMessage('ready for preprocessing')
        self.show()

    def check_data_folder(self):
        
        self.statusBar.showMessage('inspecting data folder [...]')

        self.highlight_format = QtGui.QTextCharFormat()

        # self.highlight_format.setBackground(self.cal.palette().brush(QtGui.QPalette.Highlight))
        self.highlight_format.setBackground(self.cal.palette().brush(QtGui.QPalette.Button))
        self.highlight_format.setForeground(self.cal.palette().color(QtGui.QPalette.Mid))

        date = init_date
        while date!=(datetime.date.today()+datetime.timedelta(30)):
            if not os.path.isdir(os.path.join(self.data_folder, date.strftime("%Y_%m_%d"))):
                self.cal.setDateTextFormat(QtCore.QDate(date), self.highlight_format)
            date = date+datetime.timedelta(1)
        
        
    def pick_date(self):
        date = self.cal.selectedDate()
        self.day = '%s_%s_%s' % (date.year(), date.month(), date.day())
        for i in string.digits:
            self.day = self.day.replace('_%s_' % i, '_0%s_' % i)
        self.day_folder = os.path.join(self.data_folder,self.day)
        if os.path.isdir(self.day_folder):
            self.list_protocol_per_day = list_dayfolder(self.day_folder)
        else:
            self.list_protocol_per_day = []
        self.update_protocol_names()
        
            
    def update_protocol_names(self):
        self.pbox.clear()
        self.ppbox.clear()
        if len(self.list_protocol_per_day)>0:
            names = [' ... ']+\
                [fn.split(os.path.sep)[-1] for fn in self.list_protocol_per_day]
            for i, n in enumerate(names):
                self.pbox.addItem(n)

    def update_preprocessing(self):
        if self.pbox.currentIndex()>0:
            ppL = propose_preprocessing(self.list_protocol_per_day[self.pbox.currentIndex()-1])
            self.ppbox.clear()
            names = [' ...']+ppL
            for i, n in enumerate(names):
                self.ppbox.addItem(n)

        
    def run(self):
        self.statusBar.showMessage('Preprocessing running [...]')
        data, fig1 = quick_data_view(last_datafile(tempfile.gettempdir()), realign=True)
        _, fig2 = analyze_data(data=data)
        fig1.show()
        fig2.show()

    def choose_data_folder(self):
        self.pbox.clear()
        self.ppbox.clear()
        fd = str(QtWidgets.QFileDialog.getExistingDirectory(self,
                                                            "Select Root Data Folder", self.data_folder))
        if os.path.isdir(fd):
            self.data_folder = fd
            set_data_folder(fd)
            self.check_data_folder()
            self.dfl.setText('Data-Folder (root): "%s"' % str(self.data_folder))
        else:
            self.statusBar.showMessage('Invalid folder -> folder unchanged')
        
    def view_data(self):
        _, fig = quick_data_view(last_datafile(self.data_folder))
        fig.show()
        
    def load_data(self):
        pass
    
    def run(self):
        pass
    
    def quit(self):
        sys.exit()

if __name__ == '__main__':
    
    app = QtWidgets.QApplication(sys.argv)
    main = MasterWindow(app)
    sys.exit(app.exec_())
