import sys, time, tempfile, os, pathlib, json, subprocess, datetime, string
import numpy as np
from PyQt5 import QtGui, QtWidgets, QtCore
import pyqtgraph as pg
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import day_folder, generate_filename_path, save_dict, load_dict, list_dayfolder
from assembling.dataset import Dataset, MODALITIES
from analysis import guiparts, plots

settings = {
    'window_size':(1000,600),
    # raw data plot settings
    'increase-factor':2., # so "Calcium" is twice "Eletrophy", that is twice "Pupil",..  "Locomotion"
    'blank-space':0.1, # so "Calcium" is twice "Eletrophy", that is twice "Pupil",..  "Locomotion"
    'colors':{'Screen':(100, 100, 100, 255),#'grey',
              'Locomotion':(255,255,255,255),#'white',
              'Pupil':(255,0,0,255),#'red',
              'Electrophy':(100,100,255,255),#'blue',
              'Calcium':(0,255,0,255)},#'green'},
    # general settings
    'Npoints':600}

class MainWindow(QtWidgets.QMainWindow):
    
    def __init__(self, parent=None,
                 saturation=100,
                 fullscreen=False):

        self.settings = settings
        
        super(MainWindow, self).__init__()

        # adding a "quit" keyboard shortcut
        self.quitSc = QtWidgets.QShortcut(QtGui.QKeySequence('Q'), self) # or 'Ctrl+Q'
        self.quitSc.activated.connect(self.quit)
        self.refreshSc = QtWidgets.QShortcut(QtGui.QKeySequence('R'), self) # or 'Ctrl+Q'
        self.refreshSc.activated.connect(self.refresh)
        self.maxSc = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+M'), self)
        self.maxSc.activated.connect(self.showwindow)

        ####################################################
        # BASIC style config
        self.setWindowTitle('Analysis Program -- Physiology of Visual Circuits')
        pg.setConfigOptions(imageAxisOrder='row-major')

        # default (small) geometry
        self.setGeometry(100,100,*self.settings['window_size'])

        self.statusBar = QtWidgets.QStatusBar()
        # self.setStatusBar(self.statusBar)
        # self.statusBar.showMessage('Pick a date, a data-folder and a visualization/analysis')
        
        guiparts.load_config1(self)

        self.data_folder = os.path.join(os.path.expanduser('~'), 'DATA')
        for mod in MODALITIES:
            setattr(self, mod, None)
            
        self.time = 0
        self.tzoom = [0, 10]
        self.check_data_folder()
        
        self.minView = False
        self.showwindow()

        # # for debugging
        # self.day_folder = '/home/yann/DATA/2020_10_07/'
        # self.datafolder = '/home/yann/DATA/2020_10_07/16-02-19/'
        # self.preload_datafolder(self.datafolder)
        # self.load_data()

        # self.display_quantities(force=True)
        
    def check_data_folder(self):
        
        print('inspecting data folder [...]')

        self.highlight_format = QtGui.QTextCharFormat()
        self.highlight_format.setBackground(self.cal.palette().brush(QtGui.QPalette.Link))
        self.highlight_format.setForeground(self.cal.palette().color(QtGui.QPalette.BrightText))

        date = datetime.date(2020, 8, 1)
        while date!=(datetime.date.today()+datetime.timedelta(30)):
            if os.path.isdir(os.path.join(self.data_folder, date.strftime("%Y_%m_%d"))):
                self.cal.setDateTextFormat(QtCore.QDate(date), self.highlight_format)
            date = date+datetime.timedelta(1)
        
    def pick_date(self):
        date = self.cal.selectedDate()
        self.day = '%s_%02d_%02d' % (date.year(), date.month(), date.day())
        for i in string.digits:
            self.day = self.day.replace('_%s_' % i, '_0%s_' % i)
        self.day_folder = os.path.join(self.data_folder,self.day)
        if os.path.isdir(self.day_folder):
            self.list_protocol_per_day = list_dayfolder(self.day_folder)
        else:
            self.list_protocol_per_day = []
        self.update_df_names()

    def update_df_names(self):
        self.dbox.clear()
        self.pbox.clear()
        self.plot.clear()
        self.win1.clear()
        self.win2.clear()
        if len(self.list_protocol_per_day)>0:
            self.dbox.addItem(' ...' +70*' '+'(select a data-folder) ')
            for fn in self.list_protocol_per_day:
                self.dbox.addItem(self.preload_datafolder(fn))

    def preload_datafolder(self, fn):
        output = '   '+fn.split(os.path.sep)[-1].replace('-', ':')+' --------- '
        try:
            info = np.load(os.path.join(self.day_folder,fn,'metadata.npy'),
                           allow_pickle=True).item()
            output += str(info['Stimulus'])
        except Exception:
            print('No metadata found')
        return output

    def load_data(self):

         # see assembling/dataset.py
        dataset = Dataset(self.datafolder,
                          modalities=MODALITIES)
        
        for key in MODALITIES:
            setattr(self, key, getattr(dataset, key))
        setattr(self, 'metadata', dataset.metadata)

        try:
            self.tzoom = [0, self.metadata['time_start'][-1]+self.metadata['presentation-duration']]
        except KeyError:
            pass
            
        self.time = 0
        
    def pick_datafolder(self):
        self.pbox.clear()
        self.plot.clear()
        i = self.dbox.currentIndex()
        if i>0:
            self.datafolder = self.list_protocol_per_day[i-1]
            self.metadata = np.load(os.path.join(self.datafolder,'metadata.npy'),
                                    allow_pickle=True).item()
            self.add_datafolder_annotation()
            self.pbox.addItem('...       (select a visualization/analysis)')
            self.pbox.addItem('-> Show Raw Data')
            self.load_data()
        else:
            self.metadata = None
            self.notes.setText(63*'-'+5*'\n')

    def add_datafolder_annotation(self):
        info = 63*'-'+'\n'
        print(self.metadata)
        for key in self.metadata:
            if (key[:2]=='N-') and (key!='N-repeat') and (self.metadata[key]>1): # meaning it was varied
                info += '%s=%i (%.1f to %.1f)\n' % (key, self.metadata[key], self.metadata[key[2:]+'-1'], self.metadata[key[2:]+'-2'])
        info += '%s=%i' % ('N-repeat', self.metadata['N-repeat'])
        self.notes.setText(info)

    def display_quantities(self, force=False):
        self.statusBar.showMessage('Plotting quantities [...]')
        if self.pbox.currentIndex()==1 or force:
            plots.raw_data_plot(self, self.tzoom)
            plots.update_images(self, self.time)
        self.statusBar.showMessage('')

    def back_to_initial_view(self):
        self.tzoom = [self.Screen['times'][0], self.Screen['times'][-1]]
        self.display_quantities()
        
    def play(self):
        pass

    def pause(self):
        pass

    def refresh(self):
        self.plot.clear()
        self.tzoom = self.plot.getAxis('bottom').range
        self.display_quantities()
        
    
    def update_frame(self):
        pass

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

    def change_settings(self):
        pass
    
    def quit(self):
        sys.exit()


def run(app, parent=None):
    guiparts.build_dark_palette(app)
    return MainWindow(app)
    
if __name__=='__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = run(app)
    sys.exit(app.exec_())
