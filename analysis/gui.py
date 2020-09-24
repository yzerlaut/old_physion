import sys, time, tempfile, os, pathlib, json, subprocess, datetime, string
import numpy as np
from PyQt5 import QtGui, QtWidgets, QtCore
import pyqtgraph as pg
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import day_folder, generate_filename_path, save_dict, load_dict, list_dayfolder
from assembling.fetching import Dataset
from analysis import guiparts, plots

settings = {
    'window_size':(1000,600),
    # raw data plot settings
    'increase-factor':2., # so "Calcium" is twice "Eletrophy", that is twice "Pupil",..  "Locomotion"
    'blank-space':0.1, # so "Calcium" is twice "Eletrophy", that is twice "Pupil",..  "Locomotion"
    'colors':{'Screen':np.ones(3)*255,
              'Locomotion':np.ones(3)*100,
              'Pupil':(255, 70, 70),
              'Electrophy':(70, 70, 255),
              'Calcium':(70, 255, 70)},
    # general settings
    'Npoints':600}

class MasterWindow(QtWidgets.QMainWindow):
    
    def __init__(self, app,
                 saturation=100,
                 fullscreen=False):

        self.settings = settings
        
        guiparts.build_dark_palette(app)
        
        super(MasterWindow, self).__init__()

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
        self.setGeometry(200,200,*self.settings['window_size'])

        self.statusBar = QtWidgets.QStatusBar()
        # self.setStatusBar(self.statusBar)
        # self.statusBar.showMessage('Pick a date, a data-folder and a visualization/analysis')
        
        guiparts.load_config1(self)

        self.data_folder = os.path.join(os.path.expanduser('~'), 'DATA')
        self.Screen, self.Locomotion, self.Pupil, self.Calcium,\
            self.Electrophy, self.tzoom  = None, None, None, None, None, [0,1e3]
        self.check_data_folder()
        
        self.minView = False
        self.showwindow()

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

        dataset = Dataset(self.datafolder) # see assembling/fetching.py

        for key in ['Screen', 'Locomotion', 'Electrophy', 'Pupil']:
            setattr(self, key) = getattr(dataset, key)
            
        self.tzoom = [self.Screen['times'][0],self.Screen['times'][-1]]
        self.time = self.Screen['times'][0]

        
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
        for key in self.metadata:
            if (key[:2]=='N-') and (key!='N-repeat') and (self.metadata[key]>1): # meaning it was varied
                info += '%s=%i (%.1f to %.1f)\n' % (key, self.metadata[key],
                                  np.min(self.metadata[key[2:]]),np.max(self.metadata[key[2:]]))
                # if len(info)>30:
                #     info+='\n'
        info += '%s=%i' % ('N-repeat', self.metadata['N-repeat'])
        self.notes.setText(info)

    def display_quantities(self):
        if self.pbox.currentIndex()==1:
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

app = QtWidgets.QApplication(sys.argv)
main = MasterWindow(app)
sys.exit(app.exec_())
