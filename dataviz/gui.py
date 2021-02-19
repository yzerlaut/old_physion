import sys, time, tempfile, os, pathlib, datetime, string, pynwb
import numpy as np
from PyQt5 import QtGui, QtWidgets, QtCore
import pyqtgraph as pg
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import day_folder, generate_filename_path, list_dayfolder
from assembling.dataset import Dataset, MODALITIES
from dataviz import guiparts, plots
from analysis.trial_averaging import TrialAverageWindow
from analysis.behavioral_modulation import BehavioralModWindow
from analysis.read_NWB import read as read_NWB

settings = {
    'window_size':(1000,600),
    # raw data plot settings
    'increase-factor':2., # so "Calcium" is twice "Eletrophy", that is twice "Pupil",..  "Locomotion"
    'blank-space':0.1, # so "Calcium" is twice "Eletrophy", that is twice "Pupil",..  "Locomotion"
    'colors':{'Screen':(100, 100, 100, 255),#'grey',
              'Locomotion':(255,255,255,255),#'white',
              'Whisking':(255,0,255,255),#'white',
              'Pupil':(255,0,0,255),#'red',
              'Electrophy':(100,100,255,255),#'blue',
              'CaImaging':(0,255,0,255)},#'green'},
    # general settings
    'Npoints':500}

pg.setConfigOptions(imageAxisOrder='row-major')


class MainWindow(guiparts.NewWindow):
    
    def __init__(self, app,
                 args=None,
                 parent=None,
                 raw_data_visualization=False,
                 fullscreen=False):

        self.app = app
        self.settings = settings
        self.raw_data_visualization = raw_data_visualization
        self.no_subsampling = False
        
        super(MainWindow, self).__init__(i=0,
            title='Data Visualization -- Physiology of Visual Circuits')

        # play button
        self.updateTimer = QtCore.QTimer()
        self.updateTimer.timeout.connect(self.next_frame)
        
        guiparts.load_config1(self)
        self.windowTA, self.windowBM = None, None # sub-windows

        if args is not None:
            self.root_datafolder = args.root_datafolder
        else:
            self.root_datafolder = os.path.join(os.path.expanduser('~'), 'DATA')

        self.time, self.io, self.roiIndices, self.tzoom = 0, None, [], [0,50]
        self.CaImaging_bg_key = 'meanImg'
        self.CaImaging_key = 'Fluorescence'
        self.check_data_folder()
        
        self.minView = False
        self.showwindow()

        # ----------------------------------
        # ========= for debugging ==========
        # ----------------------------------
        # self.datafolder = '/home/yann/DATA/2020_11_04/01-02-03/'
        # date = 
        # date = self.cal.setSelectedDate(datetime.date(2020, 11, 4))
        # self.pick_date()
        # self.preload_datafolder(self.datafolder)
        # self.dbox.setCurrentIndex(1)
        # self.pick_datafolder()
        # self.display_quantities()
        
        # filename = os.path.join(os.path.expanduser('~'), 'DATA', '2020_11_12', '2020_11_12-18-29-31.FULL.nwb')
        # filename = os.path.join(os.path.expanduser('~'), 'DATA', '2020_11_12', '2020_11_12-17-30-19.FULL.nwb')
        # filename = os.path.join('D:', '2021_01_20', '16-42-18', '2021_01_20-16-42-18.FULL.nwb')
        # self.load_file(filename)
        # print(self.nwbfile.acquisition)
        # plots.raw_data_plot(self, self.tzoom)

    def try_to_find_time_extents(self):
        self.tlim, safety_counter = None, 0
        while (self.tlim is None) and (safety_counter<10):
            for key in self.nwbfile.acquisition:
                try:
                    self.tlim = [self.nwbfile.acquisition[key].starting_time,
                                 self.nwbfile.acquisition[key].starting_time+\
                                 self.nwbfile.acquisition[key].data.shape[0]/self.nwbfile.acquisition[key].rate]
                except BaseException as be:
                    pass
        if self.tlim is None:
            self.tlim = [0, 50] # bad for movies
        
    def open_file(self):
        
        filename, _ = QtGui.QFileDialog.getOpenFileName(self,
                     "Open Multimodal Experimental Recording (NWB file) ",
                        os.path.join(os.path.expanduser('~'),'DATA'),
                            filter="*.nwb")
        
        if filename!='':
            self.reset()
            self.datafile=filename
            self.load_file(self.datafile)
            plots.raw_data_plot(self, self.tzoom)
        else:
            print('"%s" filename not recognized ! ')

            
    def reset(self):
        self.windowTA, self.windowBM = None, None # sub-windows
        self.no_subsampling = False
        self.plot.clear()
        self.pScreenimg.clear()
        self.pFaceimg.clear()
        self.pCaimg.clear()
        self.pPupil.clear()
        self.pPupilimg.clear()
        self.roiIndices = None


        
    def select_ROI(self):
        """ see select ROI above """
        self.roiIndices = self.select_ROI_from_pick()
        plots.raw_data_plot(self, self.tzoom, with_roi=True)

            
    def load_file(self, filename):

        read_NWB(self, filename, verbose=True) # see ../analysis/read_NWB.py

        self.tzoom = self.tlim
        self.notes.setText(self.description)

        self.cal.setSelectedDate(self.nwbfile.session_start_time.date())
        self.dbox.clear()
        self.dbox.addItem(self.df_name)
        self.dbox.setCurrentIndex(0)
        self.sbox.clear()
        self.sbox.addItem(self.nwbfile.subject.description)
        self.sbox.setCurrentIndex(0)
        self.pbox.setCurrentIndex(1)
        
        if 'ophys' in self.nwbfile.processing:
            self.roiPick.setText(' [select ROI] (%i-%i)' % (0, len(self.validROI_indices)-1))
            
        if os.path.isfile(filename.replace('.nwb', '.pupil.npy')):
            self.pupil_data = np.load(filename.replace('.nwb', '.pupil.npy'),
                                      allow_pickle=True).item()
        else:
            self.pupil_data = None
            
        
    def check_data_folder(self):
        
        print('inspecting data folder [...]')

        self.highlight_format = QtGui.QTextCharFormat()
        self.highlight_format.setBackground(self.cal.palette().brush(QtGui.QPalette.Link))
        self.highlight_format.setForeground(self.cal.palette().color(QtGui.QPalette.BrightText))

        date = datetime.date(2020, 9, 1)
        while date!=(datetime.date.today()+datetime.timedelta(30)):
            if os.path.isdir(os.path.join(self.root_datafolder, date.strftime("%Y_%m_%d"))):
                self.cal.setDateTextFormat(QtCore.QDate(date), self.highlight_format)
            date = date+datetime.timedelta(1)
        date = self.cal.setSelectedDate(date+datetime.timedelta(1))
        
    def pick_date(self):
        date = self.cal.selectedDate()
        self.day = '%s_%02d_%02d' % (date.year(), date.month(), date.day())
        for i in string.digits:
            self.day = self.day.replace('_%s_' % i, '_0%s_' % i)
        self.day_folder = os.path.join(self.root_datafolder,self.day)
        if os.path.isdir(self.day_folder):
            self.list_protocol_per_day = list_dayfolder(self.day_folder)
        else:
            self.list_protocol_per_day = []
        self.update_df_names()


    def update_df_names(self):
        self.dbox.clear()
        if not self.raw_data_visualization:
            self.pbox.clear()
        self.plot.clear()
        self.pScreenimg.setImage(np.ones((10,12))*50)
        self.pFaceimg.setImage(np.ones((10,12))*50)
        self.pPupilimg.setImage(np.ones((10,12))*50)
        self.pCaimg.setImage(np.ones((50,50))*100)
        if len(self.list_protocol_per_day)>0:
            self.dbox.addItem(' ...' +70*' '+'(select a data-folder) ')
            for fn in self.list_protocol_per_day:
                self.dbox.addItem(self.preload_datafolder(fn))

                
    def preload_datafolder(self, fn):
        output = '   '+fn.split(os.path.sep)[-1].replace('-', ':')+' --------- '
        try:
            info = np.load(os.path.join(self.day_folder,fn,'metadata.npy'),
                           allow_pickle=True).item()
            # output += str(info['Stimulus'])
            output += str(info['protocol'])
        except Exception:
            print('No metadata found')
        return output

    def pick_datafolder(self):
        if not self.raw_data_visualization:
            self.pbox.clear()
        self.plot.clear()
        i = self.dbox.currentIndex()
        if i>0:
            self.datafolder = self.list_protocol_per_day[i-1]
            self.datafile=os.path.join(self.datafolder, '%s-%s.nwb' % (self.datafolder.split(os.path.sep)[-2],self.datafolder.split(os.path.sep)[-1]))
            self.load_file(self.datafile)
            plots.raw_data_plot(self, self.tzoom)
        else:
            self.metadata = None
            self.notes.setText(20*'-'+5*'\n')

    def add_datafolder_annotation(self):
        info = 20*'-'+'\n'

        if self.metadata['protocol']=='None':
            self.notes.setText('\nNo visual stimulation')
        else:
            for key in self.metadata:
                if (key[:2]=='N-') and (key!='N-repeat') and (self.metadata[key]>1): # meaning it was varied
                    info += '%s=%i (%.1f to %.1f)\n' % (key, self.metadata[key], self.metadata[key[2:]+'-1'], self.metadata[key[2:]+'-2'])
            info += '%s=%i' % ('N-repeat', self.metadata['N-repeat'])
            self.notes.setText(info)

    def display_quantities(self,
                           force=False,
                           plot_update=True,
                           with_images=False,
                           with_scatter=False):
        """
        # IMPLEMENT OTHER ANALYSIS HERE
        """

        if self.pbox.currentText()=='-> Trial-average' and (self.windowTA is None):
            self.windowTA = TrialAverageWindow(parent=self)
            self.windowTA.show()
        elif self.pbox.currentText()=='-> Behavioral-modulation' and (self.windowBM is None):
            self.window3 = BehavioralModWindow(parent=self)
            self.window3.show()
        else:
            self.plot.clear()
            plots.raw_data_plot(self, self.tzoom,
                                plot_update=plot_update,
                                with_images=with_images,
                                with_scatter=with_scatter)
        self.pbox.setCurrentIndex(1)


    def back_to_initial_view(self):
        self.time = 0
        self.tzoom = self.tlim
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
        for key, val in self.metadata.items():
            print('- %s : ' % key, val)
            
    def change_settings(self):
        pass
    
    def quit(self):
        if self.io is not None:
            self.io.close()
        sys.exit()


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
    parser.add_argument('-v', "--visualization", action="store_true")
    args = parser.parse_args()
    app = QtWidgets.QApplication(sys.argv)
    build_dark_palette(app)
    main = MainWindow(app,
                      args=args,
                      raw_data_visualization=args.visualization)
    sys.exit(app.exec_())

