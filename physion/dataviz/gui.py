import sys, time, tempfile, os, pathlib, datetime, string, pynwb
import numpy as np
from PyQt5 import QtGui, QtWidgets, QtCore
import pyqtgraph as pg
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import day_folder, generate_filename_path, list_dayfolder, get_files_with_extension
from assembling.dataset import Dataset, MODALITIES
from dataviz import guiparts, plots
from analysis.trial_averaging import TrialAverageWindow
from analysis.make_figures import FiguresWindow
from analysis.behavioral_modulation import BehavioralModWindow
from analysis.read_NWB import read as read_NWB
from analysis.read_NWB import read as read_NWB
from misc.folders import FOLDERS
from visual_stim.psychopy_code.stimuli import build_stim # we'll load it without psychopy

settings = {
    'window_size':(1000,600),
    # raw data plot settings
    'increase-factor':2., # so "Calcium" is twice "Eletrophy", that is twice "Pupil",..  "Locomotion"
    'blank-space':0.1, # so "Calcium" is twice "Eletrophy", that is twice "Pupil",..  "Locomotion"
    'colors':{'Screen':(100, 100, 100, 255),#'grey',
              'Locomotion':(255,255,255,255),#'white',
              'Whisking':(255,0,255,255),#'purple',
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
            title='Data Visualization')

        # play button
        self.updateTimer = QtCore.QTimer()
        self.updateTimer.timeout.connect(self.next_frame)
        
        guiparts.load_config1(self)
        self.fbox.addItems(FOLDERS.keys())
        self.windowTA, self.windowBM = None, None # sub-windows

        if args is not None:
            self.root_datafolder = args.root_datafolder
        else:
            self.root_datafolder = os.path.join(os.path.expanduser('~'), 'DATA')

        self.time, self.io, self.roiIndices, self.tzoom = 0, None, [], [0,50]
        self.CaImaging_bg_key = 'meanImg'
        self.CaImaging_key = 'Fluorescence'

        self.FILES_PER_DAY, self.FILES_PER_SUBJECT, self.SUBJECTS = {}, {}, {}
        
        self.minView = False
        self.showwindow()

    def open_file(self):

        # filename = '/home/yann/DATA/data.nwb'
        filename, _ = QtGui.QFileDialog.getOpenFileName(self,
                     "Open Multimodal Experimental Recording (NWB file) ",
                        (FOLDERS[self.fbox.currentText()] if self.fbox.currentText() in FOLDERS else os.path.join(os.path.expanduser('~'), 'DATA')),
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
        self.pPupilimg.clear()
        self.roiIndices = None

        
    def select_ROI(self):
        """ see select ROI above """
        self.roiIndices = self.select_ROI_from_pick()
        plots.raw_data_plot(self, self.tzoom, with_roi=True)

            
    def load_file(self, filename):
        """ should be a minimal processing so that the loading is fast"""
        read_NWB(self, filename,
                 verbose=True) # see ../analysis/read_NWB.py

        self.tzoom = self.tlim
        self.notes.setText(self.description)

        self.cal.setSelectedDate(self.nwbfile.session_start_time.date())
        if self.dbox.currentIndex()<1:
            self.dbox.clear()
            self.dbox.addItem(self.df_name)
            self.dbox.setCurrentIndex(0)
        self.sbox.clear()
        self.sbox.addItem(self.nwbfile.subject.description)
        self.sbox.setCurrentIndex(0)
        self.pbox.setCurrentIndex(1)
        self.visual_stim = None

        if 'ophys' in self.nwbfile.processing:
            self.roiPick.setText(' [select ROI] (%i-%i)' % (0, len(self.validROI_indices)-1))

    def load_VisualStim(self):

        # load visual stimulation
        if self.metadata['VisualStim']:
            self.metadata['load_from_protocol_data'] = True
            self.metadata['no-window'] = True
            self.visual_stim = build_stim(self.metadata, no_psychopy=True)
        else:
            self.visual_stim = None
            print(' /!\ No stimulation in this recording /!\  ')


    def scan_folder(self):

        print('inspecting the folder "%s" [...]' % FOLDERS[self.fbox.currentText()])

        FILES = get_files_with_extension(FOLDERS[self.fbox.currentText()],
                                         extension='.nwb', recursive=True)

        DATES = np.array([f.split(os.path.sep)[-1].split('-')[0] for f in FILES])

        self.FILES_PER_DAY = {}
        
        for d in np.unique(DATES):
            try:
                self.cal.setDateTextFormat(QtCore.QDate(datetime.date(*[int(dd) for dd in d.split('_')])),
                                           self.highlight_format)
                self.FILES_PER_DAY[d] = [os.path.join(FOLDERS[self.fbox.currentText()], f)\
                                         for f in np.array(FILES)[DATES==d]]
            except BaseException as be:
                print(be)
            # except ValueError:
            #     pass
            
        print(' -> found n=%i datafiles ' % len(FILES))
        

    def compute_subjects(self):

        FILES = get_files_with_extension(FOLDERS[self.fbox.currentText()],
                                         extension='.nwb', recursive=True)

        SUBJECTS, DISPLAY_NAMES = [], []
        for fn in FILES:
            infos = self.preload_datafolder(fn)
            SUBJECTS.append(infos['subject'])
            DISPLAY_NAMES.append(infos['display_name'])

        self.SUBJECTS = {}
        for s in np.unique(SUBJECTS):
            cond = (np.array(SUBJECTS)==s)
            self.SUBJECTS[s] = {'display_names':np.array(DISPLAY_NAMES)[cond],
                                'datafiles':np.array(FILES)[cond]}

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
            self.list_protocol = self.SUBJECTS[self.sbox.currentText()]['datafiles']
            self.update_df_names()
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
            self.metadata = None
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
        read_NWB(self, fn, metadata_only=True)
        infos = {'display_name' : self.df_name,
                 'subject': self.nwbfile.subject.description}
        self.io.close()
        return infos

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

        if self.pbox.currentText()=='-> Trial-average':
            self.windowTA = TrialAverageWindow(parent=self)
            self.windowTA.show()
        elif self.pbox.currentText()=='-> Behavioral-modulation' and (self.windowBM is None):
            self.window3 = BehavioralModWindow(parent=self)
            self.window3.show()
        elif self.pbox.currentText()=='-> Make-figures':
            self.windowFG = FiguresWindow(parent=self)
            self.windowFG.show()
        elif self.pbox.currentText()=='-> Open PDF summary':
            print('looking for pdf summary [...]')
            PDFS = []
            if os.path.isdir(os.path.join(self.datafile.replace('.pdf', ''), 'summary')):
                PDFS = os.listdir(os.path.join(self.datafile.replace('.pdf', ''), 'summary'))
                print('set of pdf-files found:')
                print(PDFS)
            else:
                print('no PDF summary files found !')
            pdf_filename = '~/Desktop/test.pdf'
            os.system('$(basename $(xdg-mime query default application/pdf) .desktop) %s ' % pdf_filename)
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

