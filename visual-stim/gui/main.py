import sys
from PyQt5.QtWidgets import QApplication, QWidget

import sys, os, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from psychopy_code.stimuli import visual_stim

import numpy as np
from PyQt5 import QtGui, QtWidgets, QtCore

PROTOCOLS = ['Single-Stimulus', 'Stimuli-Sequence', 'Randomized-Sequence']

STIMULI = {
    'light-level':{},
    'full-field-grating':{},
    'drifting-FF-grating':{},
    'center-grating':{},
    'surround-grating':{},
    'center-surround-grating':{},
    'Natural-Image':{},
    'full-field-grating+VEM':{},
    'natural-image+VEM':{},
}

class Window(QtWidgets.QMainWindow):
    
    def __init__(self, app, parent=None):
        
        super(Window, self).__init__(parent)
        
        # buttons and functions
        LABELS = ["i) Initialize", "r) Run", "s) Stop", "q) Quit"]
        FUNCTIONS = [self.initialize, self.run, self.stop, self.quit]
        button_length = 100
        
        self.setWindowTitle('Visual Stimulation Program')
        self.setGeometry(200, 200, button_length*len(LABELS), 200)

        # protocol change
        label1 = QtWidgets.QLabel("/|===> Presentation <===|\\", self)
        label1.setMinimumWidth(320)
        label1.move(100, 50)
        self.cbp = QtWidgets.QComboBox(self)
        self.cbp.addItems(['']+PROTOCOLS)
        self.cbp.currentIndexChanged.connect(self.change_protocol)
        self.cbp.setMinimumWidth(250)
        self.cbp.move(70, 80)

        # stimulus pick
        label2 = QtWidgets.QLabel("  /|===> Stimulus <===|\\", self)
        label2.setMinimumWidth(330)
        label2.move(100, 110)
        self.cbs = QtWidgets.QComboBox(self)
        self.cbs.addItems(['']+list(STIMULI.keys()))
        self.cbs.currentIndexChanged.connect(self.change_stimulus)
        self.cbs.setMinimumWidth(250)
        self.cbs.move(70, 140)


        mainMenu = self.menuBar()
        self.fileMenu = mainMenu.addMenu('&File')
        self.fileMenu = mainMenu.addMenu('&Configuration')

        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage('...')
        
        screen = app.primaryScreen()
        rect = screen.availableGeometry()
        self.screensize = rect.width(), rect.height()

        for func, label, shift in zip(FUNCTIONS, LABELS,\
                                      button_length*np.arange(len(LABELS))):
            btn = QtWidgets.QPushButton(label, self)
            btn.clicked.connect(func)
            btn.setMinimumWidth(button_length)
            btn.move(shift, 20)
            action = QtWidgets.QAction(label, self)
            action.setShortcut(label.split(')')[0])
            action.triggered.connect(func)
            self.fileMenu.addAction(action)

        self.show()

        
    def initialize(self):
        self.stim = visual_stim(protocol=self.cbp.currentText(),
                                stimulus=self.cbs.currentText())
        self.stim.build_protocol(a=0)
    
    def run(self):
        self.initialize()
        self.stim.show()
    
    def stop(self):
        pass
    
    def quit(self):
        sys.exit()

    def change_protocol(self):
        print(self.cbp.currentText())
        
    def change_stimulus(self):
        print(self.cbs.currentText())

    def create_params_window(self):
        window = QtWidgets.QDialog()

        
    def save_results(self):
        if 'NSI' in self.data:
            results_filename = '.'.join(self.filename.split('.')[:-1]) if '.' in self.filename else self.filename
            results_filename += '_NSI_'+datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')+'.h5'
            print(self.data.keys())
            to_save = {'validated_times': self.data['new_t'][self.data['NSI_validated']],
                       'validated_NSI':self.data['NSI'][self.data['NSI_validated']]}
            save_dict_to_hdf5(to_save, results_filename)
            self.statusBar.showMessage('Results of analysis saved as : '+results_filename)
        else:
            self.statusBar.showMessage('Need to perform analysis first...')

    
# def create_stim_window(parent, x0=10, y0=250, hspace=0.1, left=0.08, right=0.99, bottom=0.2, figsize=(8,3)):

#     width, height = parent.screensize[0]/1.1, parent.screensize[1]/1.5
    
#     fig_large_view, AX_large_view = plt.subplots(1, figsize=(figsize[0], figsize[1]/1.5))
#     plt.subplots_adjust(hspace=hspace, left=left, right=right, bottom=1.5*bottom)
#     AX_large_view.set_xlabel('time (s)')
#     AX_large_view.set_yticks([])
#     AX_large_view.set_ylim([-10, 20])
        
#     fig_zoom, AX_zoom  = plt.subplots(3, 1, figsize=(figsize[0], figsize[1]))
#     plt.subplots_adjust(hspace=hspace, left=left, right=right, bottom=bottom)
#     AX_zoom[0].set_xticklabels([])
#     AX_zoom[1].set_xticklabels([])
#     AX_zoom[2].set_xlabel('time (s)')
            
#     for ax2, label in zip(AX_zoom, ['$V_{ext}$ ($\mu$V)', 'pLFP ($\mu$V)', 'NSI ($\mu$V)']):
#         ax2.set_ylabel(label)
#     for x, col, label in zip([0.75, 0.55, 0.35], ['k', 'k', 'k'],
#                              ['$V_{ext}$', 'pLFP', 'NSI']):
#         AX_large_view.annotate(label, (0.03, x), color=col, xycoords='figure fraction')
        
#     # Window size choosen appropriately
#     window = QtWidgets.QDialog()
#     window.setGeometry(x0, y0, width, height)

#     # this is the Canvas Widget that displays the `figure`
#     # it takes the `figure` instance as a parameter to __init__
    
#     canvas_large_view = FigureCanvas(fig_large_view)
#     canvas_zoom = FigureCanvas(fig_zoom)

#     layout = QtWidgets.QGridLayout(window)
#     layout.addWidget(canvas_large_view)
#     layout.addWidget(canvas_zoom)
        
#     window.setLayout(layout)
            
#     return window, AX_large_view, AX_zoom, canvas_large_view, canvas_zoom


def set_recording_params(window, x0=10, y0=30):
    # front text
    Data_label = QtWidgets.QLabel("===> Acquisition parameters:", window)
    Data_label.setMinimumWidth(200)
    Data_label.move(x0, y0)
    # filename text ---> change with open file
    window.filename_textbox = QtWidgets.QLabel('Filename: [...]', window)
    window.filename_textbox.setMinimumWidth(500)
    window.filename_textbox.move(x0+300, y0)
    myFont=QtGui.QFont() # putting a bold font
    myFont.setBold(True)
    window.filename_textbox.setFont(myFont)
    # acquisision time step ---> changed here !
    window.set_acq_freq_text = QtWidgets.QLabel('Acq. Freq.:', window)
    window.set_acq_freq_text.setMinimumWidth(300)
    window.set_acq_freq_text.move(x0, y0+30)
    window.set_acq_freq = QtWidgets.QDoubleSpinBox(window)
    window.set_acq_freq.setMaximumWidth(100)
    window.set_acq_freq.move(x0+100, y0+30)
    window.set_acq_freq.setRange(0.1, 100)
    window.set_acq_freq.setDecimals(1)
    window.set_acq_freq.setSuffix(" kHz")
    window.set_acq_freq.setSingleStep(10)
    window.set_acq_freq.setValue(DEFAULT_VALUES['dt'])
    window.set_acq_freq.valueChanged.connect(window.acq_freq_change)
    # acquisision time step ---> changed here !
    window.set_acq_gain_text = QtWidgets.QLabel('Channel Gain:', window)
    window.set_acq_gain_text.setMinimumWidth(300)
    window.set_acq_gain_text.move(x0+400, y0+30)
    window.set_acq_gain = QtWidgets.QDoubleSpinBox(window)
    window.set_acq_gain.setMaximumWidth(200)
    window.set_acq_gain.setSuffix(" mV/V")
    window.set_acq_gain.setDecimals(3)
    window.set_acq_gain.move(x0+490, y0+30)
    window.set_acq_gain.setRange(0.001, 1000.0)
    window.set_acq_gain.setSingleStep(10)
    window.set_acq_gain.setValue(DEFAULT_VALUES['gain_mVpV'])
    window.set_acq_gain.valueChanged.connect(window.gain_change)
    # acquisision channel ---> changed here !
    window.set_acq_channel_text = QtWidgets.QLabel('Channel:', window)
    window.set_acq_channel_text.setMinimumWidth(300)
    window.set_acq_channel_text.move(x0+230, y0+30)
    window.set_acq_channel = QtWidgets.QComboBox(window)
    window.set_acq_channel.currentIndexChanged.connect(window.channel_change)
    window.set_acq_channel.setMaximumWidth(100)
    window.set_acq_channel.addItem("1")
    window.set_acq_channel.move(x0+290, y0+30)

    
def set_analysis_params(window, x0=10, y0=60):
    # front text
    Data_label = QtWidgets.QLabel("===> Analysis parameters:", window)
    Data_label.setMinimumWidth(200)
    Data_label.move(x0, y0)
    # acquisision time step ---> changed here !
    window.set_alpha_text = QtWidgets.QLabel('Alpha:', window)
    window.set_alpha_text.move(x0+0, y0+30)
    window.set_alpha = QtWidgets.QDoubleSpinBox(window)
    window.set_alpha.setMaximumWidth(60)
    window.set_alpha.move(x0+45, y0+30)
    window.set_alpha.setRange(0.01, 10.0)
    window.set_alpha.setDecimals(2)
    window.set_alpha.setSingleStep(0.01)
    window.set_alpha.setValue(DEFAULT_VALUES['alpha'])
    # Tstate window step ---> changed here !
    window.set_Tstate_text = QtWidgets.QLabel('Tstate:', window)
    window.set_Tstate_text.move(x0+115, y0+30)
    window.set_Tstate = QtWidgets.QDoubleSpinBox(window)
    window.set_Tstate.setMaximumWidth(80)
    window.set_Tstate.setSuffix("ms")
    window.set_Tstate.setDecimals(1)
    window.set_Tstate.move(x0+160, y0+30)
    window.set_Tstate.setRange(1, 2000.0)
    window.set_Tstate.setSingleStep(0.1)
    window.set_Tstate.setValue(DEFAULT_VALUES['Tstate'])
    # Tsmoothing ---> changed here !
    window.set_Tsmooth_text = QtWidgets.QLabel('Tsmooth:', window)
    window.set_Tsmooth_text.move(x0+250, y0+30)
    window.set_Tsmooth = QtWidgets.QDoubleSpinBox(window)
    window.set_Tsmooth.setMaximumWidth(70)
    window.set_Tsmooth.setSuffix("ms")
    window.set_Tsmooth.setDecimals(1)
    window.set_Tsmooth.move(x0+310, y0+30)
    window.set_Tsmooth.setRange(1., 200.0)
    window.set_Tsmooth.setSingleStep(0.1)
    window.set_Tsmooth.setValue(DEFAULT_VALUES['Tsmooth'])
    # Root-Freq ---> changed here !
    window.set_rootfreq_text = QtWidgets.QLabel('f0:', window)
    # window.set_rootfreq_text.setMinimumWidth(200)
    window.set_rootfreq_text.move(x0+395, y0+30)
    window.set_rootfreq = QtWidgets.QDoubleSpinBox(window)
    window.set_rootfreq.setMaximumWidth(70)
    window.set_rootfreq.setSuffix("Hz")
    window.set_rootfreq.setDecimals(1)
    window.set_rootfreq.move(x0+415, y0+30)
    window.set_rootfreq.setRange(0.1, 100.0)
    window.set_rootfreq.setSingleStep(0.1)
    window.set_rootfreq.setValue(DEFAULT_VALUES['Root_freq'])
    # Band-Factor ---> changed here !
    window.set_bandfactor_text = QtWidgets.QLabel('w0:', window)
    # window.set_bandfactor_text.setMinimumWidth(200)
    window.set_bandfactor_text.move(x0+495, y0+30)
    window.set_bandfactor = QtWidgets.QDoubleSpinBox(window)
    window.set_bandfactor.setMaximumWidth(70)
    window.set_bandfactor.setDecimals(1)
    window.set_bandfactor.move(x0+525, y0+30)
    window.set_bandfactor.setRange(0.1, 100.0)
    window.set_bandfactor.setSingleStep(0.1)
    window.set_bandfactor.setValue(DEFAULT_VALUES['Band_Factor'])
    # N wavelets ---> changed here !
    window.set_N_wvlts_text = QtWidgets.QLabel('N wavelets:', window)
    # window.set_N_wvlts_text.setMinimumWidth(200)
    window.set_N_wvlts_text.move(x0+600, y0+30) # 
    window.set_N_wvlts = QtWidgets.QDoubleSpinBox(window)
    window.set_N_wvlts.setMaximumWidth(50)
    # window.set_N_wvlts.setSuffix("Hz")
    window.set_N_wvlts.setDecimals(0)
    window.set_N_wvlts.move(x0+670, y0+30)
    window.set_N_wvlts.setRange(1, 100)
    window.set_N_wvlts.setSingleStep(1)
    window.set_N_wvlts.setValue(DEFAULT_VALUES['N_wavelets'])
    # Subsampling ---> changed here !
    window.set_subsampling_text = QtWidgets.QLabel('pLFP-Subsampling:', window)
    window.set_subsampling_text.setMinimumWidth(200)
    window.set_subsampling_text.move(x0+730, y0+30)
    window.set_subsampling = QtWidgets.QDoubleSpinBox(window)
    window.set_subsampling.setMaximumWidth(70)
    window.set_subsampling.setSuffix("ms")
    window.set_subsampling.setDecimals(1)
    window.set_subsampling.move(x0+850, y0+30)
    window.set_subsampling.setRange(0.1, 100.0)
    window.set_subsampling.setSingleStep(0.1)
    window.set_subsampling.setValue(DEFAULT_VALUES['Tsubsampling'])
    # # p0 percentile ---> changed here !
    # window.set_p0_percentile_text = QtWidgets.QLabel('p0 percentile:', window)
    # window.set_p0_percentile_text.setMinimumWidth(300)
    # window.set_p0_percentile_text.move(x0+565, y0+30)
    # window.set_p0_percentile = QtWidgets.QDoubleSpinBox(window)
    # window.set_p0_percentile.setMaximumWidth(100)
    # window.set_p0_percentile.setSuffix("%")
    # window.set_p0_percentile.setDecimals(1)
    # window.set_p0_percentile.move(x0+650, y0+30)
    # window.set_p0_percentile.setRange(0.1, 100.0)
    # window.set_p0_percentile.setSingleStep(0.1)
    # window.set_p0_percentile.setValue(DEFAULT_VALUES['p0_percentile'])

        
if __name__ == '__main__':
    import time
    app = QtWidgets.QApplication(sys.argv)
    main = Window(app)
    # main.show()
    sys.exit(app.exec_())
    

# def main():

#     app = QApplication(sys.argv)

#     w = QWidget()
#     w.resize(600, 300)
#     w.move(300, 300)
#     w.setWindowTitle('Visual Stimulation Program')
#     w.show()

#     sys.exit(app.exec_())


# if __name__ == '__main__':
#     main()
