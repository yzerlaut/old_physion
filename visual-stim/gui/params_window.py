from PyQt5 import QtWidgets

import sys, os, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from default_params import STIMULI

LS = 25 # linespace


def draw_window(main_win):

    # Window size choosen appropriately
    window = QtWidgets.QDialog()
    window.setWindowTitle('%s -- %s' % (main_win.cbp.currentText(),
                                        main_win.cbs.currentText()))

    if main_win.cbp.currentText()=='':
        window.setGeometry(500, 100, 400, 40)
        QtWidgets.QLabel("\n"+10*' '+"Need to choose a presentation ! ", window)
    elif main_win.cbs.currentText()=='':
        window.setGeometry(500, 100, 400, 40)
        QtWidgets.QLabel("\n"+10*' '+"Need to choose a stimulus ! ", window)
    else:

        if main_win.protocol is None:
            main_win.protocol = STIMULI[main_win.cbs.currentText()]
            main_win.protocol['Presentation'] = main_win.cbp.currentText()
            main_win.protocol['Stimulus'] = main_win.cbs.currentText()

        if main_win.protocol['Presentation']=='Single-Stimulus':
            set_single_stim_params_window(main_win, window)
        elif main_win.protocol['Presentation']=='Stimuli-Sequence':
            set_stim_sequence_params_window(main_win, window)
        elif main_win.protocol['Presentation']=='Randomized-Sequence':
            set_random_sequence_params_window(main_win, window)
        else:
            QtWidgets.QLabel("Presentation type not recognized ", window)

    return window

def extract_params_from_window(main_win):

    
    if main_win.protocol['Presentation']=='Single-Stimulus':
        protocol = extract_params_from_single_stim_window(main_win)
    else:
        protocol = {}
    # elif main_win.protocol['Presentation']=='Stimuli-Sequence':
    #     protocol = extract_params_from_single_stim_window(main_win)
    # elif main_win.protocol['Presentation']=='Randomized-Sequence':
    #     set_random_sequence_params_window(main_win, window)
    protocol['Presentation'] = main_win.cbp.currentText()
    protocol['Stimulus'] = main_win.cbs.currentText()

    return protocol

################################################
############# Single-Stimulus ##################
################################################

def set_single_stim_params_window(main_win, window):
    """
    """
    params_keys = get_params_keys(main_win.protocol)
    window.setGeometry(500, 100, 320, LS*(7+len(params_keys)))
    QtWidgets.QLabel("  =======  Stimulus settings ======== ", window).move(5, LS)
    for i, key in enumerate(params_keys):
        QtWidgets.QLabel(key.split(' (')[0], window).move(20, LS*(2+i))
        setattr(window, key, QtWidgets.QDoubleSpinBox(window))
        getattr(window, key).move(150, LS*(2+i)-.15*LS)
        getattr(window, key).setValue(main_win.protocol[key])
        getattr(window, key).setSuffix(' ('+key.split(' (')[1])
    QtWidgets.QLabel(65*'-', window).move(0, LS*(2+len(params_keys)))
    QtWidgets.QLabel("  ======  Presentation settings ======= ", window).move(0, LS*(3+len(params_keys)))
    QtWidgets.QLabel(" Duration: ", window).move(0, LS*(4+len(params_keys)))
    window.durationBox = QtWidgets.QDoubleSpinBox(window)
    window.durationBox.move(100, LS*(4+len(params_keys))-.15*LS)
    window.durationBox.setSuffix(' s')
    QtWidgets.QLabel(" Pre-stim: ", window).move(0, LS*(5+len(params_keys)))
    window.prestimBox = QtWidgets.QDoubleSpinBox(window)
    window.prestimBox.move(100, LS*(5+len(params_keys))-.15*LS)
    window.prestimBox.setSuffix(' s')
    window.prestimType = QtWidgets.QComboBox(window)
    window.prestimType.addItems(['Black Screen' ,'Grey Screen', 'White Screen'])
    window.prestimType.move(200, LS*(5+len(params_keys))-.15*LS)
    
    QtWidgets.QLabel(" Post-stim: ", window).move(0, LS*(6+len(params_keys)))
    window.poststimBox = QtWidgets.QDoubleSpinBox(window)
    window.poststimBox.move(100, LS*(6+len(params_keys))-.15*LS)
    window.poststimBox.setSuffix(' s')
    window.poststimType = QtWidgets.QComboBox(window)
    window.poststimType.addItems(['Black Screen' ,'Grey Screen', 'White Screen'])
    window.poststimType.move(200, LS*(6+len(params_keys))-.15*LS)


def extract_params_from_single_stim_window(main_win):

    protocol = {}
    params_keys = get_params_keys(STIMULI[main_win.cbs.currentText()])
    for i, key in enumerate(params_keys):
        protocol[key] = getattr(main_win.params_window, key).value()
    protocol['presentation-duration'] = main_win.params_window.durationBox.value()
    protocol['presentation-prestim-delay'] = main_win.params_window.prestimBox.value()
    protocol['presentation-poststim-period'] = main_win.params_window.poststimBox.value()
    protocol['presentation-prestim-screen'] = main_win.params_window.prestimType.currentText()
    protocol['presentation-poststim-screen'] = main_win.params_window.poststimType.currentText()
    return protocol
    

def set_stim_sequence_params_window(main_win, window):
    pass
    
def set_random_sequence_params_window(main_win, window):
    pass

def get_params_keys(stim_dict):
    keys = []
    for key in stim_dict:
        if not ((key[-2:]=='-1') or (key[-2:]=='-2') or (key[:2]=='N-')\
                or (key=='Presentation') or (key=='Stimulus') or key[:13]=='presentation-'):
            keys.append(key)
    return keys

    
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
    window.set_Tsmooth_text.move(x0+LS0, y0+30)
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
    window.set_bandfactor.move(x0+5, y0+30)
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


