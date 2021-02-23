from PyQt5 import QtWidgets

import sys, os, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[0]))
from default_params import STIMULI, PRESENTATION, NAME_TO_COLOR, BLANK_SCREENS

LS = 25 # linespace

def draw_window(parent, protocol):

    # Window size choosen appropriately
    window = QtWidgets.QDialog()
    window.setWindowTitle('%s -- %s' % (parent.cbp.currentText(),
                                        parent.cbs.currentText()))

    if parent.cbp.currentText()=='':
        window.setGeometry(500, 100, 400, 40)
        QtWidgets.QLabel("\n"+10*' '+"Need to choose a presentation ! ", window)
    elif parent.cbs.currentText()=='':
        window.setGeometry(500, 100, 400, 40)
        QtWidgets.QLabel("\n"+10*' '+"Need to choose a stimulus ! ", window)
        
    else:

        if protocol is None:
            protocol = {**STIMULI[parent.cbs.currentText()], **PRESENTATION}
            protocol['Presentation'] = parent.cbp.currentText()
            protocol['Stimulus'] = parent.cbs.currentText()
            protocol['Setup'] = parent.cbst.currentText()
            
        if protocol['Presentation']=='Single-Stimulus':
            set_single_stim_params_window(parent, window, protocol)
        elif protocol['Presentation']=='Stimuli-Sequence':
            set_multiple_stim_params_window(parent, window, protocol)
        elif protocol['Presentation']=='Randomized-Sequence':
            set_multiple_stim_params_window(parent, window, protocol)
        else:
            QtWidgets.QLabel("Presentation type not recognized ", window)

    return window


################################################
############# Single-Stimulus ##################
################################################

def set_single_stim_params_window(parent, window, protocol):
    """
    window.set_acq_gain.setMaximumWidth(200)
    window.set_acq_gain.setDecimals(3)

    """
    params_keys = get_params_keys(protocol, stimulus=protocol['Stimulus'])
    window.setGeometry(500, 100, 320, LS*(7+len(params_keys)))
    QtWidgets.QLabel("  =======  Stimulus settings ======== ", window).move(5, LS)
    for i, key in enumerate(params_keys):
        new_key = key.split(' (')[0]
        QtWidgets.QLabel(new_key, window).move(20, LS*(2+i))
        setattr(window, key, QtWidgets.QDoubleSpinBox(window))
        getattr(window, key).move(150, LS*(2+i)-.15*LS)
        getattr(window, key).setValue(protocol[key])
        getattr(window, key).setSuffix(' ('+key.split(' (')[1])
        getattr(window, key).setRange(STIMULI[parent.cbs.currentText()][new_key+'-1'],
                                      STIMULI[parent.cbs.currentText()][new_key+'-2'])
        
    QtWidgets.QLabel(65*'-', window).move(0, LS*(2+len(params_keys)))
    QtWidgets.QLabel("  ======  Presentation settings ======= ", window).move(0, LS*(3+len(params_keys)))
    QtWidgets.QLabel(" Duration: ", window).move(0, LS*(4+len(params_keys)))
    window.durationBox = QtWidgets.QDoubleSpinBox(window)
    window.durationBox.move(100, LS*(4+len(params_keys))-.15*LS)
    window.durationBox.setValue(protocol['presentation-duration'])
    window.durationBox.setRange(0, 10000)
    window.durationBox.setSuffix(' s')
    QtWidgets.QLabel(" Pre-stim: ", window).move(0, LS*(5+len(params_keys)))
    window.prestimBox = QtWidgets.QDoubleSpinBox(window)
    window.prestimBox.move(100, LS*(5+len(params_keys))-.15*LS)
    window.prestimBox.setValue(protocol['presentation-prestim-period'])
    window.prestimBox.setSuffix(' s')
    window.prestimType = QtWidgets.QComboBox(window)
    window.prestimType.addItems(BLANK_SCREENS)
    index = [i for i, b in enumerate(BLANK_SCREENS) if NAME_TO_COLOR[b]==protocol['presentation-prestim-screen']][0]
    window.prestimType.setCurrentIndex(index)
    window.prestimType.move(200, LS*(5+len(params_keys))-.15*LS)
    
    QtWidgets.QLabel(" Post-stim: ", window).move(0, LS*(6+len(params_keys)))
    window.poststimBox = QtWidgets.QDoubleSpinBox(window)
    window.poststimBox.move(100, LS*(6+len(params_keys))-.15*LS)
    window.poststimBox.setValue(protocol['presentation-poststim-period'])
    window.poststimBox.setSuffix(' s')
    window.poststimType = QtWidgets.QComboBox(window)
    window.poststimType.addItems(BLANK_SCREENS)
    index = [i for i, b in enumerate(BLANK_SCREENS) if NAME_TO_COLOR[b]==protocol['presentation-poststim-screen']][0]
    window.poststimType.setCurrentIndex(index)
    window.poststimType.move(200, LS*(6+len(params_keys))-.15*LS)


################################################
############# Multiple-Stimuli ##################
################################################

def set_multiple_stim_params_window(parent, window, protocol):
    """
    """
    params_keys = get_params_keys(protocol, stimulus=protocol['Stimulus'])
    window.setGeometry(500, 100, 600, LS*(13+len(params_keys)))
    QtWidgets.QLabel("  "+20*"="+" Set-of-Stimuli settings "+20*"=", window).move(5, LS)
    QtWidgets.QLabel("|| Params ", window).move(5, 2*LS)
    QtWidgets.QLabel("|| Low-value ", window).move(150, 2*LS)
    QtWidgets.QLabel("|| High-value ", window).move(300, 2*LS)
    QtWidgets.QLabel("|| Discretization ", window).move(450, 2*LS)
    for i, key in enumerate(params_keys):
        new_key = key.split(' (')[0]
        QtWidgets.QLabel(' - '+key, window).move(0, LS*(3+i)) # params
        # low-value
        setattr(window, new_key+'-1', QtWidgets.QDoubleSpinBox(window))
        getattr(window, new_key+'-1').move(150, LS*(3+i)-.15*LS)
        getattr(window, new_key+'-1').setSuffix(' ('+key.split(' (')[1])
        getattr(window, new_key+'-1').setRange(STIMULI[parent.cbs.currentText()][new_key+'-1'],
                                               STIMULI[parent.cbs.currentText()][new_key+'-2'])
        getattr(window, new_key+'-1').setValue(protocol[new_key+'-1'])
        # high-value
        setattr(window, new_key+'-2', QtWidgets.QDoubleSpinBox(window))
        getattr(window, new_key+'-2').move(300, LS*(3+i)-.15*LS)
        getattr(window, new_key+'-2').setSuffix(' ('+key.split(' (')[1])
        getattr(window, new_key+'-2').setRange(STIMULI[parent.cbs.currentText()][new_key+'-1'],
                                               STIMULI[parent.cbs.currentText()][new_key+'-2'])
        getattr(window, new_key+'-2').setValue(protocol[new_key+'-2'])
        # high-value
        setattr(window, 'N-'+new_key, QtWidgets.QSpinBox(window))
        getattr(window, 'N-'+new_key).move(450, LS*(3+i)-.15*LS)
        getattr(window, 'N-'+new_key).setValue(protocol['N-'+new_key])
        
    QtWidgets.QLabel('   '+125*'-', window).move(0, LS*(3+len(params_keys)))
    QtWidgets.QLabel("  "+20*"="+" Presentation settings "+20*"=", window).move(5, LS*(4+len(params_keys)))
    # stim duration
    QtWidgets.QLabel(" Duration: ", window).move(0, LS*(5+len(params_keys)))
    window.durationBox = QtWidgets.QDoubleSpinBox(window)
    window.durationBox.move(100, LS*(5+len(params_keys))-.15*LS)
    window.durationBox.setValue(protocol['presentation-duration'])
    window.durationBox.setSuffix(' s')
    
    # inter-stimulation props
    QtWidgets.QLabel(" Inter-stim: ", window).move(0, LS*(6+len(params_keys)))
    window.interstimBox = QtWidgets.QDoubleSpinBox(window)
    window.interstimBox.move(100, LS*(6+len(params_keys))-.15*LS)
    window.interstimBox.setValue(protocol['presentation-interstim-period'])
    window.interstimBox.setSuffix(' s')
    window.interstimType = QtWidgets.QComboBox(window)
    window.interstimType.addItems(BLANK_SCREENS)
    index = [i for i, b in enumerate(BLANK_SCREENS) if NAME_TO_COLOR[b]==protocol['presentation-interstim-screen']][0]
    window.interstimType.setCurrentIndex(index)
    window.interstimType.move(200, LS*(6+len(params_keys))-.15*LS)
    
    # pre-stimulation props
    QtWidgets.QLabel(" Pre-stim: ", window).move(0, LS*(7+len(params_keys)))
    window.prestimBox = QtWidgets.QDoubleSpinBox(window)
    window.prestimBox.move(100, LS*(7+len(params_keys))-.15*LS)
    window.prestimBox.setValue(protocol['presentation-prestim-period'])
    window.prestimBox.setSuffix(' s')
    window.prestimType = QtWidgets.QComboBox(window)
    window.prestimType.addItems(BLANK_SCREENS)
    index = [i for i, b in enumerate(BLANK_SCREENS) if NAME_TO_COLOR[b]==protocol['presentation-prestim-screen']][0]
    window.prestimType.setCurrentIndex(index)
    window.prestimType.move(200, LS*(7+len(params_keys))-.15*LS)
    
    # post-stimulation props
    QtWidgets.QLabel(" Post-stim: ", window).move(0, LS*(8+len(params_keys)))
    window.poststimBox = QtWidgets.QDoubleSpinBox(window)
    window.poststimBox.move(100, LS*(8+len(params_keys))-.15*LS)
    window.poststimBox.setValue(protocol['presentation-poststim-period'])
    window.poststimBox.setSuffix(' s')
    window.poststimType = QtWidgets.QComboBox(window)
    window.poststimType.addItems(BLANK_SCREENS)
    index = [i for i, b in enumerate(BLANK_SCREENS) if NAME_TO_COLOR[b]==protocol['presentation-poststim-screen']][0]
    window.poststimType.setCurrentIndex(index)
    window.poststimType.move(200, LS*(8+len(params_keys))-.15*LS)


    # Starting index props
    QtWidgets.QLabel("    Starting index for Set-of-Stimuli: ", window).move(0, LS*(9+len(params_keys)))
    window.i0Box = QtWidgets.QSpinBox(window)
    window.i0Box.setValue(protocol['starting-index'])
    window.i0Box.move(240, LS*(9+len(params_keys))-.15*LS)

    # N-repeat props
    QtWidgets.QLabel("    N-repeat of full Set-of-Stimuli: ", window).move(0, LS*(10+len(params_keys)))
    window.NrepeatBox = QtWidgets.QSpinBox(window)
    window.NrepeatBox.setValue(protocol['N-repeat'])
    window.NrepeatBox.move(240, LS*(10+len(params_keys))-.15*LS)

    if protocol['Presentation']=='Randomized-Sequence':
        # shuffling-index props
        QtWidgets.QLabel("    Seed for shuffling: ", window).move(0, LS*(11+len(params_keys)))
        window.SeedBox = QtWidgets.QSpinBox(window)
        window.SeedBox.setValue(protocol['shuffling-seed'])
        window.SeedBox.move(240, LS*(11+len(params_keys))-.15*LS)
    
    
def extract_params_from_window(parent):

    protocol = {'Presentation': parent.cbp.currentText(),
                'Stimulus': parent.cbs.currentText(),
                'Setup': parent.cbst.currentText()}
    
    params_keys = get_params_keys(STIMULI[parent.cbs.currentText()])
    
    # temporal values
    protocol['presentation-duration'] = parent.params_window.durationBox.value()
    protocol['presentation-prestim-period'] = parent.params_window.prestimBox.value()
    protocol['presentation-poststim-period'] = parent.params_window.poststimBox.value()
    # screen during periods
    protocol['presentation-prestim-screen'] = NAME_TO_COLOR[parent.params_window.prestimType.currentText()]
    protocol['presentation-poststim-screen'] = NAME_TO_COLOR[parent.params_window.poststimType.currentText()]
    
    if parent.cbp.currentText()=='Single-Stimulus':
        for i, key in enumerate(params_keys):
            protocol[key] = getattr(parent.params_window, key).value()
            
    else:
        protocol['presentation-interstim-period'] = parent.params_window.interstimBox.value()
        protocol['presentation-interstim-screen'] = NAME_TO_COLOR[parent.params_window.interstimType.currentText()]
        protocol['N-repeat'] = parent.params_window.NrepeatBox.value()
        protocol['starting-index'] = parent.params_window.i0Box.value()
        if protocol['Presentation']=='Randomized-Sequence':
            protocol['shuffling-seed'] = parent.params_window.SeedBox.value()

        for i, key in enumerate(params_keys):
            new_key = key.split(' (')[0]
            protocol[new_key+'-1'] = getattr(parent.params_window, new_key+'-1').value()
            protocol[new_key+'-2'] = getattr(parent.params_window, new_key+'-2').value()
            protocol['N-'+new_key] = getattr(parent.params_window, 'N-'+new_key).value()
            
    return protocol
    

def get_params_keys(protocol, stimulus=None):
    keys = []
    if stimulus is None:
        stim_dict = protocol
    else:
        stim_dict = STIMULI[stimulus]
    for key in stim_dict:
        if not ((key[-2:]=='-1') or (key[-2:]=='-2') or (key[:2]=='N-')):
            keys.append(key)
    return keys

def get_presentation_keys(protocol):
    keys = []
    for key in protocol:
        if key[:13]=='presentation-':
            keys.append(key)
    return keys



