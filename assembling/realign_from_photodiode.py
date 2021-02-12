import numpy as np
import os
from scipy.signal import argrelextrema
from scipy.ndimage.filters import gaussian_filter1d

def realign_from_photodiode(signal, metadata,
                            debug=False, verbose=True):

    if verbose:
        print('---> Realigning data with respect to photodiode signal [...] ')

    success = True

    
    # extract parameters
    dt = 1./metadata['NIdaq-acquisition-frequency']
    t = np.arange(len(signal))*dt
    
    tlim, tnew = [0, t[-1]], 0

    pre_window = np.min([metadata['presentation-interstim-period'], metadata['presentation-prestim-period']])
    t0 = metadata['time_start'][0]
    # length = metadata['presentation-duration']+metadata['presentation-interstim-period']
    # npulses = int(metadata['presentation-duration'])
    metadata['time_start_realigned'] = []
    Nepisodes = np.sum(metadata['time_start']<tlim[1])

    H, bins = np.histogram(signal, bins=200)
    baseline = .5*(bins[np.argmax(H)]+bins[np.argmax(H)+1])
    high_level = np.max(signal)
    print(metadata['time_stop'][-1])
    i=0
    while (i<Nepisodes) and (t0<t[-1]):
        cond = (t>=t0-pre_window) & (t<=t0+metadata['time_duration'][i]+metadata['presentation-interstim-period'])
        try:
            tshift, integral, threshold = find_onset_time(t[cond]-t0, signal[cond],
                                                          baseline=baseline, high_level=high_level)
            if debug and ((i<5) or (i>Nepisodes-30)):
                fig, ax = plt.subplots()
                ax.plot(t[cond], integral, label='integral')
                ax.plot(t[cond], integral*0+threshold, label='threshold')
                ax.plot((t0+tshift)*np.ones(2), ax.get_ylim(), 'k:', label='onset')
                ax.plot((t0+tshift+metadata['time_duration'][i])*np.ones(2), ax.get_ylim(), 'k:', label='offset')
                ax.plot(t[cond], normalize_signal(signal[cond])[0]*.8*np.diff(ax.get_ylim())[0],
                        label='photodiode-signal', lw=0.5, alpha=.3)
                plt.xlabel('time (s)')
                plt.ylabel('norm. signals')
                ax.legend(frameon=False)
                plt.show()
        except BaseException as be:
            print(be)
            print(i, Nepisodes, metadata['time_duration'][i])
            success = False # one exception is enough to make it fail
        metadata['time_start_realigned'].append(t0+tshift)
        t0=t0+tshift+metadata['time_duration'][i]+metadata['presentation-interstim-period']
        i+=1

    if verbose:
        if success:
            print('[ok]          --> succesfully realigned')
        else:
            print('[X]          --> realignement failed')
    if success:
        metadata['time_start_realigned'] = np.array(metadata['time_start_realigned'])
        metadata['time_stop_realigned'] = metadata['time_start_realigned']+\
            metadata['time_duration'][:len(metadata['time_start_realigned'])]
    else:
        metadata['time_start_realigned'] = np.array([])
        metadata['time_stop_realigned'] = np.array([])
    print(metadata['time_start'], metadata['time_start_realigned'])
    return success, metadata


def find_onset_time(t, photodiode_signal,
                    time_for_threshold=20e-3, baseline=0, high_level=1):
    """
    the threshold of integral increase corresponds to spending X-ms at half the maximum
    """
    # H, bins = np.histogram(photodiode_signal, bins=50)
    integral = np.cumsum(photodiode_signal-baseline)*(t[1]-t[0])

    # threshold = time_for_threshold*bins[hist_extrema[-1]]
    threshold = time_for_threshold*np.max(photodiode_signal)
    # threshold = time_for_threshold*high_level
    cond = np.argwhere(integral>threshold)
    t0 = t[cond[0][0]]

    return t0-time_for_threshold, integral, threshold

def normalize_signal(x):
    # just to plot above
    norm, xmin = 1./(np.max(x)-np.min(x)), np.min(x)
    return norm*(x-xmin), norm, xmin

if __name__=='__main__':

    import matplotlib.pylab as plt

    import argparse, os
    parser=argparse.ArgumentParser(description="""
    Realigning from Photodiod
    """,formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-df', "--datafolder", type=str, default='')
    args = parser.parse_args()

    data = np.load(os.path.join(args.datafolder, 'NIdaq.npy'), allow_pickle=True).item()['analog'][0]
    metadata = np.load(os.path.join(args.datafolder, 'metadata.npy'), allow_pickle=True).item()
    VisualStim = np.load(os.path.join(args.datafolder, 'visual-stim.npy'), allow_pickle=True).item()
    for key in ['time_start', 'time_stop', 'time_duration']:
        metadata[key] = VisualStim[key]


    plt.plot(data[::1000][:1000])
    plt.title('photodiode-signal (subsampled/100)')
    plt.show()
    
    realign_from_photodiode(data, metadata, debug=True)
    
