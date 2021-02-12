import numpy as np
import os
from scipy.signal import argrelextrema
from scipy.ndimage.filters import gaussian_filter1d

def realign_from_photodiode(signal, metadata,
                            debug=False, verbose=True, n_vis=5):

    if verbose:
        print('---> Realigning data with respect to photodiode signal [...] ')

    success = True

    
    # extract parameters
    dt = 1./metadata['NIdaq-acquisition-frequency']
    t = np.arange(len(signal))*dt
    
    tlim, tnew = [0, t[-1]], 0

    pre_window = np.min([metadata['presentation-interstim-period'], metadata['presentation-prestim-period']])
    t0 = metadata['time_start'][0]
    metadata['time_start_realigned'] = []
    Nepisodes = np.sum(metadata['time_start']<tlim[1])

    H, bins = np.histogram(signal, bins=50)
    baseline = bins[np.argmax(H)+1]
    high_level = np.max(signal)

    i=0
    while (i<Nepisodes) and (t0<(t[-1]-metadata['time_duration'][i])):
        cond = (t>=t0-pre_window) & (t<=t0+metadata['time_duration'][i]+metadata['presentation-interstim-period'])
        try:
            tshift, integral, threshold = find_onset_time(t[cond]-t0, signal[cond],
                                                          baseline=baseline, high_level=high_level)
            if debug and ((i<n_vis) or (i>Nepisodes-n_vis)):
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
    return success, metadata


def find_onset_time(t, photodiode_signal,
                    time_for_threshold=50e-3,
                    smoothing_time = 20e-3,
                    baseline=0, high_level=1):
    """
    the threshold of integral increase corresponds to spending X-ms at half the maximum
    """
    smoothed = gaussian_filter1d(photodiode_signal, int(smoothing_time/(t[1]-t[0])))
    threshold = np.max(smoothed)/2.
    cond = (smoothed[1:]>=threshold) & (smoothed[:-1]<=threshold)
    t0 = t[:-1][cond][0]
    return t0-smoothing_time, smoothed, threshold

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
    parser.add_argument('-n', "--n_vis", type=int, default=5)
    args = parser.parse_args()

    data = np.load(os.path.join(args.datafolder, 'NIdaq.npy'), allow_pickle=True).item()['analog'][0]
    metadata = np.load(os.path.join(args.datafolder, 'metadata.npy'), allow_pickle=True).item()
    VisualStim = np.load(os.path.join(args.datafolder, 'visual-stim.npy'), allow_pickle=True).item()
    if 'time_duration' not in VisualStim:
        VisualStim['time_duration'] = np.array(VisualStim['time_stop'])-np.array(VisualStim['time_start'])
    for key in ['time_start', 'time_stop', 'time_duration']:
        metadata[key] = VisualStim[key]


    plt.plot(data[::1000][:1000])
    plt.title('photodiode-signal (subsampled/100)')
    plt.show()
    
    realign_from_photodiode(data, metadata, debug=True, n_vis=args.n_vis)
    
