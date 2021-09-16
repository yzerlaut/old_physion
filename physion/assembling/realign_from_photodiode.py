import numpy as np
import os
from scipy.signal import argrelextrema
from scipy.ndimage.filters import gaussian_filter1d

def realign_from_photodiode(signal,
                            metadata,
                            sampling_rate=None,
                            smoothing_time=20e-3,
                            debug=False, istart_debug=0,
                            verbose=True, n_vis=5):

    if verbose:
        print('---> Realigning data with respect to photodiode signal [...] ')

    success = True

    # extract parameters
    if sampling_rate is None:
        dt = 1./metadata['NIdaq-acquisition-frequency']
    else:
        dt = 1/sampling_rate
    t = np.arange(len(signal))*dt
    
    tlim, tnew = [0, t[-1]], 0

    tstart, tshift = metadata['time_start'][0]-.5, 0
    metadata['time_start_realigned'] = []
    
    # compute signal boundaries to evaluate threshold crossing of photodiode signal
    H, bins = np.histogram(signal, bins=50)
    baseline = bins[np.argmax(H)+1]
    high_level = np.max(signal)

    # looping over episodes
    i=0
    while (tstart<(t[-1]-metadata['time_duration'][i])):
        cond = (t>=tstart) & (t<=tstart+metadata['time_duration'][i]+15) # 15s max time delay (the time to build up the next stim can be quite large)
        try:
            tshift, integral, threshold = find_onset_time(t[cond]-tstart, signal[cond],
                                                          smoothing_time=smoothing_time,
                                                          baseline=baseline, high_level=high_level)
            # print('ep.#%i, tshift=%.1f, prtcl_id=%i' % (i, tshift, metadata['protocol_id'][i]))
            if debug and ((i>=istart_debug) and (i<istart_debug+n_vis)):
                fig, ax = plt.subplots()
                ax.plot(t[cond], integral, label='smoothed')
                ax.plot(t[cond], integral*0+threshold, label='threshold')
                ax.plot((tstart+tshift)*np.ones(2), ax.get_ylim(), 'k:', label='onset')
                ax.plot((tstart+tshift+metadata['time_duration'][i])*np.ones(2), ax.get_ylim(), 'k:', label='offset')
                ax.plot(t[cond], normalize_signal(signal[cond])[0]*.8*np.diff(ax.get_ylim())[0],
                        label='photodiode-signal', lw=0.5, alpha=.3)
                plt.xlabel('time (s)')
                plt.ylabel('norm. signals')
                ax.set_title('ep. #%i' % i)
                ax.legend(frameon=False)
                plt.show()
        except BaseException as be:
            print('\n', be)
            print('\n'+' /!\ REALIGNEMENT FAILED (@ i=%i ) /!\ \n' % i)
            # print(i, Nepisodes, metadata['time_duration'][i])
            success = False # one exception is enough to make it fail
        metadata['time_start_realigned'].append(tstart+tshift)
        tstart=tstart+tshift+metadata['time_duration'][i] # update tstart by tshift_observed+duration
        i+=1
        
    if verbose:
        if success:
            print('[ok]          --> succesfully realigned')
            print('                  found n=%i episodes over the %i of the protocol ' % (len(metadata['time_start_realigned']), len(metadata['time_start'])))
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
                    smoothing_time = 20e-3,
                    # advance_time = 15e-3,
                    baseline=0, high_level=1):
    """
    we smooth the photodiode signal, with a gaussian filter of extent Tsmoothing
    Tonset = Tcrossing-3./4.*Tsmoothing
    Tcrossing is the time of crossing of half the max-min level (of the smoothed signal)
    """
    advance_time = 3./4.*smoothing_time
    smoothed = gaussian_filter1d(photodiode_signal, int(smoothing_time/(t[1]-t[0])))
    smoothed = (smoothed-smoothed.min())/(smoothed.max()-smoothed.min())
    cond = (smoothed[1:]>=0.5) & (smoothed[:-1]<=0.5)
    t0 = t[:-1][cond][0]
    return t0-advance_time, smoothed, smoothed.min()+0.5*(smoothed.max()-smoothed.min())


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
    parser.add_argument('-id', "--istart_debug", type=int, default=0)
    parser.add_argument("--smoothing_time", type=float, help='in s', default=20e-3)
    args = parser.parse_args()

    data = np.load(os.path.join(args.datafolder, 'NIdaq.npy'), allow_pickle=True).item()['analog'][0]
    metadata = np.load(os.path.join(args.datafolder, 'metadata.npy'), allow_pickle=True).item()
    VisualStim = np.load(os.path.join(args.datafolder, 'visual-stim.npy'), allow_pickle=True).item()
    if 'time_duration' not in VisualStim:
        VisualStim['time_duration'] = np.array(VisualStim['time_stop'])-np.array(VisualStim['time_start'])
    for key in VisualStim:
        metadata[key] = VisualStim[key]

    plt.plot(data[::1000][:1000])
    plt.title('photodiode-signal (subsampled/100)')
    plt.show()
    
    realign_from_photodiode(data, metadata,
                            debug=True,
                            istart_debug=args.istart_debug,
                            n_vis=args.n_vis, verbose=True)
    









