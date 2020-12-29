import numpy as np
import os

def realign_from_photodiode(signal, metadata,
                            debug=False, verbose=True):

    if verbose:
        print('---> Realigning data with respect to photodiode signal [...] ')

    success = True

    
    # extract parameters
    dt = 1./metadata['NIdaq-acquisition-frequency']
    t = np.arange(len(signal))*dt
    
    tlim, tnew = [0, t[-1]], 0

    t0 = metadata['time_start'][0]
    length = metadata['presentation-duration']+metadata['presentation-interstim-period']
    npulses = int(metadata['presentation-duration'])
    metadata['time_start_realigned'] = []
    Nepisodes = np.sum(metadata['time_start']<tlim[1])
    for i in range(Nepisodes):
        cond = (t>=t0-.3) & (t<=t0+length)
        try:
            tnew, integral, threshold = find_onset_time(t[cond]-t0,
                                                        signal[cond], npulses)
            if debug and ((i<3) or (i>Nepisodes-3)):
                fig, ax = plt.subplots()

                ax.plot(t[cond], normalize_signal(signal[cond])[0], label='photodiode-signal')
                norm_integral, norm, xmin = normalize_signal(integral)
                ax.plot(t[cond], norm_integral, label='integral')
                ax.plot(t[cond], integral*0+norm*(threshold-xmin), label='threshold')
                ax.plot([t0,t0,], [0,1], 'k:', label='onset')
                plt.xlabel('time (s)')
                plt.ylabel('norm. signals')
                ax.legend(frameon=False)
                plt.show()
        except Exception:
            success = False # one exception is enough to make it fail
        t0+=tnew
        metadata['time_start_realigned'].append(t0)
        t0+=length

    if verbose:
        if success:
            print('[ok]          --> succesfully realigned')
        else:
            print('[X]          --> realignement failed')
    if success:
        metadata['time_start_realigned'] = np.array(metadata['time_start_realigned'])
        metadata['time_stop_realigned'] = metadata['time_start_realigned']+\
            metadata['presentation-duration']
    else:
        metadata['time_start_realigned'] = np.array([])
        metadata['time_stop_realigned'] = np.array([])
    return success, metadata


def find_onset_time(t, photodiode_signal, npulses,
                    time_for_threshold=10e-3):
    """
    the threshold of integral increase corresponds to spending X-ms at half the maximum
    """
    H, bins = np.histogram(photodiode_signal, bins=100)
    baseline = bins[np.argmax(H)]

    integral = np.cumsum(photodiode_signal-baseline)*(t[1]-t[0])

    threshold = time_for_threshold*np.max(photodiode_signal)
    t0 = t[np.argwhere(integral>threshold)[0][0]]
    return t0-time_for_threshold, integral, threshold

def normalize_signal(x):
    # just to plot above
    norm, xmin = 1./(np.max(x)-np.min(x)), np.min(x)
    return norm*(x-xmin), norm, xmin

if __name__=='__main__':

    import matplotlib.pylab as plt

    fn = sys.argv[-1]
    # fn = '/media/yann/Yann/2020_11_10/16-59-49/'
    
    data = np.load(os.path.join(fn, 'NIdaq.npy'), allow_pickle=True).item()['analog'][0]
    metadata = np.load(os.path.join(fn, 'metadata.npy'), allow_pickle=True).item()
    VisualStim = np.load(os.path.join(fn, 'visual-stim.npy'), allow_pickle=True).item()
    for key in ['time_start', 'time_stop']:
        metadata[key] = VisualStim[key]

    realign_from_photodiode(data, metadata, debug=True)
    
