import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import interp1d

def resample_signal(original_signal,
                    original_freq=1e4,
                    t_sample=None,
                    new_freq=1e3,
                    pre_smoothing=0,
                    post_smoothing=0,
                    verbose=False):

    if verbose:
        print('resampling signal [...]')
        

    if (pre_smoothing*original_freq)>1:
        if verbose:
            print(' - gaussian smoothing - pre')
        signal = gaussian_filter1d(original_signal, int(pre_smoothing*original_freq), mode='nearest')
    else:
        signal = original_signal
        
    if t_sample is None:
       t_sample = np.arange(len(signal))/original_freq
       
    if verbose:
        print(' - signal interpolation')
    func = interp1d(t_sample, signal)
    new_t = np.arange(int((t_sample[-1]-t_sample[0])*new_freq))/new_freq
    new_signal = func(new_t)

    if (post_smoothing*new_freq)>1:
        if verbose:
            print(' - gaussian smoothing - post')
        new_signal = gaussian_filter1d(new_signal, int(post_smoothing*new_freq), mode='nearest')
        
    return new_t, new_signal
    

if __name__=='__main__':


    """
    some common tools for analysis
    """

    import matplotlib.pylab as plt
    import sys, os, pathlib











