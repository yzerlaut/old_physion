import os
import numpy as np

from scipy.ndimage.filters import gaussian_filter1d # for gaussian smoothing

def load_data_from_folder(folder,
                          soft_prefix='suite2p',
                          plane_prefix='plane0'):

    data = {}
    for key in ['F', 'stat', 'iscell', 'Fneu']:
        data[key] = np.load(os.path.join(folder, soft_prefix, plane_prefix, '%s.npy' % key), allow_pickle=True)

    return data



# numpy code for ~efficiently evaluating the distrib percentile over a sliding window
def strided_app(a, L, S ):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n))


def sliding_percentile(array, percentile, Window):

    x = np.zeros(len(array))
    y0 = strided_app(array, Window, 1)

    y = np.percentile(y0, percentile, axis=-1)
    
    x[:int(Window/2)] = y[0]
    x[-int(Window/2):] = y[-1]
    x[int(Window/2)-1:-int(Window/2)] = y
    
    return x


def from_raw_data_to_deltaFoverF(folder,
                                 freq_acq=30.,
                                 fraction_of_substracted_neuropil=0.7,
                                 percentile_threshold_for_baseline=0.2,
                                 sliding_window_for_baseline = 30.,
                                 fluo_factor_wrt_neuropil_for_inclusion = 1.5,
                                 verbose=False):

    if verbose:
        data = load_data_from_folder(folder)
        print(' data from: "%s" succesfully loaded ' % folder)

    print('----> Pre-processing the fluorescence of n=%i ROIs across n=%i time samples' % data['F'].shape)
    
    if verbose:
        print(' 1) [...] applying the ROI selection and the neuropil criteria')
    data['neuropil_cond'] = (np.mean(data['F'], axis=1)>fluo_factor_wrt_neuropil_for_inclusion*np.mean(data['Fneu'], axis=1))
    data['iscell'] = np.array(data['iscell'][:,0], dtype=bool)
    iscell = data['iscell'] & data['neuropil_cond']
    data['fluo_valid_cells'] = data['F'][iscell,:]
    print('----> n=%i ROIs are considered as valid cells' % np.sum(iscell))
    print
    
    if verbose:
        print(' 2) [...] substracting neuropil contamination')
    data['fluo_valid_cells'] -= fraction_of_substracted_neuropil*data['Fneu'][iscell,:]

    
    if verbose:
        print(' 3) [...] calculating sliding baseline per cell')
    Twidth = int(sliding_window_for_baseline*freq_acq) # window in sample units 
    # sliding minimum using the max filter of scipy, followed by gaussian smoothing
    data['sliding_min'] = 0*data['fluo_valid_cells']
    Twidth = int(sliding_window_for_baseline*freq_acq)
    for icell in range(data['fluo_valid_cells'].shape[0]):
        data['sliding_min'][icell,:] = gaussian_filter1d(\
                   sliding_percentile(data['fluo_valid_cells'][icell,:],
                                      percentile_threshold_for_baseline,
                                      Twidth), Twidth)
        
    if verbose:
        print(' 4) [...] performing DeltaF/F normalization')
    data['dFoF'] = np.divide((data['fluo_valid_cells']-data['sliding_min']),data['sliding_min'])

    # some useful quantites
    data['t'] = np.arange(int(data['fluo_valid_cells'].shape[1]))/freq_acq # time axis

    # formatted data
    Data = {}
    data['mean_Data'], data['std_Data'], data['norm_Data'] = {}, {}, {}
    
    for i in range(data['fluo_valid_cells'].shape[0]):
        key = 'cell%i' % (i+1)
        Data[key] = data['dFoF'][i, :]
        data['mean_Data'][key] = Data[key].mean()
        data['std_Data'][key] = Data[key].std()
        data['norm_Data'][key] = (Data[key]-Data[key].mean())/Data[key].std()
    if verbose:
        print('----> Pre-processing done !')

    return data, Data


