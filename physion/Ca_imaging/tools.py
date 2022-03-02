import numpy as np
from scipy.ndimage import filters


####################################
# ---------------------------------
# DEFAULT_CA_IMAGING_OPTIONS

T_SLIDING_MIN = 60. # seconds
PERCENTILE_SLIDING_MIN = 5. # percent

# ---------------------------------
####################################

def fill_center_and_edges(N, Window, smv):
    sliding_min = np.zeros(N)
    iw = int(Window/2)+1
    if len(smv.shape)==1:
        sliding_min[:iw] = smv[0]
        sliding_min[-iw:] = smv[-1]
        sliding_min[iw:iw+smv.shape[-1]] = smv
    elif len(smv.shape)==2:
        sliding_min[:,:iw] = np.broadcast_to(smv[:,0], (iw, array.shape[0])).T
        sliding_min[:,-iw:] = np.broadcast_to(smv[:,-1], (iw, array.shape[0])).T
        sliding_min[:,iw:iw+smv.shape[-1]] = smv
    return sliding_min

def compute_sliding_percentile(array, percentile, Window,
                              with_smoothing=True):
    """
    trying numpy code to evaluate efficiently the distrib percentile over a sliding window
    making use of "stride tricks" for fast looping over the sliding window
    
        not really efficient so far... :(
    
    see: 
    https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html
    """
    
    # using a sliding "view" of the array
    view = np.lib.stride_tricks.sliding_window_view(array, Window, axis=-1)
    smv = np.percentile(view, percentile, axis=-1)
    # replacing values, N.B. need to deal with edges
    sliding_min = fill_center_and_edges(len(array), Window, smv)
    if with_smoothing:
        return filters.gaussian_filter1d(sliding_min, Window, axis=-1)
    else:
        return sliding_min

def compute_sliding_mean(array, Window):
    """ sliding average by convolution with unit array of length window
    """
    iw = int(Window/2)+1
    return fill_center_and_edges(len(array), Window,
                                 np.convolve(array, np.ones(Window), 'valid')/Window)


def compute_sliding_minmax(array, Window, sig=10):
    Flow = filters.gaussian_filter1d(array, sig)
    Flow = filters.minimum_filter1d(Flow, Window, mode='wrap')
    Flow = filters.maximum_filter1d(Flow, Window, mode='wrap')
    return Flow


def compute_sliding_F0(data, F,
                       method='minmax',
                       sliding_percentile=5.,
                       sliding_window=60):
    if method in ['maximin', 'minmax']:
        return compute_sliding_minmax(F,
                                      int(sliding_window/data.CaImaging_dt))
    elif 'percentile' in method:
        return compute_sliding_percentile(F, 
                                          sliding_percentile,
                                          int(sliding_window/data.CaImaging_dt))
    else:
        print('\n --- method not recognized --- \n ')
        


def compute_dFoF(data,  
                 neuropil_correction_factor=0.,
                 method_for_F0='maximin',
                 sliding_percentile=5,
                 sliding_window=60,
                 return_corrected_F_and_F0=False,
                 verbose=True):
    """
    compute fluorescence variation with a neuropil correction set by the 
    factor neuropil_correction_factor
    """

    if verbose:
        print('\ncalculating dFoF [...]')
        
    if (neuropil_correction_factor>1) or (neuropil_correction_factor<0):
        print('/!\ neuropil_correction_factor has to be in the interval [0.,1]')
        print('neuropil_correction_factor set to 0 !')
        neuropil_correction_factor=0.

    # performing correction 
    F = data.Fluorescence.data[:,:]-neuropil_correction_factor*data.Neuropil.data[:,:]

        
    F0 = compute_sliding_F0(data, F,
                            method=method_for_F0)

    # exclude cells with negative F0
    data.valid_roiIndices = np.min(F0, axis=1)>0

    if np.sum(~data.valid_roiIndices)>0 and verbose:
        print('\n  ** %i ROIs were discarded with the positive F0 criterion ** \n'\
              % np.sum(~data.valid_roiIndices) )
        
    data.nROIs = np.sum(data.valid_roiIndices)
    data.dFoF = (F[data.valid_roiIndices,:]-F0[data.valid_roiIndices,:])/F0[data.valid_roiIndices,:]
    data.t_dFoF = data.Neuropil.timestamps[:]

    if verbose:
        print('-> dFoF calculus done !')
    
    if return_corrected_F_and_F0:
        return F[data.valid_roiIndices,:], F0[data.valid_roiIndices,:]

    
#         F0 = self.compute_F0(new_F,
#                              sliding_minmax_as_F0=sliding_minmax_as_F0,
#                              sliding_mean_as_F0=sliding_mean_as_F0,
#                              sliding_percentile=sliding_percentile,
#                              sliding_window=sliding_window)
            
#         if np.sum(F0<=0)>1:
#             if verbose:
#                 print(' /! \ TOO STRONG NEUROPIL CORRECTION FOR FACTOR = %.2f')
#                 print('           --> NEGATIVE FLUORESCENCE !!') 
#                 print('    --> returning zero array') 
#             return None, None
#         else:
#             if type(roi_index) in [list, range, np.array, np.ndarray]:
#                 return (new_F-F0)/F0, F0
#             else:
#                 return (new_F-F0)/F0, F0
#  def compute_dFoF(data,
#                  neuropil_correction_factor=0.7,
#                  neuropil_correction_factor=0.7):
#     """
    
#     """



########################################################################################    
####################### old code, should be deprecated #################################
########################################################################################    


# numpy code for ~efficiently evaluating the distrib percentile over a sliding window
def strided_app(a, L, S ):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n))


def sliding_percentile(array, percentile, Window,
                       with_smoothing=True):

    x = np.zeros(len(array))
    y0 = strided_app(array, Window, 1)
    
    y = np.percentile(y0, percentile, axis=-1)
    
    x[:int(Window/2)] = y[0]
    x[int(Window/2):int(Window/2)+len(y)] = y
    x[-int(Window/2):] = y[-1]
    if with_smoothing:
        return gaussian_filter1d(x, iTsm)
    else:
        return x


def sliding_percentile_new(array, percentile, Window,
                           with_smoothing=True):
    """
    trying numpy code for efficiently evaluating the distrib percentile over a sliding window
    somehow making use of "stride tricks" for fast looping over the sliding window
    
        not really efficient so far... :(
    
    see: 
    https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html
    """
    sliding_min = np.zeros(array.shape)
    # using a sliding "view" of the array
    view = np.lib.stride_tricks.sliding_window_view(array, Window, axis=-1)
    smv = np.percentile(view, percentile, axis=-1)
    # replacing values, N.B. need to deal with edges
    iw = int(Window/2)+1
    if len(array.shape)==1:
        sliding_min[:iw] = smv[0]
        sliding_min[-iw:] = smv[-1]
        sliding_min[iw:iw+smv.shape[-1]] = smv
    elif len(array.shape)==2:
        sliding_min[:,:iw] = np.broadcast_to(smv[:,0], (iw, array.shape[0])).T
        sliding_min[:,-iw:] = np.broadcast_to(smv[:,-1], (iw, array.shape[0])).T
        sliding_min[:,iw:iw+smv.shape[-1]] = smv
    if with_smoothing:
        return gaussian_filter1d(sliding_min, Window, axis=-1)
    else:
        return sliding_min

def compute_CaImaging_trace(data, CaImaging_key, roiIndices,
                            with_baseline=False,
                            new_method=False,
                            T_sliding_min=T_SLIDING_MIN,
                            percentile_sliding_min=PERCENTILE_SLIDING_MIN):
    """
    # /!\ the validROI_indices are used here  /!\ (July 2021: DEPRECATED NOW STORING ONLY THE VALID ROIS)
    """

    if CaImaging_key in ['Fluorescence', 'Neuropil', 'Deconvolved']:
        return getattr(data, CaImaging_key).data[data.validROI_indices[roiIndices], :]
        
    elif CaImaging_key in ['dF/F', 'dFoF']:
        """
        computes dF/F with a smotthed sliding percentile
        """
        iTsm = int(T_sliding_min/data.CaImaging_dt)

        if not new_method:
            DFoF, FMIN = [], []
            for ROI in data.validROI_indices[np.array(roiIndices)]:
                Fmin = sliding_percentile(data.Fluorescence.data[ROI,:], percentile_sliding_min, iTsm,
                                          with_smoothing=True) # sliding percentile + smoothing
                if np.sum(Fmin<=0)==0:
                    DFoF.append((data.Fluorescence.data[ROI,:]-Fmin)/Fmin)
                else:
                    DFoF.append(0*data.Fluorescence.data[0,:]) # let's just put 0

                if with_baseline:
                    FMIN.append(Fmin)
        else:
            FMIN = sliding_percentile_new(data.Fluorescence.data[data.validROI_indices[np.array(roiIndices)],:], percentile_sliding_min, iTsm,
                                          with_smoothing=True) # sliding percentile + smoothing
            DFoF = (data.Fluorescence.data[data.validROI_indices[np.array(roiIndices)],:]-FMIN)/FMIN
                    
        if with_baseline:
            return np.array(DFoF), np.array(FMIN)
        else:
            return np.array(DFoF)

    elif 'd(F-' in CaImaging_key:
        """
        computes dF/F with a smotthed sliding percentile
        """
        if '*Fneu' in CaImaging_key:
            coef = float(CaImaging_key.replace('d(F-', '').replace('*Fneu)', ''))
        else:
            coef = 1. # d(F-Fneu)
            
        iTsm = int(T_sliding_min/data.CaImaging_dt)

        DFoF, FMIN = [], []
        for ROI in data.validROI_indices[np.array(roiIndices)]:
            Fmin = sliding_percentile(data.Fluorescence.data[ROI,:]-coef*data.Neuropil.data[ROI,:],
                                      percentile_sliding_min, iTsm, with_smoothing=True) # sliding percentile + smoothing
            if np.sum(Fmin<0)>0:
                print(' /!\ sliding percentile gets negative -> pb !  ')

            DFoF.append((data.Fluorescence.data[ROI,:]-coef*data.Neuropil.data[ROI,:]-Fmin)/Fmin)
                
            if with_baseline:
                FMIN.append(Fmin)
        if with_baseline:
            return np.array(DFoF), np.array(FMIN)
        else:
            return np.array(DFoF)
        
    elif CaImaging_key in ['F-Fneu', 'dF']:
        DF = []
        for ROI in data.validROI_indices[roiIndices]: # /!\ validROI_indices here /!\
            DF.append(data.Fluorescence.data[ROI,:]-data.Neuropil.data[ROI,:])
        return np.array(DF)
    
    elif 'F-' in CaImaging_key: # key of the form "F-0.853*Fneu"
        coef = float(CaImaging_key.replace('F-', '').replace('*Fneu', ''))
        DF = []
        for ROI in data.validROI_indices[roiIndices]: # /!\ validROI_indices here /!\
            DF.append(data.Fluorescence.data[ROI,:]-coef*data.Neuropil.data[ROI,:])
        return np.array(DF)
    else:
        print(20*'--')
        print(' /!\ "%s" not recognized to process the CaImaging signal /!\ ' % CaImaging_key)
        print(20*'--')


def compute_CaImaging_raster(data, CaImaging_key,
                             roiIndices='all',
                             normalization='None',
                             compute_CaImaging_options=dict(T_sliding_min=T_SLIDING_MIN,
                                                            percentile_sliding_min=PERCENTILE_SLIDING_MIN),
                             verbose=False):
    """
    normalization can be: 'None', 'per line'

    """

    if (not type(roiIndices) in [list, np.array]) and (roiIndices=='all'):
        roiIndices = np.arange(data.iscell.sum())

    if verbose:
        print('computing raster [...]')
    raster = compute_CaImaging_trace(data, CaImaging_key, roiIndices, **compute_CaImaging_options)

    if verbose:
        print('normalizing raster [...]')
    if normalization in ['per line', 'per-line', 'per cell', 'per-cell']:
        for n in range(raster.shape[0]):
            Fmax, Fmin = raster[n,:].max(), raster[n,:].min()
            if Fmax>Fmin:
                raster[n,:] = (raster[n,:]-Fmin)/(Fmax-Fmin)

    return raster
              







        
