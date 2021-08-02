import numpy as np
from scipy.ndimage.filters import gaussian_filter1d # for gaussian smoothing

####################################
# ---------------------------------
# DEFAULT_CA_IMAGING_OPTIONS

T_SLIDING_MIN = 60. # seconds
PERCENTILE_SLIDING_MIN = 5. # percent

# ---------------------------------
####################################


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
    x[int(Window/2):int(Window/2)+len(y)] = y
    x[-int(Window/2):] = y[-1]
    return x

def compute_CaImaging_trace(data, CaImaging_key, roiIndices,
                            with_baseline=False,
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

        DFoF, FMIN = [], []
        for ROI in data.validROI_indices[np.array(roiIndices)]:
            Fmin = sliding_percentile(data.Fluorescence.data[ROI,:], percentile_sliding_min, iTsm) # sliding percentile
            Fmin = gaussian_filter1d(Fmin, iTsm) # + smoothing
            DFoF.append((data.Fluorescence.data[ROI,:]-Fmin)/Fmin)
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
                                                            percentile_sliding_min=PERCENTILE_SLIDING_MIN)):
    """
    normalization can be: 'None', 'per line'

    """

    if roiIndices=='all':
        roiIndices = np.arange(data.iscell.sum())

    raster = compute_CaImaging_trace(data, CaImaging_key, roiIndices, **compute_CaImaging_options)

    if normalization in ['per line', 'per-line']:
        for n in raster.shape[0]:
            raster[n,:] = (raster[n,:]-raster[n,:].min())/(raster[n,:].max()-raster[n,:].min())

    return raster
              







        
