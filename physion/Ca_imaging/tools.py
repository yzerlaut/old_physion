import numpy as np
from scipy.ndimage.filters import gaussian_filter1d # for gaussian smoothing

#########################
#########################

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
    # CHECK THE GENERALITY HERE
    if (len(y)-int(Window/2))%2==1:
        x[int(Window/2):-int(Window/2)] = y
    else:
        x[int(Window/2)-1:-int(Window/2)] = y
    
    return x

def compute_CaImaging_trace(cls, CaImaging_key, roiIndices,
                            Tsliding=60, percentile=5.,
                            with_sliding_mean = False,
                            sum=False):

    if CaImaging_key in ['Fluorescence', 'Neuropil', 'Deconvolved']:
        return getattr(cls, CaImaging_key).data[roiIndices, :]
        
    elif CaImaging_key in ['dF/F', 'dFoF']:
        """
        computes dF/F with a smotthed sliding percentile
        """
        iTsm = int(Tsliding/cls.CaImaging_dt)

        DFoF = []
        for ROI in cls.validROI_indices[roiIndices]: # /!\ validROI_indices here /!\
            Fmin = sliding_percentile(cls.Fluorescence.data[ROI,:], percentile, iTsm) # sliding percentile
            Fmin = gaussian_filter1d(Fmin, Tsliding) # + smoothing
            DFoF.append((cls.Fluorescence.data[ROI,:]-Fmin)/Fmin)
        return np.array(DFoF)

    elif CaImaging_key in ['F-Fneu', 'dF']:
        DF = []
        for ROI in cls.validROI_indices[roiIndices]: # /!\ validROI_indices here /!\
            DF.append(cls.Fluorescence.data[ROI,:]-cls.Neuropil.data[ROI,:])
        return np.array(DF)
    
    elif 'F-' in CaImaging_key: # key of the form "F-0.85*Fneu"
        coef = float(CaImaging_key.replace('F-', '').replace('*Fneu', ''))
        DF = []
        for ROI in cls.validROI_indices[roiIndices]: # /!\ validROI_indices here /!\
            DF.append(cls.Fluorescence.data[ROI,:]-coef*cls.Neuropil.data[ROI,:])
        return np.array(DF)
    else:
        print(20*'--')
        print(' /!\ "%s" not recognized to process the CaImaging signal /!\ ')
        print(20*'--')
        return None

              
