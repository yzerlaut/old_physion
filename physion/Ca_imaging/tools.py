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
    x[int(Window/2)-1:-int(Window/2)] = y
    
    return x

def compute_CaImaging_trace(cls, CaImaging_key, isampling, roiIndices,
                            Tsliding_min=120, min_percentile=10,
                            with_sliding_mean = False,
                            sum=False):

    if ('dF' in CaImaging_key) or ('d(F-' in CaImaging_key):
        iTsm = int(Tsliding_min/(cls.Neuropil.timestamps[1]-cls.Neuropil.timestamps[0]))+1
        if iTsm>=len(isampling):
            print(' /!\ too short sample to have a sliding percentile with time scale to %.1fs' % Tsliding_min)
            print(' /!\  ---> dF/F as sliding percentile removed taking the ')
            iTsm = 1
            
    if CaImaging_key=='dF/F':
        if sum or len(roiIndices)==1:
            y = getattr(cls, 'Fluorescence').data[:,isampling][cls.validROI_indices[roiIndices],:].sum(axis=0).flatten()
            sliding_min = sliding_percentile(y, min_percentile, iTsm)
            if with_sliding_mean:
                return (y-sliding_min)/sliding_min, sliding_min
            else:
                return (y-sliding_min)/sliding_min
        else:
            Y, SM = [], []
            for i in cls.validROI_indices[roiIndices]:
                y = getattr(cls, 'Fluorescence').data[:,isampling][i,:]
                print(y.size, int(Tsliding_min/(cls.Neuropil.timestamps[1]-cls.Neuropil.timestamps[0]))+1)
                sliding_min = sliding_percentile(y, sliding_percentile, iTsm)
                if with_sliding_mean:
                    SM.append(sliding_min)
                Y.append((y-sliding_min)/sliding_min)
            if with_sliding_mean:
                return np.array(Y), np.array(SM) 
            else:
                return np.array(Y)
                
        
    # elif CaImaging_key=='d(F-Fneu)/(F-Fneu)':
        
    #     if sum or len(roiIndices)==1:
    #         y2 = getattr(cls, 'Fluorescence').data[:,isampling][cls.validROI_indices[roiIndices],:].sum(axis=0)
    #         y1 = getattr(cls, 'Neuropil').data[:,isampling][cls.validROI_indices[roiIndices],:].sum(axis=0)
    #     else:
    #         y2 = getattr(cls, 'Fluorescence').data[:,isampling][cls.validROI_indices[roiIndices],:]
    #         y1 = getattr(cls, 'Neuropil').data[:,isampling][cls.validROI_indices[roiIndices],:]
    #     y = y2-y1
    #     sliding_min = sliding_percentile(y, 10., int(120/(cls.Neuropil.timestamps[1]-cls.Neuropil.timestamps[0]))+1) # 2 min sliding window !!
    #     return (y-sliding_min)/sliding_min
    
    elif 'F-' in CaImaging_key:
        coef = float(CaImaging_key.replace('F-', '').replace('*Fneu', ''))
        if sum or len(roiIndices)==1:
            return getattr(cls, 'Fluorescence').data[:,isampling][cls.validROI_indices[roiIndices],:].sum(axis=0)-\
                coef*getattr(cls, 'Neuropil').data[:,isampling][cls.validROI_indices[roiIndices],:].sum(axis=0)
        else:
            return getattr(cls, 'Fluorescence').data[:,isampling][cls.validROI_indices[roiIndices],:]-\
                coef*getattr(cls, 'Neuropil').data[:,isampling][cls.validROI_indices[roiIndices],:]

    else:
        if sum or len(roiIndices)==1:
            return getattr(cls, CaImaging_key).data[:,isampling][cls.validROI_indices[roiIndices],:].sum(axis=0)
        else:
            return getattr(cls, CaImaging_key).data[:,isampling][cls.validROI_indices[roiIndices],:]
