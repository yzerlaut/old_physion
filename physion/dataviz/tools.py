import numpy as np
from scipy.ndimage.filters import gaussian_filter1d # for gaussian smoothing

#########################
#########################

def scale_and_position(self, y, value=None, i=0):
    if value is None:
        value=y
    ymin, ymax = y.min(), y.max()
    if ymin<ymax:
        return shift(self, i)+\
            self.settings['increase-factor']**i*\
            (value-ymin)/(ymax-ymin)
    else:
        return shift(self, i)+value

def shift(self, i):
    return self.settings['blank-space']*i+\
        np.sum(np.power(self.settings['increase-factor'], np.arange(i)))

def convert_time_to_index(time, nwb_quantity, axis=0):
    if nwb_quantity.timestamps is not None:
        cond = nwb_quantity.timestamps[:]>=time
        if np.sum(cond)>0:
            return np.arange(nwb_quantity.timestamps.shape[0])[cond][0]
        else:
            return nwb_quantity.timestamps.shape[axis]-1
    elif nwb_quantity.starting_time is not None:
        t = time-nwb_quantity.starting_time
        dt = 1./nwb_quantity.rate
        imax = nwb_quantity.data.shape[axis]-1 # maybe shift to -1 to handle images
        return max([1, min([int(t/dt), imax-1])]) # then we add +1 / -1 in the visualization
    else:
        return 0

def convert_times_to_indices(t1, t2, nwb_quantity, axis=0):
    if nwb_quantity.timestamps is not None:
        cond = (nwb_quantity.timestamps[:]>=t1) & (nwb_quantity.timestamps[:]<=t2)
        if np.sum(cond)>0:
            return np.arange(nwb_quantity.timestamps.shape[0])[cond][np.array([0,-1])]
        else:
            return (0, nwb_quantity.timestamps.shape[axis]-1)
    elif nwb_quantity.starting_time is not None:
        T1, T2 = t1-nwb_quantity.starting_time, t2-nwb_quantity.starting_time
        dt = 1./nwb_quantity.rate
        imax = nwb_quantity.data.shape[axis]-1 # maybe shift to -1 to handle images
        return (max([1, min([int(T1/dt), imax-1])]), max([1, min([int(T2/dt), imax-1])]))
    else:
        return (0, imax)

def extract_from_times(t1, t2, nwb_quantity, axis=0):
    
    imax = nwb_quantity.data.shape[axis]-1
    
    if nwb_quantity.timestamps is not None:
        
        cond = (nwb_quantity.timestamps[:]>=t1) & (nwb_quantity.timestamps[:]<=t2)
        if np.sum(cond)>0:
            indices = np.arange(nwb_quantity.timestamps.shape[axis])[cond]
            times = nwb_quantity.timestamps[cond]
        else:
            ii, indices = np.argmin((nwb_quantity.timestamps[:]-t1)**2), [ii]
            times = [nwb_quantity.timestamps[ii]]
            
    elif nwb_quantity.starting_time is not None:
        
        dt = 1./nwb_quantity.rate
        i1, i2 = int((t1-nwb_quantity.starting_time)/dt), int((t2-nwb_quantity.starting_time)/dt)
        indices = np.arange(imax+1)[max([0, min([i1, imax])]):max([0, min([i2, imax])])]
        times = nwb_quantity.starting_time+dt*indices
        
    else:
        
        indices = [0]
        times = [0]
    
    return indices, times
    
def convert_index_to_time(index, nwb_quantity):
    """ index can be an array """
    return nwb_quantity.starting_time+index/nwb_quantity.rate

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
                            sum=False):

    if CaImaging_key=='dF/F':
        
        if sum or len(roiIndices)==1:
            y = getattr(cls, 'Fluorescence').data[:,isampling][cls.validROI_indices[roiIndices],:].sum(axis=0)
        else:
            y = getattr(cls, 'Fluorescence').data[:,isampling][cls.validROI_indices[roiIndices],:]
        sliding_min = sliding_percentile(y, 10., int(120/(cls.Neuropil.timestamps[1]-cls.Neuropil.timestamps[0]))+1) # 2 min sliding window !!
        return (y-sliding_min)/sliding_min

    elif CaImaging_key=='d(F-Fneu)/(F-Fneu)':
        
        if sum or len(roiIndices)==1:
            y2 = getattr(cls, 'Fluorescence').data[:,isampling][cls.validROI_indices[roiIndices],:].sum(axis=0)
            y1 = getattr(cls, 'Neuropil').data[:,isampling][cls.validROI_indices[roiIndices],:].sum(axis=0)
        else:
            y2 = getattr(cls, 'Fluorescence').data[:,isampling][cls.validROI_indices[roiIndices],:]
            y1 = getattr(cls, 'Neuropil').data[:,isampling][cls.validROI_indices[roiIndices],:]
        y = y2-y1
        sliding_min = sliding_percentile(y, 10., int(120/(cls.Neuropil.timestamps[1]-cls.Neuropil.timestamps[0]))+1) # 2 min sliding window !!
        return (y-sliding_min)/sliding_min
    
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
