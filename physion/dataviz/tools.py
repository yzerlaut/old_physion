import numpy as np

#########################
#########################

def add_bar_annotations(ax,
                        Xbar=0, Xbar_label='',
                        Ybar=0, Ybar_label='',
                        lw=2, fontsize=10):

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.plot(xlim[0]+Xbar*np.arange(2), ylim[1]*np.ones(2), 'k-', lw=lw)
    ax.annotate(Xbar_label, (xlim[0], ylim[1]), fontsize=fontsize)
    ax.plot(xlim[0]*np.ones(2), ylim[1]-Ybar*np.arange(2), 'k-', lw=lw)
    ax.annotate(Ybar_label, (xlim[0], ylim[1]), fontsize=fontsize, ha='right', va='top', rotation=90)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    
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
            return nwb_quantity.timestamps.shape[0]-1
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
    if nwb_quantity.timestamps is not None:
        return nwb_quantity.timestamps[index]
    else:
        return nwb_quantity.starting_time+index/nwb_quantity.rate









