# general modules
import pynwb, os, sys
import numpy as np
import matplotlib.pylab as plt
from scipy.ndimage import gaussian_filter    
# custom modules
sys.path.append('.')
from physion.analysis.read_NWB import Data
from physion.Ca_imaging.tools import compute_CaImaging_trace
from physion.visual_stim.psychopy_code.stimuli import build_stim

class DataWithStim(Data):
    
    def __init__(self, filename):
        super().__init__(filename)
        self.metadata['load_from_protocol_data'], self.metadata['no-window'] = True, True
        self.visual_stim = build_stim(self.metadata, no_psychopy=True)

    def reverse_correlation(self, roiIndex, subquantity='Fluorescence',
                            metrics='mean'):

        dF = compute_CaImaging_trace(self, subquantity, [roiIndex]).sum(axis=0) # validROI inside
        self.t = data.Fluorescence.timestamps[:]
        
        full_img, cum_weight = np.zeros(self.visual_stim.screen['resolution'], dtype=float).T, 0
    
        for i in np.arange(self.nwbfile.stimulus['time_start_realigned'].num_samples):
            tstart = self.nwbfile.stimulus['time_start_realigned'].data[i]
            tstop = self.nwbfile.stimulus['time_stop_realigned'].data[i]
            cond = (self.t>tstart) & (self.t<tstop)
            weight = np.inf
            if metrics == 'mean':
                weight = np.mean(dF[cond])
            elif np.sum(cond)>0 and (metrics=='max'):
                weight = np.max(dF[cond])
            if np.isfinite(weight):
                full_img += weight*2*(self.visual_stim.get_image(i)-.5)
                cum_weight += weight
            else:
                print('For episode #%i in t=(%.1f, %.1f), pb in estimating the weight !' % (i, tstart, tstop) )
                
        return full_img/cum_weight

filename = os.path.join(os.path.expanduser('~'), 'DATA', 'Wild_Type', '2021_03_11-17-32-34.nwb')
data = DataWithStim(filename)

for i in range(np.sum(data.iscell)):
    print('ROI#', i+1)
    fig, ax = plt.subplots(1, figsize=(7,4))
    img = data.reverse_correlation(i, subquantity='Deconvolved', metrics='max')
    img = gaussian_filter(img, (10,10))
    ax.imshow(img, cmap=plt.cm.PiYG,
               vmin=-np.max(np.abs(img)), vmax=np.max(np.abs(img)))
    ax.axis('off')
    fig.savefig(os.path.join(os.path.expanduser('~'), 'Desktop', 'RF', 'ROI#%i.png' % (i+1)))    
    plt.close()






