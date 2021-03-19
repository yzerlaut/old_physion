# general modules
import pynwb, os, sys
import numpy as np
import matplotlib.pylab as plt
from scipy.ndimage import gaussian_filter    
# custom modules
sys.path.append('.')
from physion.analysis.read_NWB import Data
from physion.visual_stim.psychopy_code.stimuli import build_stim

class DataWithStim(Data):
    
    def __init__(self, filename):
        super().__init__(filename)
        self.metadata['load_from_protocol_data'], self.metadata['no-window'] = True, True
        self.visual_stim = build_stim(self.metadata, no_psychopy=True)

    def reverse_correlation(self, roiIndex):
        
        # super simple estimate for now
        self.dF = (data.Fluorescence.data[roiIndex,:]-data.Neuropil.data[roiIndex,:])/data.Neuropil.data[roiIndex,:]
        self.t = data.Fluorescence.timestamps[:]
        
        full_img, cum_weight = np.zeros(self.visual_stim.screen['resolution'], dtype=float).T, 0
    
        for i in np.arange(self.nwbfile.stimulus['time_start_realigned'].num_samples):
            tstart = self.nwbfile.stimulus['time_start_realigned'].data[i]
            tstop = self.nwbfile.stimulus['time_stop_realigned'].data[i]
            cond = (self.t>tstart) & (self.t<tstop)
            weight = np.trapz(self.dF[cond], self.t[cond])
            full_img += weight*self.visual_stim.get_image(i)
            cum_weight += weight
        return full_img/cum_weight

filename = os.path.join(os.path.expanduser('~'), 'DATA', '2021_03_11-17-32-34.nwb')
data = DataWithStim(filename)

for i in range(np.sum(data.iscell)):
    print('ROI#', i+1)
    fig, ax = plt.subplots(1, figsize=(7,4))
    img = data.reverse_correlation(i)
    img = img-np.mean(img)
    plt.imshow(gaussian_filter(img, (20,20)),
               vmin=-np.max(np.abs(img)), vmax=np.max(np.abs(img)), cmap=plt.cm.PiYG)
    ax.axis('off')
    fig.savefig(os.path.join(os.path.expanduser('~'), 'Desktop', 'RF', 'ROI#%i.png' % (i+1)))    
    plt.close()
