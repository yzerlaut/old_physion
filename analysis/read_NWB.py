import numpy as np
import pynwb, time, ast

def init(self):

    self.tlim = None
    self.df_name = ''
    self.description = ''
    self.keys = []
    
    
def read(self, verbose=False):

    if verbose:
        t0 = time.time()
    
    data = {}
    self.description = ''

    self.metadata = ast.literal_eval(\
                    self.nwbfile.session_description)

    self.tlim, safety_counter = None, 0
    while (self.tlim is None) and (safety_counter<10):
        for key in self.nwbfile.acquisition:
            try:
                self.tlim = [self.nwbfile.acquisition[key].starting_time,
                             self.nwbfile.acquisition[key].starting_time+self.nwbfile.acquisition[key].data.shape[0]/self.nwbfile.acquisition[key].rate]
            except BaseException as be:
                pass
    if self.tlim is None:
        self.tlim = [0, 50] # bad for movies
        
    self.df_name = self.nwbfile.session_start_time.strftime("%Y/%m/%d -- %H:%M:%S")+' ---- '+self.nwbfile.experiment_description


    self.description += 'Visual-Stim:\n'
    if 'ophys' in self.nwbfile.processing:
        self.Segmentation = self.nwbfile.processing['ophys'].data_interfaces['ImageSegmentation'].plane_segmentations['PlaneSegmentation']
        self.pixel_masks_index = self.Segmentation.columns[0].data[:]
        self.pixel_masks = self.Segmentation.columns[1].data[:]
        self.iscell = self.Segmentation.columns[2].data[:,0].astype(bool)
        self.validROI_indices = np.arange(len(self.iscell))[self.iscell]
        self.Fluorescence = self.nwbfile.processing['ophys'].data_interfaces['Fluorescence'].roi_response_series['Fluorescence']
        self.Neuropil = self.nwbfile.processing['ophys'].data_interfaces['Neuropil'].roi_response_series['Neuropil']
        self.Deconvolved = self.nwbfile.processing['ophys'].data_interfaces['Deconvolved'].roi_response_series['Deconvolved']
    else:
        self.Segmentation, self.Fluorescence, self.iscell,\
            self.Neuropil, self.Deconvolved = None, None, None, None, None

    self.keys = []
    for key in self.nwbfile.stimulus.keys():
        if key not in ['index', 'time_start', 'time_start_realigned', 'time_stop', 'time_stop_realigned', 'visual-stimuli']:
            if len(np.unique(self.nwbfile.stimulus[key].data[:]))>1:
                s = '-*  N-%s = %i' % (key,len(np.unique(self.nwbfile.stimulus[key].data[:])))
                self.description += s+(35-len(s))*' '+'[%.1f, %.1f]\n' % (np.min(self.nwbfile.stimulus[key].data[:]),
                                                                        np.max(self.nwbfile.stimulus[key].data[:]))
                self.keys.append(key)
            else:
                self.description += '- %s=%.1f\n' % (key, np.unique(self.nwbfile.stimulus[key].data[:]))

    if 'time_start_realigned' in self.nwbfile.stimulus.keys():
        self.description += '   =>  completed N=%i/%i episodes  <=' %(self.nwbfile.stimulus['time_start_realigned'].data.shape[0],
                                                           self.nwbfile.stimulus['time_start'].data.shape[0])

        
    if verbose:
        print('NWB-file reading time: %.1f' % (time.time()-t0))

