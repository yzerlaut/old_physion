import pynwb, time, ast, sys, pathlib, os
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import day_folder, list_dayfolder, get_files_with_extension
from visual_stim.psychopy_code.stimuli import build_stim

def init(self):

    self.tlim = None
    self.df_name = ''
    self.description = ''
    self.keys = []

class Data:
    
    """
    a basic class to be the parent of specific applications
    """
    
    def __init__(self, filename,
                 verbose=False, with_visual_stim=False):
        
        read(self, filename,
             with_visual_stim=with_visual_stim,
             verbose=verbose)

        
    def close(self):
        self.io.close()
        
            
    def init_visual_stim(self):
        self.metadata['load_from_protocol_data'], self.metadata['no-window'] = True, True
        self.visual_stim = build_stim(self.metadata, no_psychopy=True)

        
    def get_protocol_id(protocol_name):
        # TO BE DONE
        return -1

    
    def get_protocol_cond(self, protocol_id):
        """
        ## a recording can have multiple protocols inside
        -> find the condition of a given protocol ID
        """

        if ('protocol_id' in self.nwbfile.stimulus) and (len(np.unique(self.nwbfile.stimulus['protocol_id'].data[:]))>1):
            Pcond = (self.nwbfile.stimulus['protocol_id'].data[:]==protocol_id)
        else:
            Pcond = np.ones(self.nwbfile.stimulus['time_start'].data.shape[0], dtype=bool)
            
        # limiting to available episodes
        Pcond[np.arange(len(Pcond))>=self.nwbfile.stimulus['time_start_realigned'].num_samples] = False

        return Pcond
        
    
    def get_stimulus_conditions(self, X, K, protocol_id):
        """
        find the episodes where the keys "K" have the values "X"
        """
        Pcond = self.get_protocol_cond(protocol_id)
        
        if len(K)>0:
            CONDS = []
            XK = np.meshgrid(*X)
            for i in range(len(XK[0].flatten())): # looping over joint conditions
                cond = np.ones(np.sum(Pcond), dtype=bool)
                for k, xk in zip(K, XK):
                    cond = cond & (self.nwbfile.stimulus[k].data[Pcond]==xk.flatten()[i])
                CONDS.append(cond)
            return CONDS
        else:
            return [np.ones(np.sum(Pcond), dtype=bool)]
            
        
    
def read(self, filename, verbose=False, with_tlim=True,
         metadata_only=False, with_visual_stim=False):

    self.io = pynwb.NWBHDF5IO(filename, 'r')
    self.nwbfile = self.io.read()
    self.df_name = self.nwbfile.session_start_time.strftime("%Y/%m/%d -- %H:%M:%S")+' ---- '+\
        self.nwbfile.experiment_description
    
    if verbose:
        t0 = time.time()
    
    data = {}

    self.metadata = ast.literal_eval(\
                    self.nwbfile.session_description)

    if self.metadata['protocol']=='None':
        self.description = 'Spont. Act.\n'
    else:
        self.description = 'Visual-Stim:\n'

    # deal with multi-protocols
    if ('Presentation' in self.metadata) and (self.metadata['Presentation']=='multiprotocol'):
        self.protocols, ii = [], 1
        while ('Protocol-%i' % ii) in self.metadata:
            self.protocols.append(self.metadata['Protocol-%i' % ii].replace('.json',''))
            self.description += '- %s \n' % self.protocols[ii-1]
            ii+=1
    else:
        self.protocols = [self.metadata['protocol']]
        self.description += '- %s \n' % self.metadata['protocol']
    self.protocols = np.array(self.protocols, dtype=str)

    if not metadata_only or with_tlim:
        self.tlim, safety_counter = None, 0
        while (self.tlim is None) and (safety_counter<10):
            for key in self.nwbfile.acquisition:
                try:
                    self.tlim = [self.nwbfile.acquisition[key].starting_time,
                                 self.nwbfile.acquisition[key].starting_time+\
                                 self.nwbfile.acquisition[key].data.shape[0]/self.nwbfile.acquisition[key].rate]
                except BaseException as be:
                    pass
        if self.tlim is None:
            self.tlim = [0, 50] # bad for movies

    if not metadata_only:
        
        if 'ophys' in self.nwbfile.processing:

            self.Segmentation = self.nwbfile.processing['ophys'].data_interfaces['ImageSegmentation'].plane_segmentations['PlaneSegmentation']
            self.pixel_masks_index = self.Segmentation.columns[0].data[:]
            self.pixel_masks = self.Segmentation.columns[1].data[:]

            self.Fluorescence = self.nwbfile.processing['ophys'].data_interfaces['Fluorescence'].roi_response_series['Fluorescence']
            self.Neuropil = self.nwbfile.processing['ophys'].data_interfaces['Neuropil'].roi_response_series['Neuropil']
            self.Deconvolved = self.nwbfile.processing['ophys'].data_interfaces['Deconvolved'].roi_response_series['Deconvolved']
            self.CaImaging_dt = (self.Neuropil.timestamps[1]-self.Neuropil.timestamps[0])

            if len(self.Segmentation.columns)>2 and (self.Segmentation.columns[2].name=='iscell') : # DEPRECATED
                self.iscell = self.Segmentation.columns[2].data[:,0].astype(bool)
                self.validROI_indices = np.arange(len(self.iscell))[self.iscell]
            else:
                self.iscell = np.ones(len(self.Fluorescence.data[:,0]), dtype=bool)
                self.validROI_indices = np.arange(len(self.Fluorescence.data[:,0]))
                
        else:
            self.Segmentation, self.Fluorescence, self.iscell, self.validROI_indices,\
                self.Neuropil, self.Deconvolved = None, None, None, None, None, None

        if 'Pupil' in self.nwbfile.processing:
            pd = str(self.nwbfile.processing['Pupil'].description)
            if len(pd.split('pix_to_mm='))>1:
                self.FaceCamera_mm_to_pix = int(1./float(pd.split('pix_to_mm=')[-1]))
            else:
                self.FaceCamera_mm_to_pix = 1


        if 'FaceMotion' in self.nwbfile.processing:
            fd = str(self.nwbfile.processing['FaceMotion'].description)
            self.FaceMotion_ROI = [int(i) for i in fd.split('y0,dy)=(')[1].split(')')[0].split(',')]
                
        #     self.t_pupil = self.nmbfile.processing['Pupil']
        #     self.nwbfile.processing['Pupil'].data_interfaces['cx']

            
        # FIND A BETTER WAY TO DESCRIBE
        # if self.metadata['protocol']!='multiprotocols':
        #     self.keys = []
        #     for key in self.nwbfile.stimulus.keys():
        #         if key not in ['index', 'time_start', 'time_start_realigned',
        #                        'time_stop', 'time_stop_realigned', 'visual-stimuli', 'frame_run_type']:
        #             if len(np.unique(self.nwbfile.stimulus[key].data[:]))>1:
        #                 s = '-*  N-%s = %i' % (key,len(np.unique(self.nwbfile.stimulus[key].data[:])))
        #                 self.description += s+(35-len(s))*' '+'[%.1f, %.1f]\n' % (np.min(self.nwbfile.stimulus[key].data[:]),
        #                                                                         np.max(self.nwbfile.stimulus[key].data[:]))
        #                 self.keys.append(key)
        #             else:
        #                 self.description += '- %s=%.1f\n' % (key, np.unique(self.nwbfile.stimulus[key].data[:]))
                    

        if 'time_start_realigned' in self.nwbfile.stimulus.keys():
            self.description += ' =>  completed N=%i/%i episodes  <=' %(self.nwbfile.stimulus['time_start_realigned'].data.shape[0],
                                                               self.nwbfile.stimulus['time_start'].data.shape[0])

    if with_visual_stim:
        self.init_visual_stim()

    if verbose:
        print('NWB-file reading time: %.1fms' % (1e3*(time.time()-t0)))


class DummyParent:
    def __init__(self):
        pass

def scan_folder_for_NWBfiles(folder, verbose=True):

    if verbose:
        print('inspecting the folder "%s" [...]' % folder)

    parent = DummyParent()
    
    FILES = get_files_with_extension(folder, extension='.nwb', recursive=True)
    DATES = np.array([f.split(os.path.sep)[-1].split('-')[0] for f in FILES])
    SUBJECTS = []
    
    for f in FILES:
        try:
            read(parent, f, metadata_only=True)
            SUBJECTS.append(parent.metadata['subject_ID'])
        except BaseException as be:
            SUBJECTS.append('N/A')
            if verbose:
                print(be)
                print('\n /!\ Pb with "%s" \n' % f)
        parent.io.close()
        
    if verbose:
        print(' -> found n=%i datafiles ' % len(FILES))

    return np.array(FILES), np.array(DATES), np.array(SUBJECTS)


if __name__=='__main__':

    # FILES, DATES, SUBJECTS = scan_folder_for_NWBfiles('/home/yann/DATA/')
    # for f, d, s in zip(FILES, DATES, SUBJECTS):
    #     print(f, d, s)
    # print(np.unique(SUBJECTS))

    data = DummyParent()
    read(data, sys.argv[-1])
    # print(data.nwbfile.processing['ophys'])
    print(data.iscell)

    








