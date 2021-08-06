import pynwb, time, ast, sys, pathlib, os
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import get_files_with_extension
from visual_stim.psychopy_code.stimuli import build_stim


class Data:
    
    """
    a basic class to read NWB
    thought to be the parent for specific applications
    """
    
    def __init__(self, filename,
                 with_tlim=True,
                 metadata_only=False,
                 with_visual_stim=False,
                 verbose=False):

        self.io = pynwb.NWBHDF5IO(filename, 'r')
        self.nwbfile = self.io.read()

        if verbose:
            t0 = time.time()

        self.read_metadata()
        
        if with_tlim:
            self.read_tlim()
        else:
            self.tlim=None
            
        if not metadata_only:
            self.read_data()

        if with_visual_stim:
            self.init_visual_stim()
        else:
            self.visual_stim = None

        if verbose:
            print('NWB-file reading time: %.1fms' % (1e3*(time.time()-t0)))

        if metadata_only:
            self.close()
            

    def read_metadata(self):
        
        self.df_name = self.nwbfile.session_start_time.strftime("%Y/%m/%d -- %H:%M:%S")+' ---- '+\
            self.nwbfile.experiment_description
        
        self.metadata = ast.literal_eval(self.nwbfile.session_description)
        
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

        if 'time_start_realigned' in self.nwbfile.stimulus.keys():
            self.description += ' =>  completed N=%i/%i episodes  <=' %(self.nwbfile.stimulus['time_start_realigned'].data.shape[0],
                                                               self.nwbfile.stimulus['time_start'].data.shape[0])
                
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
                    
        
    def read_tlim(self):
        
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

    def read_data(self):

        # ophys data
        if 'ophys' in self.nwbfile.processing:
            self.read_and_format_ophys_data()
        else:
            for key in ['Segmentation', 'Fluorescence', 'iscell', 'validROI_indices', 'Neuropil', 'Deconvolved']:
                setattr(self, key, None)
                
        if 'Pupil' in self.nwbfile.processing:
            self.read_pupil()
            
        if 'FaceMotion' in self.nwbfile.processing:
            self.read_facemotion()
            

    def read_and_format_ophys_data(self):
        
        self.Segmentation = self.nwbfile.processing['ophys'].data_interfaces['ImageSegmentation'].plane_segmentations['PlaneSegmentation']
        self.pixel_masks_index = self.Segmentation.columns[0].data[:]
        self.pixel_masks = self.Segmentation.columns[1].data[:]

        self.Fluorescence = self.nwbfile.processing['ophys'].data_interfaces['Fluorescence'].roi_response_series['Fluorescence']
        self.Neuropil = self.nwbfile.processing['ophys'].data_interfaces['Neuropil'].roi_response_series['Neuropil']
        self.Deconvolved = self.nwbfile.processing['ophys'].data_interfaces['Deconvolved'].roi_response_series['Deconvolved']
        self.CaImaging_dt = (self.Neuropil.timestamps[1]-self.Neuropil.timestamps[0])

        if len(self.Segmentation.columns)>2 and (self.Segmentation.columns[2].name=='iscell'): # DEPRECATED
            self.iscell = self.Segmentation.columns[2].data[:,0].astype(bool)
            self.validROI_indices = np.arange(len(self.iscell))[self.iscell]
        else:
            self.iscell = np.ones(len(self.Fluorescence.data[:,0]), dtype=bool)
            self.validROI_indices = np.arange(len(self.Fluorescence.data[:,0]))

        # red-labelling
        if len(self.Segmentation.columns)>1 and (self.Segmentation.columns[2].name=='redcell'):
            self.redcell = self.Segmentation.columns[2].data[:,0].astype(bool)
        else:
            self.redcell = np.zeros(len(self.Fluorescence.data[:,0]), dtype=bool)
        
            
    def read_pupil(self):

        pd = str(self.nwbfile.processing['Pupil'].description)
        if len(pd.split('pix_to_mm='))>1:
            self.FaceCamera_mm_to_pix = int(1./float(pd.split('pix_to_mm=')[-1]))
        else:
            self.FaceCamera_mm_to_pix = 1
        #     self.t_pupil = self.nmbfile.processing['Pupil']
        #     self.nwbfile.processing['Pupil'].data_interfaces['cx']

        
    def read_facemotion(self):
        
        fd = str(self.nwbfile.processing['FaceMotion'].description)
        self.FaceMotion_ROI = [int(i) for i in fd.split('y0,dy)=(')[1].split(')')[0].split(',')]


    def close(self):
        self.io.close()
        
            
    def init_visual_stim(self):
        self.metadata['load_from_protocol_data'], self.metadata['no-window'] = True, True
        self.visual_stim = build_stim(self.metadata, no_psychopy=True)

        
    def get_protocol_id(protocol_name):
        # TO BE DONE
        return 0

    
    def get_protocol_cond(self, protocol_id):
        """
        ## a recording can have multiple protocols inside
        -> find the condition of a given protocol ID

        'None' to have them all 
        """

        if ('protocol_id' in self.nwbfile.stimulus) and (len(np.unique(self.nwbfile.stimulus['protocol_id'].data[:]))>1) and (protocol_id is not None):
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

    def list_subquantities(self, quantity):
        if quantity=='CaImaging':
            return ['dF/F', 'Fluorescence', 'Neuropil', 'Deconvolved', 'F-0.7*Fneu', 'F-Fneu']
        else:
            return ['']
        
        
def scan_folder_for_NWBfiles(folder, Nmax=1000000, verbose=True):

    if verbose:
        print('inspecting the folder "%s" [...]' % folder)
        t0 = time.time()

    FILES = get_files_with_extension(folder, extension='.nwb', recursive=True)
    DATES = np.array([f.split(os.path.sep)[-1].split('-')[0] for f in FILES])
    SUBJECTS = []
    
    for f in FILES[:Nmax]:
        try:
            data = Data(f, metadata_only=True)
            SUBJECTS.append(data.metadata['subject_ID'])
        except BaseException as be:
            SUBJECTS.append('N/A')
            if verbose:
                print(be)
                print('\n /!\ Pb with "%s" \n' % f)
        
    if verbose:
        print(' -> found n=%i datafiles (in %.1fs) ' % (len(FILES), (time.time()-t0)))

    return np.array(FILES), np.array(DATES), np.array(SUBJECTS)


if __name__=='__main__':

    FILES, DATES, SUBJECTS = scan_folder_for_NWBfiles('/home/yann/DATA/', Nmax=500)
    # for f, d, s in zip(FILES, DATES, SUBJECTS):
    #     print(f, d, s)
    # print(np.unique(SUBJECTS))

    # data = Data(sys.argv[-1])
    # # print(data.nwbfile.processing['ophys'])
    # print(data.iscell)

    








