import sys, time, tempfile, os, pathlib, json, datetime, string, itertools
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from Ca_imaging import tools as Ca_imaging_tools
from scipy.interpolate import interp1d
from analysis import stat_tools

class EpisodeResponse:
    """
    - Using the photodiode-signal-derived timestamps to build "episode response" by interpolating the raw signal on a fixed time interval (surrounding the stim)
    - Using metadata to store stimulus informations per episode
    """

    def __init__(self, full_data,
                 protocol_id=0,
                 quantities=['Photodiode-Signal'],
                 quantities_args=None,
                 prestim_duration=None, # to force the prestim window otherwise, half the value in between episodes
                 dt_sampling=1, # ms
                 interpolation='linear',
                 tfull=None,
                 verbose=True):

        self.dt_sampling = dt_sampling
        
        # choosing protocol (if multiprotocol)
        self.protocol_cond_in_full_data = full_data.get_protocol_cond(protocol_id)
        
        if quantities_args is None:
            quantities_args = [{} for q in quantities]
        quantities_args['verbose'] = verbose

        if verbose:
            print('  Number of episodes over the whole recording: %i/%i (with protocol condition)' % (np.sum(self.protocol_cond_in_full_data), len(self.protocol_cond_in_full_data)))
            print('  building episodes with %i modalities [...]' % len(quantities))

        # find the parameter(s) varied within that specific protocol
        self.varied_parameters, self.fixed_parameters =  {}, {}
        for key in full_data.nwbfile.stimulus.keys():
            if key not in ['frame_run_type', 'index', 'protocol_id', 'time_duration', 'time_start',
                           'time_start_realigned', 'time_stop', 'time_stop_realigned', 'interstim']:
                unique = np.unique(full_data.nwbfile.stimulus[key].data[self.protocol_cond_in_full_data])
                if len(unique)>1:
                    self.varied_parameters[key] = unique
                elif len(unique)==1:
                    self.fixed_parameters[key] = unique

        # new sampling, a window arround stimulus presentation
        if (prestim_duration is None) and ('interstim' in full_data.nwbfile.stimulus):
            prestim_duration = np.min(full_data.nwbfile.stimulus['interstim'].data[:])/2. # half the stim duration
        if (prestim_duration is None) or (prestim_duration<1):
            prestim_duration = 1 # still 1s is a minimum
        ipre = int(prestim_duration/dt_sampling*1e3)

        duration = full_data.nwbfile.stimulus['time_stop'].data[self.protocol_cond_in_full_data][0]-full_data.nwbfile.stimulus['time_start'].data[self.protocol_cond_in_full_data][0]
        idur = int(duration/dt_sampling/1e-3)
        # -> time array:
        self.t = np.arange(-ipre+1, idur+ipre-1)*dt_sampling*1e-3


        #############################################################################
        ############ we do it modality by modality (quantity)  ######################
        #############################################################################
        
        QUANTITIES, QUANTITY_VALUES, QUANTITY_TIMES = [], [], []
        
        for iq, quantity, quantity_args in zip(range(len(quantities)), quantities, quantities_args):

            if type(quantity)!=str and (tfull is not None):
                QUANTITY_VALUES.append(quantity)
                QUANTITY_TIMES.append(tfull)
                QUANTITIES.append('quant_%i' % iq)
                
            elif quantity in ['dFoF', 'dF/F']:
                if not hasattr(full_data, 'dFoF'):
                    full_data.build_dFoF(**quantity_args)
                QUANTITY_VALUES.append(full_data.dFoF)
                QUANTITY_TIMES.append(full_data.t_dFoF)
                QUANTITIES.append('dFoF')

            elif quantity in ['Neuropil', 'neuropil']:
                if not hasattr(full_data, 'Neuropil'):
                    full_data.build_Neuropil(**quantity_args)
                QUANTITY_VALUES.append(full_data.Neuropil)
                QUANTITY_TIMES.append(full_data.t_Neuropil)
                QUANTITIES.append('neuropil')

            elif quantity in ['Fluorescence', 'rawFluo']:
                if not hasattr(full_data, 'rawFluo'):
                    full_data.build_rawFluo(**quantity_args)
                QUANTITY_VALUES.append(full_data.rawFluo)
                QUANTITY_TIMES.append(full_data.t_rawFluo)
                QUANTITIES.append('rawFluo')
                
            elif quantity in ['Pupil', 'pupil-size', 'Pupil-diameter', 'pupil-diameter', 'pupil']:
                if not hasattr(full_data, 'pupil_diameter'):
                    full_data.build_pupil_diameter(**quantity_args)
                QUANTITY_VALUES.append(full_data.pupil_diameter)
                QUANTITY_TIMES.append(full_data.t_pupil)
                QUANTITIES.append('pupilSize')

            elif quantity in ['gaze', 'Gaze', 'gaze_movement', 'gazeMovement', 'gazeDirection']:
                if not hasattr(full_data, 'gaze_movement'):
                    full_data.build_gaze_movement(**quantity_args)
                QUANTITY_VALUES.append(full_data.gaze_movement)
                QUANTITY_TIMES.append(full_data.t_pupil)
                QUANTITIES.append('gazeDirection')
                
            elif quantity in ['facemotion', 'FaceMotion', 'faceMotion']:
                if not hasattr(full_data, 'facemotion'):
                    full_data.build_facemotion(**quantity_args)
                QUANTITY_VALUES.append(full_data.facemotion)
                QUANTITY_TIMES.append(full_data.t_facemotion)
                QUANTITIES.append('faceMotion')
                
            else:
                if quantity in full_data.nwbfile.acquisition:
                    QUANTITY_TIMES.append(np.arange(full_data.nwbfile.acquisition[quantity].data.shape[0])/full_data.nwbfile.acquisition[quantity].rate)
                    QUANTITY_VALUES.append(full_data.nwbfile.acquisition[quantity].data[:])
                    QUANTITIES.append(full_data.nwbfile.acquisition[quantity].name.replace('-', '').replace('_', ''))
                elif quantity in full_data.nwbfile.processing:
                    QUANTITY_TIMES.append(np.arange(full_data.nwbfile.processing[quantity].data.shape[0])/full_data.nwbfile.processing[quantity].rate)
                    QUANTITY_VALUES.append(full_data.nwbfile.processing[quantity].data[:])
                    QUANTITIES.append(full_data.nwbfile.processing[quantity].name.replace('-', '').replace('_', ''))
                else:
                    print(30*'-')
                    print(quantity, 'not recognized')
                    print(30*'-')
                    
        # adding the parameters
        for key in full_data.nwbfile.stimulus.keys():
            setattr(self, key, [])

        for q in QUANTITIES:
            setattr(self, q, [])

        for iEp in np.arange(full_data.nwbfile.stimulus['time_start'].num_samples)[self.protocol_cond_in_full_data]:
            
            tstart = full_data.nwbfile.stimulus['time_start_realigned'].data[iEp]
            tstop = full_data.nwbfile.stimulus['time_stop_realigned'].data[iEp]

            RESPS, success = [], True
            for quantity, tfull, valfull in zip(QUANTITIES, QUANTITY_TIMES, QUANTITY_VALUES):
                
                # compute time and interpolate
                ep_cond = (tfull>=(tstart-2.*prestim_duration)) & (tfull<(tstop+1.5*prestim_duration)) # higher range of interpolation to avoid boundary problems
                try:
                    if len(valfull.shape)>1:
                        # multi-dimensional response, e.g. dFoF = (rois, time)
                        resp = np.zeros((valfull.shape[0], len(self.t)))
                        for j in range(valfull.shape[0]):
                            func = interp1d(tfull[ep_cond]-tstart, valfull[j,ep_cond],
                                            kind=interpolation)
                            resp[j, :] = func(self.t)
                        RESPS.append(resp)
                        
                    else:
                        func = interp1d(tfull[ep_cond]-tstart, valfull[ep_cond],
                                        kind=interpolation)
                        RESPS.append(func(self.t))
                        
                except BaseException as be:
                    success=False # we switch this off to remove the episode in all modalities
                    if verbose:
                        print('----')
                        print(be)
                        print(tfull[ep_cond][0]-tstart, tfull[ep_cond][-1]-tstart, tstop-tstart)
                        
                        print('Problem with episode %i between (%.2f, %.2f)s' % (iEp, tstart, tstop))
                        
            if success:
                # only succesful episodes in all modalities
                for quantity, response in zip(QUANTITIES, RESPS):
                    getattr(self, quantity).append(response)
                for key in full_data.nwbfile.stimulus.keys():
                    getattr(self, key).append(full_data.nwbfile.stimulus[key].data[iEp])

        # transform stim params to np.array
        for key in full_data.nwbfile.stimulus.keys():
            setattr(self, key, np.array(getattr(self, key)))
        for q in QUANTITIES:
            setattr(self, q, np.array(getattr(self, q)))

        self.index_from_start = np.arange(len(self.protocol_cond_in_full_data))[self.protocol_cond_in_full_data][:getattr(self, QUANTITIES[0]).shape[0]]
        self.quantities = QUANTITIES
        self.protocol_id = protocol_id
        
        if verbose:
            print('  -> [ok] episodes ready !')


    def get_response(self, quantity=None, roiIndex=None, roiIndices='all'):
        """
        to deal with the fact that single-episode responses can be multidimensional
        """
        if quantity is None:
            if len(self.quantities)>1:
                print('\n there are several modalities in that episode')
                print('     -> need to define the desired quantity, here taking: "%s"' % self.quantities[0])
            quantity = self.quantities[0]

        if len(getattr(self, quantity).shape)>2:
            if roiIndex is not None:
                roiIndices = roiIndex
            elif roiIndices in ['all', 'sum', 'mean']:
                roiIndices = np.arange(getattr(self, quantity).shape[1])
            response = getattr(self, quantity)[:,roiIndices,:]
            if len(response.shape)>2:
                response = response.mean(axis=1)
            return response
        else:
            return getattr(self, quantity)

        
    def compute_interval_cond(self, interval):
        return (self.t>=interval[0]) & (self.t<=interval[1])

    
    def find_episode_cond(self, key, index):
        if (type(key) in [list, np.ndarray]) and (type(index) in [list, np.ndarray, tuple]) :
            cond = (getattr(self, key[0])==self.varied_parameters[key[0]][index[0]])
            for n in range(1, len(key)):
                cond = cond & (getattr(self, key[n])==self.varied_parameters[key[n]][index[n]])
        else:
            cond = (getattr(self, key)==self.varied_parameters[key][index])
        return cond

    
    def stat_test_for_evoked_responses(self,
                                       episode_cond=None, response_args={},
                                       interval_pre=[-2,0], interval_post=[1,3],
                                       test='wilcoxon',
                                       positive=True):

        response = self.get_response(**response_args)
        
        if episode_cond is None:
            episode_cond = np.ones(response.shape[0], dtype=bool)

        pre_cond  = self.compute_interval_cond(interval_pre)
        post_cond  = self.compute_interval_cond(interval_post)

        return stat_tools.StatTest(response[episode_cond,:][:,pre_cond].mean(axis=1),
                                   response[episode_cond,:][:,post_cond].mean(axis=1),
                                   test=test, positive=positive)

    def compute_stats_over_repeated_trials(self, key, index,
                                           response_args={},
                                           interval_cond=None,
                                           quantity='mean'):
        
        cond = self.find_episode_cond(key, index)
        response = self.get_response(**response_args)
        
        if interval_cond is None:
            interval_cond = np.ones(len(self.t), dtype=bool)

        quantities = []
        for i in np.arange(response.shape[0])[cond]:
            if quantity=='mean':
                quantities.append(np.mean(response[i, interval_cond]))
            elif quantity=='integral':
                quantities.append(np.trapz(response[i, interval_cond]))
        return np.array(quantities)


    def compute_summary_data(self, stat_test_props,
                             exclude_keys=['repeat'],
                             response_args={},
                             response_significance_threshold=0.01):

        VARIED_KEYS, VARIED_VALUES, VARIED_INDICES, Nfigs, VARIED_BINS = [], [], [], 1, []
        for key in self.varied_parameters:
            if key not in exclude_keys:
                VARIED_KEYS.append(key)
                VARIED_VALUES.append(self.varied_parameters[key])
                VARIED_INDICES.append(np.arange(len(self.varied_parameters[key])))
                x = np.unique(self.varied_parameters[key])
                VARIED_BINS.append(np.concatenate([[x[0]-.5*(x[1]-x[0])],
                                                   .5*(x[1:]+x[:-1]),
                                                   [x[-1]+.5*(x[-1]-x[-2])]]))

        summary_data = {'value':[], 'significant':[]}
        for key, bins in zip(VARIED_KEYS, VARIED_BINS):
            summary_data[key] = []
            summary_data[key+'-bins'] = bins

        if len(VARIED_KEYS)>0:
            for indices in itertools.product(*VARIED_INDICES):
                stats = self.stat_test_for_evoked_responses(episode_cond=self.find_episode_cond(VARIED_KEYS,
                                                                                                list(indices)),
                                                            response_args=response_args,
                                                            **stat_test_props)

                for key, index in zip(VARIED_KEYS, indices):
                    summary_data[key].append(self.varied_parameters[key][index])
                summary_data['value'].append(np.mean(stats.y-stats.x))
                summary_data['significant'].append(stats.significant(threshold=response_significance_threshold))
        else:
            stats = self.stat_test_for_evoked_responses(response_args=response_args,
                                                        **stat_test_props)
            summary_data['value'].append(np.mean(stats.y-stats.x))
            summary_data['significant'].append(stats.significant(threshold=response_significance_threshold))

        for key in summary_data:
            summary_data[key] = np.array(summary_data[key])
    
        return summary_data
    
if __name__=='__main__':

    from analysis.read_NWB import Data
    
    filename = sys.argv[-1]
    
    if '.nwb' in sys.argv[-1]:
        data = Data(filename)
        data.build_dFoF()

        episode = EpisodeResponse(data,
                                  quantities=['Photodiode-Signal', 'pupil', 'gaze', 'facemotion', 'dFoF', 'rawFluo', 'Running-Speed'])
        print(episode.quantities)
        # from datavyz import ge
        # ge.plot(episode.t, episode.PhotodiodeSignal.mean(axis=0), sy=episode.PhotodiodeSignal.std(axis=0))
        # ge.show()
        # episode = EpisodeResponse(data,
        #                           quantities=['Pupil', 'CaImaging', 'CaImaging'],
        #                           quantities_args=[{}, {'subquantity':'Fluorescence'}, {'subquantity':'dFoF', 'roiIndices':np.arange(10)}])
        # print(episode.CaImaging_dFoF.shape)

        # from datavyz import ge
        # episode = EpisodeResponse(data,
        #                           quantities=['dFoF'])
        # summary_data = episode.compute_summary_data(dict(interval_pre=[-1,0], interval_post=[1,2], test='wilcoxon', positive=True),
        #                                             response_args={'quantity':'dFoF', 'roiIndex':2})

        # print(summary_data)

        
    else:
        print('/!\ Need to provide a NWB datafile as argument ')
            







