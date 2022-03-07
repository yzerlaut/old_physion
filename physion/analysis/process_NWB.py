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
                 quantities_args=[{}],
                 prestim_duration=None, # to force the prestim window otherwise, half the value in between episodes
                 dt_sampling=1, # ms
                 interpolation='linear',
                 tfull=None,
                 verbose=True):

        self.dt_sampling = dt_sampling
        
        # choosing protocol (if multiprotocol)
        Pcond = full_data.get_protocol_cond(protocol_id)

        if verbose:
            print('  Number of episodes over the whole recording: %i/%i (with protocol condition)' % (np.sum(Pcond), len(Pcond)))
            print('  building episodes [...]')

        # find the parameter(s) varied within that specific protocol
        self.varied_parameters, self.fixed_parameters =  {}, {}
        for key in full_data.nwbfile.stimulus.keys():
            if key not in ['frame_run_type', 'index', 'protocol_id', 'time_duration', 'time_start',
                           'time_start_realigned', 'time_stop', 'time_stop_realigned', 'interstim']:
                unique = np.unique(full_data.nwbfile.stimulus[key].data[Pcond])
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

        duration = full_data.nwbfile.stimulus['time_stop'].data[Pcond][0]-full_data.nwbfile.stimulus['time_start'].data[Pcond][0]
        idur = int(duration/dt_sampling/1e-3)
        # -> time array:
        self.t = np.arange(-ipre+1, idur+ipre-1)*dt_sampling*1e-3
        
        QUANTITIES, QUANTITY_VALUES, QUANTITY_TIMES = [], [], []
        
        for iq, quantity, quantity_args in zip(range(len(quantities)), quantities, quantities_args):
            
            if type(quantity)!=str and (tfull is not None):
                QUANTITY_VALUES.append(quantity)
                QUANTITY_TIMES.append(tfull)
                QUANTITIES.append('quant_%i' % iq)
                
            elif (quantity=='CaImaging') and (quantity_args['subquantity'] in ['dFoF', 'dF/F']):
                if not hasattr(full_data, 'dFoF'):
                    full_data.build_dFoF(**quantity_args)
                QUANTITY_VALUES.append(full_data.dFoF)
                QUANTITY_TIMES.append(full_data.t_dFoF)
                QUANTITIES.append('CaImaging_%s' % quantity_args['subquantity'])

            elif quantity=='CaImaging':
                QUANTITY_VALUES.append(Ca_imaging_tools.compute_CaImaging_trace(full_data, **quantity_args))
                QUANTITY_TIMES.append(full_data.Neuropil.timestamps[:])
                QUANTITIES.append('CaImaging_%s' % quantity_args['subquantity'])
                
            elif quantity in ['Pupil', 'pupil-size', 'Pupil-diameter', 'pupil-diameter', 'pupil']:
                if not hasattr(full_data, 'pupil_diameter'):
                    full_data.build_pupil_diameter(**quantity_args)
                QUANTITY_VALUES.append(full_data.pupil_diameter)
                QUANTITY_TIMES.append(full_data.t_pupil)
                QUANTITIES.append('pupil')

            elif quantity in ['facemotion', 'FaceMotion']:
                if not hasattr(full_data, 'facemotion'):
                    full_data.build_facemotion(**quantity_args)
                QUANTITY_VALUES.append(full_data.facemotion)
                QUANTITY_TIMES.append(full_data.t_facemotion)
                QUANTITIES.append('facemotion')
                
            else:
                if quantity in full_data.nwbfile.acquisition:
                    QUANTITY_TIMES.append(np.arange(full_data.nwbfile.acquisition[quantity].data.shape[0])/full_data.nwbfile.acquisition[quantity].rate)
                    QUANTITY_VALUES.append(full_data.nwbfile.acquisition[quantity].data[:])
                    QUANTITIES.append('quant_%i' % iq)
                elif quantity in full_data.nwbfile.processing:
                    QUANTITY_TIMES.append(np.arange(full_data.nwbfile.processing[quantity].data.shape[0])/full_data.nwbfile.processing[quantity].rate)
                    QUANTITY_VALUES.append(full_data.nwbfile.processing[quantity].data[:])
                    QUANTITIES.append('quant_%i' % iq)
                else:
                    print(30*'-')
                    print(quantity, 'not recognized')
                    print(30*'-')

        # adding the parameters
        for key in full_data.nwbfile.stimulus.keys():
            setattr(self, key, [])

        for q in QUANTITIES:
            setattr(self, q, [])

            
        for iEp in np.arange(full_data.nwbfile.stimulus['time_start'].num_samples)[Pcond]:
            
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

        self.index_from_start = np.arange(len(Pcond))[Pcond][:getattr(self, QUANTITIES[0]).shape[0]]
        
        if verbose:
            print('  -> [ok] episodes ready !')


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
                                       episode_cond=None,
                                       interval_pre=[-2,0],
                                       interval_post=[1,3],
                                       test='wilcoxon',
                                       positive=True):

        if episode_cond is None:
            episode_cond = np.ones(self.resp.shape[0], dtype=bool)

        pre_cond  = self.compute_interval_cond(interval_pre)
        post_cond  = self.compute_interval_cond(interval_post)

        return stat_tools.StatTest(self.resp[episode_cond,:][:,pre_cond].mean(axis=1),
                                   self.resp[episode_cond,:][:,post_cond].mean(axis=1),
                                   test=test, positive=positive)

    def compute_stats_over_repeated_trials(self, key, index,
                                           interval_cond=None,
                                           quantity='mean'):
        
        cond = self.find_episode_cond(key, index)
        
        if interval_cond is None:
            interval_cond = np.ones(len(self.t), dtype=bool)

        quantities = []
        for i in np.arange(self.resp.shape[0])[cond]:
            if quantity=='mean':
                quantities.append(np.mean(self.resp[i, interval_cond]))
            elif quantity=='integral':
                quantities.append(np.trapz(self.resp[i, interval_cond]))
        return np.array(quantities)


    def compute_summary_data(self, stat_test_props,
                             exclude_keys=['repeat'],
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

        for indices in itertools.product(*VARIED_INDICES):
            stats = self.stat_test_for_evoked_responses(episode_cond=self.find_episode_cond(VARIED_KEYS,
                                                                                            list(indices)),
                                                        **stat_test_props)
            
            for key, index in zip(VARIED_KEYS, indices):
                summary_data[key].append(self.varied_parameters[key][index])
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
        # pupil_eps = EpisodeResponse(data, quantity='Pupil')
        episode = EpisodeResponse(data,
                                  quantities=['Pupil', 'CaImaging'],
                                  quantities_args=[{}, {'subquantity':'dF/F', 'roiIndices':np.arange(10)}])
        print(episode)
    else:
        print('/!\ Need to provide a NWB datafile as argument ')
            







