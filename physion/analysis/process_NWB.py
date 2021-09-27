import sys, time, tempfile, os, pathlib, json, datetime, string, itertools
import numpy as np
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from Ca_imaging import tools as Ca_imaging_tools
from scipy.interpolate import interp1d
from analysis import stat_tools

class EpisodeResponse:

    def __init__(self, full_data,
                 protocol_id=0,
                 quantity='Photodiode-Signal',
                 subquantity='',
                 roiIndex=None, roiIndices=[0],
                 prestim_duration=None, # to force the prestim window otherwise, half the value in between episodes
                 dt_sampling=1, # ms
                 interpolation='linear',
                 baseline_substraction=False,
                 verbose=False):

        self.dt_sampling = dt_sampling,
        self.quantity = quantity
        if roiIndex is not None:
            self.roiIndices = [roiIndex]
        else:
            self.roiIndices = roiIndices
        resp = []

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

        # new sampling
        if (prestim_duration is None) and ('interstim' in full_data.nwbfile.stimulus):
            prestim_duration = np.min(full_data.nwbfile.stimulus['interstim'].data[:])/2. # half the stim duration
        if (prestim_duration is None) or (prestim_duration<1):
            prestim_duration = 1 # still 1s is a minimum
        ipre = int(prestim_duration/dt_sampling*1e3)

        duration = full_data.nwbfile.stimulus['time_stop'].data[Pcond][0]-full_data.nwbfile.stimulus['time_start'].data[Pcond][0]
        idur = int(duration/dt_sampling/1e-3)
        # -> time array:
        self.t = np.arange(-ipre+1, idur+ipre-1)*dt_sampling*1e-3

        if quantity=='CaImaging':
            tfull = full_data.Neuropil.timestamps[:]
            valfull = Ca_imaging_tools.compute_CaImaging_trace(full_data, subquantity,
                                                               self.roiIndices).sum(axis=0) # valid ROI indices inside
        elif quantity=='Pupil':
            if not hasattr(full_data, 'pupil_diameter'):
                full_data.build_pupil_diameter()
            tfull, valfull = full_data.t_pupil, full_data.pupil_diameter
        else:
            if quantity in full_data.nwbfile.acquisition:
                tfull = np.arange(full_data.nwbfile.acquisition[quantity].data.shape[0])/full_data.nwbfile.acquisition[quantity].rate
                valfull = full_data.nwbfile.acquisition[quantity].data[:]
            elif quantity in full_data.nwbfile.processing:
                tfull = np.arange(full_data.nwbfile.processing[quantity].data.shape[0])/full_data.nwbfile.processing[quantity].rate
                valfull = full_data.nwbfile.processing[quantity].data[:]
            else:
                print(30*'-')
                print(quantity, 'not recognized')
                print(30*'-')

        # adding the parameters
        for key in full_data.nwbfile.stimulus.keys():
            setattr(self, key, [])

        for iEp in np.arange(full_data.nwbfile.stimulus['time_start'].num_samples)[Pcond]:
            tstart = full_data.nwbfile.stimulus['time_start_realigned'].data[iEp]
            tstop = full_data.nwbfile.stimulus['time_stop_realigned'].data[iEp]

            # compute time and interpolate
            cond = (tfull>=(tstart-2.*prestim_duration)) & (tfull<(tstop+1.5*prestim_duration)) # higher range of interpolation to avoid boundary problems
            func = interp1d(tfull[cond]-tstart, valfull[cond],
                            kind=interpolation)
            try:
                if baseline_substraction:
                    y = func(self.t)
                    resp.append(y-np.mean(y[self.t<0])) # we remove the level before stim
                else:
                    resp.append(func(self.t))
                for key in full_data.nwbfile.stimulus.keys():
                    getattr(self, key).append(full_data.nwbfile.stimulus[key].data[iEp])
            except BaseException as be:
                print('----')
                print(be)
                print(tfull[cond][0]-tstart, tfull[cond][-1]-tstart, tstop-tstart)
                print('Problem with episode %i between (%.2f, %.2f)s' % (iEp, tstart, tstop))

        self.resp = np.array(resp)
        self.index_from_start = np.arange(len(Pcond))[Pcond][:self.resp.shape[0]]
        
        for key in full_data.nwbfile.stimulus.keys():
            setattr(self, key, np.array(getattr(self, key)))

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
        # cell_resp = EpisodeResponse(data, roiIndex=0)
        pupil_eps = EpisodeResponse(data, quantity='Pupil')
        print(pupil_eps.t)
    else:
        print('/!\ Need to provide a NWB datafile as argument ')
            







