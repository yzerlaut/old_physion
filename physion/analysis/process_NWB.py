import sys, time, tempfile, os, pathlib, json, datetime, string
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
                 roiIndices=[0],
                 prestim_duration=None, # to force the prestim window otherwise, half the value in between episodes
                 dt_sampling=1, # ms
                 interpolation='linear',
                 baseline_substraction=False,
                 verbose=True):

        self.dt_sampling = dt_sampling,
        self.quantity = quantity
        resp = []

        # choosing protocol (if multiprotocol)
        Pcond = full_data.get_protocol_cond(protocol_id)

        if verbose:
            print('  Number of episodes over the whole recording: %i/%i (with protocol condition)' % (np.sum(Pcond), len(Pcond)))
            print('  building episodes [...]')

        # find the parameter(s) varied within that specific protocol
        self.varied_parameters =  {}
        for key in full_data.nwbfile.stimulus.keys():
            if key not in ['frame_run_type', 'index', 'protocol_id', 'time_duration', 'time_start',
                           'time_start_realigned', 'time_stop', 'time_stop_realigned']:
                unique = np.unique(full_data.nwbfile.stimulus[key].data[Pcond])
                if len(unique)>1:
                    self.varied_parameters[key] = unique

                
        # new sampling
        if (prestim_duration is None) and ('interstim' in full_data.nwbfile.stimulus):
            prestim_duration = np.min(full_data.nwbfile.stimulus['interstim'].data[:])/2. # half the stim duration
        elif prestim_duration is None:
            prestim_duration = 1
        ipre = int(prestim_duration/dt_sampling*1e3)

        duration = full_data.nwbfile.stimulus['time_stop'].data[Pcond][0]-full_data.nwbfile.stimulus['time_start'].data[Pcond][0]
        idur = int(duration/dt_sampling/1e-3)
        # -> time array:
        self.t = np.arange(-ipre+1, idur+ipre-1)*dt_sampling*1e-3

        if quantity=='CaImaging':
            tfull = full_data.Neuropil.timestamps[:]
            valfull = Ca_imaging_tools.compute_CaImaging_trace(full_data, subquantity,
                                                               roiIndices).sum(axis=0) # valid ROI indices inside
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
            cond = (tfull>=(tstart-1.5*prestim_duration)) & (tfull<(tstop+1.5*prestim_duration)) # higher range of interpolation to avoid boundary problems
            func = interp1d(tfull[cond]-tstart, valfull[cond],
                            kind=interpolation)
            try:
                if baseline_substraction:
                    y = func(self.t)
                    resp.append(y-np.mean(y[self.t<0]))
                else:
                    resp.append(func(self.t))
                for key in full_data.nwbfile.stimulus.keys():
                    getattr(self, key).append(full_data.nwbfile.stimulus[key].data[iEp])
            except BaseException as be:
                print('----')
                print(be)
                print('Problem with episode %i between (%.2f, %.2f)s' % (iEp, tstart, tstop))

        self.index_from_start = np.arange(len(Pcond))[Pcond]
        self.resp = np.array(resp)
        
        for key in full_data.nwbfile.stimulus.keys():
            setattr(self, key, np.array(getattr(self, key)))

        if verbose:
            print('  -> [ok] episodes ready !')



if __name__=='__main__':

    from analysis.read_NWB import Data
    
    filename = sys.argv[-1]
    
    if '.nwb' in sys.argv[-1]:
        data = Data(filename)
        cell_resp = EpisodeResponse(data)
        print(cell_resp.t)
    else:
        print('/!\ Need to provide a NWB datafile as argument ')
            







