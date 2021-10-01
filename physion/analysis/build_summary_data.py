import pynwb, time, ast, sys, pathlib, os, datetime
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from analysis.read_NWB import Data, scan_folder_for_NWBfiles
from analysis.process_NWB import EpisodeResponse


def find_protocol_details(FILES):
    """
    
    """
    
    # protocol based on the first one:
    data0 = Data(FILES[0])
    protocols = list(data0.protocols)
    
    STIM = {}
    for key in data0.nwbfile.stimulus.keys():
        STIM[key] = []

    subjects = ['' for i in range(len(FILES))]
    sessions, sessions_per_subject = np.arange(len(FILES)), np.zeros(len(FILES), dtype=int)
    for i, f in enumerate(FILES):
        data = Data(f)
        if not list(data.protocols)==protocols:
            print('/!\ not a set of consistent files /!\ ')
            break;
        subjects[i] = data.metadata['subject_ID']
        sessions_per_subject[i] = np.sum(data.metadata['subject_ID']==np.array(subjects))
        data.close()

    filename = '%s-%isubjects-%isessions.nwb' % (data0.metadata['protocol'], len(np.unique(subjects)), len(FILES))
    data0.close()
    
    return protocols, subjects, sessions, sessions_per_subject, filename, STIM


def build_summary_episodes(FILES,
                           roi_prefix=10000, # to have unique roi per session, session X has roi IDs: X*roi_prefix+i_ROI
                           prestim_duration=2,
                           modalities=['pupil', 'facemotion', 'running-speed'],
                           dt_sampling=20, # ms
                           Nmax=100000):
    """
    
    """

    protocols, subjects, sessions, sessions_per_subject, filename, STIM = find_protocol_details(FILES)
    FULL_EPISODE_ARRAY, QUANT = [], {'subject':[],
                                     'session_per_subject':[],
                                     'session':[], 'roi':[],
                                     'pupil':[], 'running-speed':[], 'facemotion':[]}
    print('- building "%s" by concatenating episodes from n=%i files [...]' % (filename, len(FILES)))
    
    for session, f in enumerate(FILES):

        print('   -> session #%i: %s' % (session+1, f))
        data = Data(f)

        for ip, p in enumerate(protocols):


            if len(protocols)>1:
                duration = data.metadata['Protocol-%i-presentation-duration' % (ip+1)]
            else:
                duration = data.metadata['presentation-duration']
                
            # build episodes of other modalities (running, ...)
            if ('Pupil' in data.nwbfile.processing) and ('pupil' in modalities):
                Pupil_episodes = EpisodeResponse(data,
                                                 protocol_id=ip,
                                                 prestim_duration=prestim_duration,
                                                 dt_sampling=dt_sampling, # ms
                                                 quantity='Pupil')
                t_pupil_cond = (Pupil_episodes.t>0) & (Pupil_episodes.t<duration)
            else:
                Pupil_episodes = None

            if ('Running-Speed' in data.nwbfile.acquisition) and ('running-speed' in modalities):
                Running_episodes = EpisodeResponse(data,
                                                   protocol_id=ip,
                                                   prestim_duration=prestim_duration,
                                                   dt_sampling=dt_sampling, # ms
                                                   quantity='Running-Speed')
                t_running_cond = (Running_episodes.t>0) & (Running_episodes.t<duration)
            else:
                Running_episodes = None

            if ('FaceMotion' in data.nwbfile.processing) and ('facemotion' in modalities):
                FaceMotion_episodes = EpisodeResponse(data,
                                                      protocol_id=ip,
                                                      prestim_duration=prestim_duration,
                                                      dt_sampling=dt_sampling, # ms
                                                      quantity='FaceMotion')
                t_facemotion_cond = (FaceMotion_episodes.t>0) & (FaceMotion_episodes.t<duration)
            else:
                FaceMotion_episodes = None
                

            for roi in range(np.sum(data.iscell))[:Nmax]:

                roiID = roi_prefix*session+roi
                
                EPISODES = EpisodeResponse(data,
                                           protocol_id=ip,
                                           quantity='CaImaging',
                                           subquantity='dF/F',
                                           dt_sampling=dt_sampling, # ms
                                           roiIndex=roi,
                                           prestim_duration=prestim_duration)
                
                for iEp in range(EPISODES.resp.shape[0]):
                    FULL_EPISODE_ARRAY.append(EPISODES.resp[iEp,:])
                    for key in data.nwbfile.stimulus.keys():
                        STIM[key].append(data.nwbfile.stimulus[key].data[iEp])
                    QUANT['roi'].append(roiID)
                    QUANT['session'].append(session)
                    QUANT['subject'].append(data.metadata['subject_ID'])
                    QUANT['session_per_subject'].append(sessions_per_subject[session])
                    
                    if Running_episodes is not None:
                        QUANT['running-speed'].append(Running_episodes.resp[iEp,:][t_running_cond])
                    else:
                        QUANT['running-speed'].append(666.) # flag for None

                    if Pupil_episodes is not None:
                        QUANT['pupil'].append(Pupil_episodes.resp[iEp,:][t_pupil_cond])
                    else:
                        QUANT['pupil'].append(666.) # flag for None

                    if FaceMotion_episodes is not None:
                        QUANT['facemotion'].append(FaceMotion_episodes.resp[iEp,:][t_facemotion_cond])
                    else:
                        QUANT['facemotion'].append(666.) # flag for None
                        
    # set up the NWBFile
    description = 'Summary data concatenating episodes from the datafiles:\n'
    for f in FILES:
        description += '- %s\n' % f
    nwbfile = pynwb.NWBFile(session_description=description,
                            identifier=filename,
                            session_start_time=datetime.datetime.now().astimezone())

    
    episode_waveforms = pynwb.TimeSeries(name='episode_waveforms',
                                       data=np.array(FULL_EPISODE_ARRAY),
                                       unit='dF/F',
                                       timestamps=EPISODES.t)

    nwbfile.add_acquisition(episode_waveforms)

    for key in STIM:
        stim = pynwb.TimeSeries(name=key,
                                data=np.array(STIM[key]),
                                unit='None', rate=1.)
        nwbfile.add_stimulus(stim)

    for key in QUANT:
        stim = pynwb.TimeSeries(name=key,
                                data=np.array(QUANT[key]),
                                unit='None', rate=1.)
        nwbfile.add_acquisition(stim)
        
    io=pynwb.NWBHDF5IO(filename, mode='w')
    io.write(nwbfile)
    io.close()


def read_file(filename):
    
    io = pynwb.NWBHDF5IO(filename, mode='r')
    nwbfile = io.read()

    from datavyz import ge
    #
    cond = (nwbfile.stimulus['protocol_id'].data[:]==2)
    ge.plot(nwbfile.acquisition['episode_waveforms'].timestamps[:], nwbfile.acquisition['episode_waveforms'].data[cond,:].mean(axis=0))
    ge.show()

    cond = (nwbfile.acquisition['running-speed'].data[:]!=666)
    ge.hist(nwbfile.acquisition['running-speed'].data[:][cond])
    ge.show()
    
    io.close()
    

if __name__=='__main__':
    
    import argparse

    parser=argparse.ArgumentParser()
    parser.add_argument("datafiles", type=str, nargs='*', help='can be a text file, or a folder, or a list of datafiles')
    parser.add_argument('-m', "--modalities", type=str, nargs='*', default=['pupil', 'facemotion', 'running-speed'])
    parser.add_argument("--iprotocol", type=int, default=0, help='index for the protocol in case of multiprotocol in datafile')
    parser.add_argument('-nmax', "--Nmax", type=int, default=1000000)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    FILES = []
    if (len(args.datafiles)==1) and os.path.isfile(args.datafiles[0]) and args.datafiles[0].endswith('.txt'):
        with open(args.datafiles[0], 'r') as f:
            FILES = [x for x in f.read().split('\n') if x.ensdwith('.nwb')]
    elif (len(args.datafiles)==1) and os.path.isdir(args.datafiles[0]):
        FILES, _, _ = scan_folder_for_NWBfiles(args.datafiles[0], Nmax=args.Nmax)
    else:
        FILES = args.datafiles

    print('list of NWB files:\n', FILES)
    print(' - concatenating [...]')
    build_summary_episodes(FILES, modalities=args.modalities)



