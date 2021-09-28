import pynwb, time, ast, sys, pathlib, os, datetime
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from analysis.read_NWB import Data, scan_folder_for_NWBfiles
from analysis.process_NWB import EpisodeResponse


def find_protocol_details(FILES):

    # protocol based on the first one:
    data = Data(FILES[0])
    protocols = list(data.protocols)
    data.io.close()
    STIM = {'subject':[],
            'session_per_subject':[],
            'session':[],
            'roi':[]}
    for key in data.nwbfile.stimulus.keys():
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
        data.io.close()

    return protocols, subjects, sessions, sessions_per_subject, STIM

def build_summary_episodes(FILES,
                           roi_prefix=10000,
                           Nmax=100000):

    protocols, subjects, sessions, sessions_per_subject, STIM = find_protocol_details(FILES)
    FULL_EPISODE_ARRAY = []
    
    for session, f in enumerate(FILES):
        data = Data(f)

        # build episodes of other modalities (running, ...)
        # [...]
        
        for ip, p in enumerate(protocols):

            for roi in range(np.sum(data.iscell))[:Nmax]:

                roiID = roi_prefix*session+roi
                
                EPISODES = EpisodeResponse(data,
                                           protocol_id=ip,
                                           quantity='CaImaging',
                                           subquantity='dF/F',
                                           dt_sampling=20, # ms
                                           roiIndex=roi)
                for iEp in range(EPISODES.resp.shape[0]):
                    FULL_EPISODE_ARRAY.append(EPISODES.resp[iEp,:])
                    for key in data.nwbfile.stimulus.keys():
                        STIM[key].append(data.nwbfile.stimulus[key].data[iEp])
                    STIM['roi'].append(roiID)
                    STIM['session'].append(session)
                    STIM['subject'].append(data.metadata['subject_ID'])
                    STIM['session_per_subject'].append(sessions_per_subject[session])

    # set up the NWBFile
    nwbfile = pynwb.NWBFile(session_description='demonstrate NWB object IDs',
                            identifier='NWB456',
                            session_start_time=datetime.datetime.now().astimezone())

    episode_summary = pynwb.TimeSeries(name='episode_summary',
                                       data=np.array(FULL_EPISODE_ARRAY),
                                       unit='dF/F',
                                       timestamps=EPISODES.t)

    nwbfile.add_acquisition(episode_summary)

    for key in STIM:
        stim = pynwb.TimeSeries(name=key,
                                data=np.array(STIM[key]),
                                unit='None', rate=1.)
        nwbfile.add_stimulus(stim)

    io=pynwb.NWBHDF5IO('example.nwb', mode='w')
    io.write(nwbfile)
    io.close()


def read_file():
    
    # set up the NWBFile
    nwbfile = pynwb.NWBFile(session_description='demonstrate NWB object IDs',
                            identifier='NWB456',
                            session_start_time=datetime.datetime.now().astimezone())

    episode_summary = pynwb.TimeSeries(name='episode_summary',
                                       data=np.random.randn(1000,300000),
                                       unit='dF/F',
                                       timestamps=np.linspace(0, 1, 1000))

    nwbfile.add_acquisition(episode_summary)

    io=pynwb.NWBHDF5IO('example.nwb', mode='w')
    io.write(nwbfile)
    io.close()
    
    
FILES, DATES, SUBJECTS = scan_folder_for_NWBfiles('/home/yann/DATA/CaImaging/NDNFcre_GCamp6s/Batch-2_September_2021', Nmax=3)
# find_protocol_details(FILES)
# create_file()

        

build_summary_episodes(FILES[:2], Nmax=2)

# for f in FILES[:2]:
#     data = Data(f)
#     print(data.protocols)
