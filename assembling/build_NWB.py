import os, sys, pathlib, shutil, time, datetime
import numpy as np
import pynwb
from dateutil.tz import tzlocal

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import get_files_with_extension, list_dayfolder, check_datafolder, get_TSeries_folders

from behavioral_monitoring.locomotion import compute_position_from_binary_signals



def compute_locomotion(binary_signal, acq_freq=1e4,
                       speed_smoothing=10e-3, # s
                       t0=0):

    A = binary_signal%2
    B = np.round(binary_signal/2, 0)

    return compute_position_from_binary_signals(A, B,
                                                smoothing=int(speed_smoothing*acq_freq))


def build_NWB(datafolder):
    
    metadata = np.load(os.path.join(datafolder, 'metadata.npy'), allow_pickle=True).item()
    day = datafolder.split(os.path.sep)[-2].split('_')
    time = datafolder.split(os.path.sep)[-1].split('-')
    start_time = datetime.datetime(int(day[0]),int(day[1]),int(day[2]), int(time[0]),int(time[1]),int(time[2]),tzinfo=tzlocal())

    nwbfile = pynwb.NWBFile(session_description=metadata['protocol'],
                            identifier='NWB123',  # required
                            experimenter='Yann Zerlaut',
                            lab='Rebola and Bacci labs',
                            institution='Institut du Cerveau et de la Moelle, Paris',
                            session_start_time=start_time)  # optional
    

    #################################################
    ####         IMPORTING NI-DAQ data        #######
    #################################################
    try:
        NIdaq_data = np.load(os.path.join(datafolder, 'NIdaq.npy'), allow_pickle=True).item()
        NIdaq_Tstart = np.load(os.path.join(datafolder, 'NIdaq.start.npy'))[0]
    except FileNotFoundError:
        NIdaq_data, NIdaq_Tstart = None, None

    
    #################################################
    ####         Locomotion data              #######
    #################################################
    if metadata['Locomotion'] and (NIdaq_data is not None) and (NIdaq_Tstart is not None):
        # compute running speed from binary NI-daq signal
        running = pynwb.TimeSeries(name='Running-Speed',
                                   data = compute_locomotion(NIdaq_data['digital'][0],
                                                             acq_freq=metadata['NIdaq-acquisition-frequency']),
                                   unit='second', rate=metadata['NIdaq-acquisition-frequency'])
        nwbfile.add_acquisition(running)
    elif metadata['Locomotion']:
        print('\n /!\  NO NI-DAQ data found /!\ ')

        
    #################################################
    ####         Visual Stimulation           #######
    #################################################

    ## ---> Realignement


    filename = 
    io = pynwb.NWBHDF5IO(os.path.join(datafolder, 'full.nwb'), mode='w')
    io.write(nwbfile)
    io.close()
        

if __name__=='__main__':

    import argparse, os
    parser=argparse.ArgumentParser(description="transfer interface",
                       formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-rf', "--root_datafolder", type=str,
                        default=os.path.join(os.path.expanduser('~'), 'DATA'))
    parser.add_argument('-d', "--day", type=str,
                        default='2020_12_09')
    parser.add_argument('-wt', "--with_transfer", action="store_true")
    parser.add_argument('-v', "--verbose", action="store_true")
    args = parser.parse_args()

    if args.day!='':
        folder = os.path.join(args.root_datafolder, args.day)
    else:
        folder = args.root_datafolder

    PROTOCOL_LIST = list_dayfolder(folder)
    
    build_NWB(PROTOCOL_LIST[0])
    
    # if args.day!='':
    # else: # loop over days
    #     PROTOCOL_LIST = []
    #     for day in os.listdir(vis_folder):
    #         PROTOCOL_LIST += list_dayfolder(os.path.join(vis_folder, day))
    #     print(PROTOCOL_LIST)
    # CA_FILES = find_matching_data(PROTOCOL_LIST, CA_FILES,
    #                               verbose=args.verbose)

    # if args.with_transfer:
    #     transfer_analyzed_data(CA_FILES)
