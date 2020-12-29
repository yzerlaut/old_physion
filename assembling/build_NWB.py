import os, sys, pathlib, shutil, time, datetime
import numpy as np
import pynwb
from dateutil.tz import tzlocal

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import get_files_with_extension, list_dayfolder, check_datafolder, get_TSeries_folders
from assembling.IO.binary import BinaryFile
from assembling.IO.bruker_xml_parser import bruker_xml_parser
from behavioral_monitoring.locomotion import compute_position_from_binary_signals

def compute_locomotion(binary_signal, acq_freq=1e4,
                       speed_smoothing=10e-3, # s
                       t0=0):

    A = binary_signal%2
    B = np.round(binary_signal/2, 0)

    return compute_position_from_binary_signals(A, B,
                                                smoothing=int(speed_smoothing*acq_freq))


def build_NWB(args,
              Ca_Imaging_options={'Suite2P-binary-filename':'data_raw.bin',
                                  'plane':0}):
    
    #################################################
    ####            BASIC metadata            #######
    #################################################
    metadata = np.load(os.path.join(args.datafolder, 'metadata.npy'), allow_pickle=True).item()
    day = args.datafolder.split(os.path.sep)[-2].split('_')
    Time = args.datafolder.split(os.path.sep)[-1].split('-')
    start_time = datetime.datetime(int(day[0]),int(day[1]),int(day[2]), int(Time[0]),int(Time[1]),int(Time[2]),tzinfo=tzlocal())

    nwbfile = pynwb.NWBFile(session_description=metadata['protocol'],
                            identifier='NWB123',  # required
                            experimenter='Yann Zerlaut',
                            lab='Rebola and Bacci labs',
                            institution='Institut du Cerveau et de la Moelle, Paris',
                            session_start_time=start_time,
                            file_create_date=datetime.datetime.today())
    

    #################################################
    ####         IMPORTING NI-DAQ data        #######
    #################################################
    try:
        NIdaq_data = np.load(os.path.join(args.datafolder, 'NIdaq.npy'), allow_pickle=True).item()
        NIdaq_Tstart = np.load(os.path.join(args.datafolder, 'NIdaq.start.npy'))[0]
    except FileNotFoundError:
        print(' /!\ No NI-DAQ data found /!\ ')
        print('   -----> Not able to build NWB file')
        break

    
    #################################################
    ####         Locomotion                   #######
    #################################################
    if metadata['Locomotion']:
        # compute running speed from binary NI-daq signal
        running = pynwb.TimeSeries(name='Running-Speed',
                                   data = compute_locomotion(NIdaq_data['digital'][0],
                                                             acq_freq=metadata['NIdaq-acquisition-frequency']),
                                   starting_time=0.,
                                   unit='cm/s', rate=metadata['NIdaq-acquisition-frequency'])
        nwbfile.add_acquisition(running)

        
    #################################################
    ####         Visual Stimulation           #######
    #################################################
    if metadata['VisualStim']:
        if not os.path.isfile(os.path.join(args.datafolder, 'visual-stim.npy')):
            print(' /!\ No VisualStim metadata found /!\ ')
            print('   -----> Not able to build NWB file')
            break
        VisualStim = np.load(os.path.join(args.datafolder,
                        'visual-stim.npy'), allow_pickle=True).item()
        for key in VisualStim:
            VisualStimProp = pynwb.TimeSeries(name=key,
                                              data = VisualStim[key],
                                              unit='NA',
                                              timestamps=np.arange(len(VisualStim[key])))
            nwbfile.add_stimulus(VisualStimProp)
            
        # storing photodiode signal
        photodiode = pynwb.TimeSeries(name='Photodiode-Signal',
                                      data = NIdaq_data['analog'][0],
                                      starting_time=0.,
                                      unit='[current]',
                                      rate=metadata['NIdaq-acquisition-frequency'])
        nwbfile.add_acquisition(photodiode)

        # using the photodiod signal for the realignement
        for key in ['time_start', 'time_stop']:
            metadata[key] = VisualStim[key]
        success, metadata = realign_from_photodiode(NIdaq_data['analog'][0], metadata,
                                                    verbose=args.verbose)

        
    #################################################
    ####    Electrophysiological Recording    #######
    #################################################
    if metadata['Electrophy']:
        # storing electrophy signal
        electrophy = pynwb.TimeSeries(name='Electrophysiological-Signal',
                                      data = NIdaq_data['analog'][1],
                                      starting_time=0.,
                                      unit='[voltage]',
                                      rate=metadata['NIdaq-acquisition-frequency'])
        nwbfile.add_acquisition(electrophy)


    #################################################
    ####         Calcium Imaging              #######
    #################################################
    try:
        Ca_subfolder = get_TSeries_folders(args.datafolder)[0] # get Tseries folder
        CaFn = get_files_with_extension(Ca_subfolder, extension='.xml')[0] # get Tseries metadata
    except BaseException as be:
        Ca_subfolder, CaFn = None, None
        
    if metadata['CaImaging'] and (Ca_subfolder is not None) and (CaFn is not None) and (export=='FULL'):
        xml = bruker_xml_parser(CaFn) # metadata
        Ca_data = BinaryFile(Ly=int(xml['settings']['linesPerFrame']),
                             Lx=int(xml['settings']['pixelsPerLine']),
                             read_filename=os.path.join(Ca_subfolder, 'suite2p', 'plane%i' % Ca_Imaging_options['plane'],
                                                        Ca_Imaging_options['Suite2P-binary-filename']))

        device = pynwb.ophys.Device('Imaging device with settings: \n %s' % str(xml['settings'])) # TO BE FILLED
        nwbfile.add_device(device)
        optical_channel = pynwb.ophys.OpticalChannel('excitation_channel 1',
                                                     'Excitation 1',
                                                     float(xml['settings']['laserWavelength']['Excitation 1']))
        imaging_plane = nwbfile.create_imaging_plane('my_imgpln', optical_channel,
                                                     description='Depth=%.1f[um]' % float(xml['settings']['positionCurrent']['ZAxis']),
                                                     device=device,
                                                     excitation_lambda=float(xml['settings']['laserWavelength']['Excitation 1']),
                                                     imaging_rate=1./float(xml['settings']['framePeriod']),
                                                     indicator='GCamp',
                                                     location='V1',
                                                     # reference_frame='A frame to refer to',
                                                     grid_spacing=(float(xml['settings']['micronsPerPixel']['YAxis']),
                                                                   float(xml['settings']['micronsPerPixel']['XAxis'])))

        image_series = pynwb.ophys.TwoPhotonSeries(name='Ca[2+] imaging time series',
                                                   dimension=[2],
                                                   data = Ca_data.data[:],
                                                   imaging_plane=imaging_plane,
                                                   unit='s',
                                                   rate=1./float(xml['settings']['framePeriod']),
                                                   starting_time=0.0)
        nwbfile.add_acquisition(image_series)
    
    elif metadata['CaImaging']:
        print('\n /!\  No CA-IMAGING data found in %s  /!\ ' % args.datafolder)
        
    
    #################################################
    ####         Writing NWB file             #######
    #################################################

    if export=='FULL':
        filename = os.path.join(args.datafolder, '%s-%s.FULL.nwb' % (args.datafolder.split(os.path.sep)[-2],
                                                                args.datafolder.split(os.path.sep)[-1]))
    else:
        filename = os.path.join(args.datafolder, '%s-%s.PROCESSED-ONLY.nwb' % (args.datafolder.split(os.path.sep)[-2],
                                                                          args.datafolder.split(os.path.sep)[-1]))
        
    io = pynwb.NWBHDF5IO(filename, mode='w')
    io.write(nwbfile)
    io.close()

    return filename

    # io = pynwb.NWBHDF5IO(os.path.join(os.path.expanduser('~'), 'DATA', 'test.nwb'), mode='w')
    # pynwb.NWBHDF5IO.copy_file(filename,
    #                           os.path.join(os.path.expanduser('~'), 'DATA', 'test.nwb'),
    #                           expand_external=True)

    

def load(filename):

    # reading nwb
    io = pynwb.NWBHDF5IO(filename, 'r')
    t0 = time.time()
    nwbfile_in = io.read()
    print(nwbfile_in.acquisition['Running-Speed'].data)
    print(nwbfile_in.acquisition['Running-Speed'].timestamps)
    print(time.time()-t0)

    
        
if __name__=='__main__':

    import argparse, os
    parser=argparse.ArgumentParser(description="""
    Building NWB file from mutlimodal experimental recordings
    """,formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-df', "--datafolder", type=str, default='')
    parser.add_argument('-rf', "--root_datafolder", type=str, default='')
    parser.add_argument('-d', "--day", type=str, default='2020_12_09')
    parser.add_argument('-e', "--export", type=str, default='FULL', help='export option [FULL / PROCESSED-ONLY]')
    parser.add_argument('-r', "--recursive", action="store_true")
    parser.add_argument('-v', "--verbose", action="store_true")
    args = parser.parse_args()

    if args.datafolder!='':
        if os.path.isdir(args.datafolder):
            build_NWB(args)
        else:
            print('"%s" not a valid datafolder' % args.datafolder)

    # if args.day!='':
    #     folder = os.path.join(args.root_datafolder, args.day)
    # else:
    #     folder = args.root_datafolder

    # PROTOCOL_LIST = list_dayfolder(folder)
    
    # load(fn)
    
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
