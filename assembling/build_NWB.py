import os, sys, pathlib, shutil, time, datetime, tempfile
import numpy as np
import pynwb
from dateutil.tz import tzlocal

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import get_files_with_extension, list_dayfolder, check_datafolder, get_TSeries_folders
from assembling.move_CaImaging_folders import StartTime_to_day_seconds
from assembling.realign_from_photodiode import realign_from_photodiode
from assembling.IO.binary import BinaryFile
from assembling.IO.bruker_xml_parser import bruker_xml_parser
from behavioral_monitoring.locomotion import compute_position_from_binary_signals
from exp.gui import STEP_FOR_CA_IMAGING

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
        raise BaseException


    true_tstart0 = np.load(os.path.join(args.datafolder, 'NIdaq.start.npy'))[0]
    st = datetime.datetime.fromtimestamp(true_tstart0).strftime('%H:%M:%S.%f')
    true_tstart = StartTime_to_day_seconds(st)
    
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
        VisualStim = np.load(os.path.join(args.datafolder,
                        'visual-stim.npy'), allow_pickle=True).item()

        # using the photodiod signal for the realignement
        for key in ['time_start', 'time_stop']:
            metadata[key] = VisualStim[key]
        success, metadata = realign_from_photodiode(NIdaq_data['analog'][0], metadata,
                                                    verbose=args.verbose)
        if success:
            timestamps = metadata['time_start_realigned']
            if args.verbose:
                print('Realignement form photodiode successful')
        else:
            print(' /!\ Realignement unsuccessful /!\ ')
            
        for key in ['time_start_realigned', 'time_stop_realigned']:
            VisualStimProp = pynwb.TimeSeries(name=key,
                                              data = metadata[key],
                                              unit='seconds',
                                              timestamps=timestamps)
            nwbfile.add_stimulus(VisualStimProp)
        for key in VisualStim:
            VisualStimProp = pynwb.TimeSeries(name=key,
                                              data = VisualStim[key],
                                              unit='NA',
                                              timestamps=timestamps)
            nwbfile.add_stimulus(VisualStimProp)
            
        # storing photodiode signal
        photodiode = pynwb.TimeSeries(name='Photodiode-Signal',
                                      data = NIdaq_data['analog'][0],
                                      starting_time=0.,
                                      unit='[current]',
                                      rate=metadata['NIdaq-acquisition-frequency'])
        nwbfile.add_acquisition(photodiode)


        
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
    if metadata['CaImaging']:
        try:
            Ca_subfolder = get_TSeries_folders(args.datafolder)[0] # get Tseries folder
            CaFn = get_files_with_extension(Ca_subfolder, extension='.xml')[0] # get Tseries metadata
        except BaseException as be:
            print(be)
            print('\n /!\  Problem with the CA-IMAGING data in %s  /!\ ' % args.datafolder)
            raise Exception
        
        xml = bruker_xml_parser(CaFn) # metadata
        CaImaging_timestamps = STEP_FOR_CA_IMAGING['onset']+xml['Ch1']['relativeTime']+\
            float(xml['settings']['framePeriod'])/2. # in the middle in-between two time stamps
        
    if metadata['CaImaging'] and (args.export=='FULL'):
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
                                                   timestamps = CaImaging_timestamps)
        nwbfile.add_acquisition(image_series)
    
    
    #################################################
    ####         Writing NWB file             #######
    #################################################

    if args.export=='FULL':
        filename = os.path.join(args.datafolder, '%s-%s.FULL.nwb' % (args.datafolder.split(os.path.sep)[-2],
                                                                args.datafolder.split(os.path.sep)[-1]))
    else:
        filename = os.path.join(args.datafolder, '%s-%s.PROCESSED-ONLY.nwb' % (args.datafolder.split(os.path.sep)[-2],
                                                                          args.datafolder.split(os.path.sep)[-1]))

    if os.path.isfile(filename):
        temp = str(tempfile.NamedTemporaryFile().name)+'.nwb'
        print("""
        "%s" already exists
        ---> moving the file to the temporary file directory as: "%s" [...]
        """ % (filename, temp))
        shutil.move(filename, temp)
        print('---> done !')
        
    io = pynwb.NWBHDF5IO(filename, mode='w')
    print("""
    ---> Creating the NWB file: "%s"
    """ % filename)
    io.write(nwbfile)
    print('---> done !')
    io.close()

    return filename
    
        
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
