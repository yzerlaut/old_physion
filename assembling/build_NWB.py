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


def build_NWB(datafolder,
              Ca_Imaging_options={'Suite2P-binary-filename':'data.bin',
                                  'plane':0}):
    
    #################################################
    ####            BASIC metadata            #######
    #################################################
    metadata = np.load(os.path.join(datafolder, 'metadata.npy'), allow_pickle=True).item()
    day = datafolder.split(os.path.sep)[-2].split('_')
    Time = datafolder.split(os.path.sep)[-1].split('-')
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
        NIdaq_data = np.load(os.path.join(datafolder, 'NIdaq.npy'), allow_pickle=True).item()
        NIdaq_Tstart = np.load(os.path.join(datafolder, 'NIdaq.start.npy'))[0]
    except FileNotFoundError:
        # print(' /!\ No NI-DAQ data found .... /!\ ')
        NIdaq_data, NIdaq_Tstart = None, None

    
    #################################################
    ####         Locomotion                   #######
    #################################################
    if metadata['Locomotion'] and (NIdaq_data is not None) and (NIdaq_Tstart is not None):
        # compute running speed from binary NI-daq signal
        running = pynwb.TimeSeries(name='Running-Speed',
                                   data = compute_locomotion(NIdaq_data['digital'][0],
                                                             acq_freq=metadata['NIdaq-acquisition-frequency']),
                                   starting_time=0.,
                                   unit='cm/s', rate=metadata['NIdaq-acquisition-frequency'])
        nwbfile.add_acquisition(running)
    elif metadata['Locomotion']:
        print('\n /!\  NO NI-DAQ data found to fill the Locomotion data in %s  /!\ ' % datafolder)

        
    #################################################
    ####         Visual Stimulation           #######
    #################################################
    if os.path.isfile(os.path.join(datafolder, 'visual-stim.npy')):
        VisualStim = np.load(os.path.join(datafolder,
                        'visual-stim.npy'), allow_pickle=True).item()
    else:
        VisualStim = None
        print('[X] Visual-Stim metadata not found !')

    if metadata['VisualStim'] and VisualStim is not None:
        ATTRIBUTES = []
        for key in VisualStim:
            ATTRIBUTES.append()
            
    elif metadata['VisualStim']:
        print('\n /!\  No VisualStim metadata found for %s  /!\ ' % datafolder)
        
    # if self.metadata['VisualStim'] and ('Screen' in modalities):
    #     self.Screen = ScreenData(self.datafolder, self.metadata,
    #                              NIdaq_trace=data['analog'][Photodiode_NIdaqChannel,:])
    # elif 'Screen' in modalities:
    #     print('[X] Screen data not found !')
    #             self.VisualStim = np.load(os.path.join(self.datafolder, 'visual-stim.npy'), allow_pickle=True).item()

    ################################
    ## ---> Realignement <----- ####
    ###########  Can we realign ? ##
    if True:
        pass

    

    #################################################
    ####         Calcium Imaging              #######
    #################################################
    try:
        
        Ca_subfolder = get_TSeries_folders(datafolder)[0] # get Tseries folder
        CaFn = get_files_with_extension(Ca_subfolder, extension='.xml')[0] # get Tseries metadata
    except BaseException as be:
        Ca_subfolder, CaFn = None, None
        
    if metadata['CaImaging'] and Ca_subfolder is not None and CaFn is not None:
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
        print('\n /!\  No CA-IMAGING data found in %s  /!\ ' % datafolder)
        
    
    #################################################
    ####         Writing NWB file             #######
    #################################################
    
    filename = os.path.join(datafolder, '%s-%s-%s.nwb' % (datafolder.split(os.path.sep)[-2], datafolder.split(os.path.sep)[-1], metadata['protocol']))
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
    
    fn = build_NWB(PROTOCOL_LIST[0])
    load(fn)
    
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
