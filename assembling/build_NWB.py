import os, sys, pathlib, shutil, time, datetime, tempfile
from PIL import Image
import numpy as np

import pynwb
from hdmf.data_utils import DataChunkIterator
from dateutil.tz import tzlocal

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import get_files_with_extension, list_dayfolder, check_datafolder, get_TSeries_folders, insure_ordered_frame_names, insure_ordered_FaceCamera_picture_names
from assembling.move_CaImaging_folders import StartTime_to_day_seconds
from assembling.realign_from_photodiode import realign_from_photodiode
from assembling.IO.binary import BinaryFile
from assembling.IO.bruker_xml_parser import bruker_xml_parser
from assembling.IO.suite2p_to_nwb import add_ophys_processing_from_suite2p
from behavioral_monitoring.locomotion import compute_position_from_binary_signals
from exp.gui import STEP_FOR_CA_IMAGING

def compute_locomotion(binary_signal, acq_freq=1e4,
                       speed_smoothing=10e-3, # s
                       t0=0):

    A = binary_signal%2
    B = np.round(binary_signal/2, 0)

    return compute_position_from_binary_signals(A, B,
                                                smoothing=int(speed_smoothing*acq_freq))

ALL_MODALITIES = ['raw_CaImaging', 'processed_CaImaging',  'raw_FaceCamera', 'VisualStim', 'Locomotion', 'Pupil', 'Whisking', 'Electrophy']        

def build_NWB(args,
              Ca_Imaging_options={'Suite2P-binary-filename':'data.bin',
                                  'plane':0}):
    
    if args.verbose:
        print('Initializing NWB file [...]')

    #################################################
    ####            BASIC metadata            #######
    #################################################
    metadata = np.load(os.path.join(args.datafolder, 'metadata.npy'), allow_pickle=True).item()
    day = args.datafolder.split(os.path.sep)[-2].split('_')
    Time = args.datafolder.split(os.path.sep)[-1].split('-')
    start_time = datetime.datetime(int(day[0]),int(day[1]),int(day[2]),
                int(Time[0]),int(Time[1]),int(Time[2]),tzinfo=tzlocal())

    # subject info
    dob = metadata['subject_props']['date_of_birth'].split('_')
    subject = pynwb.file.Subject(description=metadata['subject_props']['description'],
                                 sex=metadata['subject_props']['sex'],
	                         genotype=metadata['subject_props']['genotype'],
                                 species=metadata['subject_props']['species'],
	                         subject_id=metadata['subject_props']['subject_id'],
	                         weight=metadata['subject_props']['weight'],
	                         date_of_birth=datetime.datetime(int(dob[0]),int(dob[1]),int(dob[2])))
        
    nwbfile = pynwb.NWBFile(identifier='%s-%s' % (args.datafolder.split(os.path.sep)[-2],args.datafolder.split(os.path.sep)[-1]),
                            session_description=str(metadata),
                            experiment_description=metadata['protocol'],
                            experimenter=metadata['experimenter'],
                            lab=metadata['lab'],
                            institution=metadata['institution'],
                            notes=metadata['notes'],
                            virus=metadata['subject_props']['virus'],
                            surgery=metadata['subject_props']['surgery'],
                            session_start_time=start_time,
                            subject=subject,
                            source_script=str(pathlib.Path(__file__).resolve()),
                            source_script_file_name=str(pathlib.Path(__file__).resolve()),
                            file_create_date=datetime.datetime.today())
    
    # deriving filename
    if args.export=='FULL' and (args.modalities==ALL_MODALITIES):
        filename = os.path.join(args.datafolder, '%s-%s.FULL.nwb' % (args.datafolder.split(os.path.sep)[-2],
                                                                     args.datafolder.split(os.path.sep)[-1]))
    elif (args.export=='LIGHTWEIGHT') and (args.modalities==ALL_MODALITIES):
        filename = os.path.join(args.datafolder, '%s-%s.LIGHTWEIGHT.nwb' % (args.datafolder.split(os.path.sep)[-2],
                                                                            args.datafolder.split(os.path.sep)[-1]))
    elif (args.modalities!=ALL_MODALITIES):
        filename = os.path.join(args.datafolder, '%s-%s.%s.nwb' % (args.datafolder.split(os.path.sep)[-2],
                                                                   args.datafolder.split(os.path.sep)[-1],
                                                                   str(args.modalities)))
    else:
        raise BaseException(2*'\n'+10*' '+ '===> Export format not recognized !')
    
    manager = pynwb.get_manager() # we need a manager to link raw and processed data
    
    #################################################
    ####         IMPORTING NI-DAQ data        #######
    #################################################
    if args.verbose:
        print('Loading NIdaq data [...]')
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
    
    # #################################################
    # ####         Locomotion                   #######
    # #################################################

    if metadata['Locomotion'] and ('Locomotion' in args.modalities):
        # compute running speed from binary NI-daq signal
        if args.verbose:
            print('Computing and storing running-speed [...]')
        running = pynwb.TimeSeries(name='Running-Speed',
                                   data = compute_locomotion(NIdaq_data['digital'][0],
                                                             acq_freq=metadata['NIdaq-acquisition-frequency']),
                                   starting_time=0.,
                                   unit='cm/s', rate=float(metadata['NIdaq-acquisition-frequency']))
        nwbfile.add_acquisition(running)

    # #################################################
    # ####         Visual Stimulation           #######
    # #################################################
    if metadata['VisualStim'] and ('VisualStim' in args.modalities):
        if not os.path.isfile(os.path.join(args.datafolder, 'visual-stim.npy')):
            print(' /!\ No VisualStim metadata found /!\ ')
            print('   -----> Not able to build NWB file')
        VisualStim = np.load(os.path.join(args.datafolder,
                        'visual-stim.npy'), allow_pickle=True).item()
        # using the photodiod signal for the realignement
        if args.verbose:
            print('=> Performing realignement from photodiode [...]')
        if 'refresh-times' in VisualStim: # FOR NOISE PROTOCOLS !!
            metadata['time_start'] = VisualStim['refresh-times'][:-1]
            metadata['time_stop'] = VisualStim['refresh-times'][1:]
        else:
            for key in ['time_start', 'time_stop']:
                metadata[key] = VisualStim[key]
        success, metadata = realign_from_photodiode(NIdaq_data['analog'][0], metadata,
                                                    verbose=args.verbose)
        if success:
            timestamps = metadata['time_start_realigned']
            if args.verbose:
                print('Realignement form photodiode successful')
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
        else:
            # TEMPORARY FOR TROUBLESHOOTING !!
            metadata['time_start_realigned'] = metadata['time_start']
            metadata['time_stop_realigned'] = metadata['time_stop']
            print(' /!\ Realignement unsuccessful /!\ ')

        if args.verbose:
            print('=> Storing the photodiode signal [...]')
        photodiode = pynwb.TimeSeries(name='Photodiode-Signal',
                                      data = NIdaq_data['analog'][0],
                                      starting_time=0.,
                                      unit='[current]',
                                      rate=float(metadata['NIdaq-acquisition-frequency']))
        nwbfile.add_acquisition(photodiode)

        if args.verbose:
            print('=> Storing the recorded frames [...]')
        insure_ordered_frame_names(args.datafolder)
        frames = np.sort(os.listdir(os.path.join(args.datafolder,'screen-frames')))
        MOVIE = []
        for fn in frames:
            im  = np.array(Image.open(os.path.join(args.datafolder,'screen-frames',fn))).mean(axis=-1)
            MOVIE.append(im.astype(np.uint8)[::8,::8]) # subsampling !
        frame_timestamps = [0]
        for x1, x2 in zip(metadata['time_start_realigned'], metadata['time_stop_realigned']):
            frame_timestamps.append(x1)
            frame_timestamps.append(x2)

        frame_stimuli = pynwb.image.ImageSeries(name='visual-stimuli',
                                                data=np.array(MOVIE).astype(np.uint8),
                                                unit='NA',
                                                timestamps=np.array(frame_timestamps)[:len(MOVIE)])
        nwbfile.add_stimulus(frame_stimuli)
        
    #################################################
    ####         FaceCamera Recording         #######
    #################################################

    
    if metadata['FaceCamera'] and ('raw_FaceCamera' in args.modalities):
        
        if args.verbose:
            print('=> Storing FaceCamera acquisition [...]')
        if not os.path.isfile(os.path.join(args.datafolder, 'FaceCamera-times.npy')):
            print(' /!\ No FaceCamera metadata found /!\ ')
# <<<<<<< HEAD
#         else:
#             FaceCamera_times = np.load(os.path.join(args.datafolder,
#                                           'FaceCamera-times.npy'))
#             insure_ordered_FaceCamera_picture_names(args.datafolder)
#             FaceCamera_times = FaceCamera_times-NIdaq_Tstart # times relative to NIdaq start
#             IMGS = []
#             for fn in np.sort(os.listdir(os.path.join(args.datafolder, 'FaceCamera-imgs'))):
#                 IMGS.append(np.load(os.path.join(args.datafolder, 'FaceCamera-imgs', fn)))
#             FaceCamera_frames = pynwb.image.ImageSeries(name='FaceCamera',
#                                                         data=np.array(IMGS).astype(np.uint8),
#                                                         unit='NA',
#                                                         timestamps=np.array(FaceCamera_times))
#             nwbfile.add_acquisition(FaceCamera_frames)
# =======
            print('   -----> Not able to build NWB file')
        FaceCamera_times = np.load(os.path.join(args.datafolder,
                                      'FaceCamera-times.npy'))
        insure_ordered_FaceCamera_picture_names(args.datafolder)
        FaceCamera_times = FaceCamera_times-NIdaq_Tstart # times relative to NIdaq start

        IMAGES = np.sort(os.listdir(os.path.join(args.datafolder,
                                                 'FaceCamera-imgs')))
        img = np.load(os.path.join(args.datafolder,
                         'FaceCamera-imgs', IMAGES[0]))
        def frame_generator():
            for i, fn in enumerate(FILES):
                yield np.load(os.path.join(args.datafolder,
                            'FaceCamera-imgs', fn)).astype(np.uint8)

        data = DataChunkIterator(data=frame_generator(),
            maxshape=(None,self.img.shape[0], self.img.shape[1]),
            dtype=np.dtype(np.uint8))
            
        FaceCamera_frames = pynwb.image.ImageSeries(name='FaceCamera',
                                                    data=data,
                                                    unit='NA',
                                timestamps=np.array(FaceCamera_times))
        nwbfile.add_acquisition(FaceCamera_frames)

# >>>>>>> 165d72d35bc04343d8e9ad82d76741416d94ceaf
        
    #################################################
    ####    Electrophysiological Recording    #######
    #################################################
    
    if metadata['Electrophy'] and ('Electrophy' in args.modalities):
    
        if args.verbose:
            print('=> Storing electrophysiological signal [...]')
        electrophy = pynwb.TimeSeries(name='Electrophysiological-Signal',
                                      data = NIdaq_data['analog'][1],
                                      starting_time=0.,
                                      unit='[voltage]',
                                      rate=float(metadata['NIdaq-acquisition-frequency']))
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

        if  'raw_CaImaging' in args.modalities:
            
            if args.verbose:
                print('=> Storing Calcium Imaging data [...]')
                
            Ca_data = BinaryFile(Ly=int(xml['settings']['linesPerFrame']),
                                 Lx=int(xml['settings']['pixelsPerLine']),
                                 read_filename=os.path.join(Ca_subfolder, 'suite2p', 'plane%i' % Ca_Imaging_options['plane'],
                                                            Ca_Imaging_options['Suite2P-binary-filename']))
            image_series = pynwb.ophys.TwoPhotonSeries(name='CaImaging-TimeSeries',
                                                       dimension=[2],
                                                       data = Ca_data.data[:].astype(np.uint16),
                                                       imaging_plane=imaging_plane,
                                                       unit='s',
                                                       timestamps = CaImaging_timestamps)
            Ca_data.close()
        else:
            image_series = pynwb.ophys.TwoPhotonSeries(name='CaImaging-TimeSeries',
                                                       dimension=[2],
                                                       data = np.ones((2,2,2)),
                                                       imaging_plane=imaging_plane,
                                                       unit='s',
                                                       timestamps = np.arange(2))
        nwbfile.add_acquisition(image_series)

        if ('processed_CaImaging' in args.modalities) and os.path.isdir(os.path.join(Ca_subfolder, 'suite2p')):
            # we add the suite2p processing !
            add_ophys_processing_from_suite2p(os.path.join(Ca_subfolder, 'suite2p'), nwbfile,
                                              device=device,
                                              optical_channel=optical_channel,
                                              imaging_plane=imaging_plane,
                                              image_series=image_series)
        elif ('processed_CaImaging' in args.modalities):
            print('\n /!\  no "suite2p" folder found in "%s"  /!\ ' % Ca_subfolder)


    #################################################
    ####         Writing NWB file             #######
    #################################################

    if os.path.isfile(filename):
        temp = str(tempfile.NamedTemporaryFile().name)+'.nwb'
        print("""
        "%s" already exists
        ---> moving the file to the temporary file directory as: "%s" [...]
        """ % (filename, temp))
        shutil.move(filename, temp)
        print('---> done !')
        
    io = pynwb.NWBHDF5IO(filename, mode='w', manager=manager)
    print("""
    ---> Creating the NWB file: "%s"
    """ % filename)
    io.write(nwbfile, link_data=False)
    io.close()
    print('---> done !')
    
    return filename
    
if __name__=='__main__':

    import argparse, os
    parser=argparse.ArgumentParser(description="""
    Building NWB file from mutlimodal experimental recordings
    """,formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-df', "--datafolder", type=str, default='')
    parser.add_argument('-rf', "--root_datafolder", type=str, default='')
    parser.add_argument('-m', "--modalities", nargs='*', type=str, default=ALL_MODALITIES)
    parser.add_argument('-d', "--day", type=str, default='2020_12_09')
    parser.add_argument('-e', "--export", type=str, default='FULL', help='export option [FULL / PROCESSED-ONLY]')
    parser.add_argument('-r', "--recursive", action="store_true")
    parser.add_argument('-v', "--verbose", action="store_true")
    parser.add_argument("--silent", action="store_true")
    args = parser.parse_args()

    if not args.silent:
        args.verbose = True

    if args.datafolder!='':
        if os.path.isdir(args.datafolder):
            if args.datafolder[-1]==os.path.sep:
                args.datafolder = args.datafolder[:-1]
            build_NWB(args)
        else:
            print('"%s" not a valid datafolder' % args.datafolder)
