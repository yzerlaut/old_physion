import os, sys, pathlib, shutil, time, datetime, tempfile
from PIL import Image
import numpy as np

import pynwb
from hdmf.data_utils import DataChunkIterator
from hdmf.backends.hdf5.h5_utils import H5DataIO
from dateutil.tz import tzlocal

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import get_files_with_extension, list_dayfolder, check_datafolder, get_TSeries_folders, insure_ordered_frame_names, insure_ordered_FaceCamera_picture_names
from assembling.move_CaImaging_folders import StartTime_to_day_seconds
from assembling.realign_from_photodiode import realign_from_photodiode
from assembling.IO.binary import BinaryFile
from assembling.IO.bruker_xml_parser import bruker_xml_parser
from assembling.IO.suite2p_to_nwb import add_ophys_processing_from_suite2p
from behavioral_monitoring.locomotion import compute_position_from_binary_signals

def compute_locomotion(binary_signal, acq_freq=1e4,
                       speed_smoothing=10e-3, # s
                       t0=0):

    A = binary_signal%2
    B = np.round(binary_signal/2, 0)

    return compute_position_from_binary_signals(A, B,
                                                smoothing=int(speed_smoothing*acq_freq))


def build_subsampling_from_freq(subsampled_freq, original_freq, N, Nmin=3):
    """

    """
    if original_freq==0:
        print('  /!\ problem with original sampling freq /!\ ')
    if subsampled_freq==0:
        SUBSAMPLING = np.linspace(0, N-1, Nmin).astype(np.int)
    else:
        SUBSAMPLING = np.arange(0, N, max([int(subsampled_freq/original_freq),Nmin]))

    return SUBSAMPLING


ALL_MODALITIES = ['raw_CaImaging', 'processed_CaImaging',  'raw_FaceCamera', 'VisualStim', 'Locomotion', 'Pupil', 'Whisking', 'Electrophy']
LIGHT_MODALITIES = ['processed_CaImaging',  'VisualStim', 'Locomotion', 'Pupil', 'Whisking', 'Electrophy']


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
    if 'subject_props' in metadata and (metadata['subject_props'] is not None):
        subject_props = metadata['subject_props']
        print(subject_props)
        dob = subject_props['date_of_birth'].split('_')
    else:
        subject_props = {}
        print('subject properties not in metadata ...')
        dob = ['1988', '24', '4']
    
    subject = pynwb.file.Subject(description=(subject_props['description'] if ('description' in subject_props) else 'Unknown'),
                                 sex=(subject_props['sex'] if ('sex' in subject_props) else 'Unknown'),
                                 genotype=(subject_props['genotype'] if ('genotype' in subject_props) else 'Unknown'),
                                 species=(subject_props['species'] if ('species' in subject_props) else 'Unknown'),
                                 subject_id=(subject_props['subject_id'] if ('subject_id' in subject_props) else 'Unknown'),
                                 weight=(subject_props['weight'] if ('weight' in subject_props) else 'Unknown'),
	                         date_of_birth=datetime.datetime(int(dob[0]),int(dob[2]),int(dob[1])))
        
    nwbfile = pynwb.NWBFile(identifier='%s-%s' % (args.datafolder.split(os.path.sep)[-2],args.datafolder.split(os.path.sep)[-1]),
                            session_description=str(metadata),
                            experiment_description=metadata['protocol'],
                            experimenter=(metadata['experimenter'] if ('experimenter' in metadata) else 'Unknown'),
                            lab=(metadata['lab'] if ('lab' in metadata) else 'Unknown'),
                            institution=(metadata['institution'] if ('institution' in metadata) else 'Unknown'),
                            notes=(metadata['notes'] if ('notes' in metadata) else 'Unknown'),
                            virus=(subject_props['virus'] if ('virus' in subject_props) else 'Unknown'),
                            surgery=(subject_props['surgery'] if ('surgery' in subject_props) else 'Unknown'),
                            session_start_time=start_time,
                            subject=subject,
                            source_script=str(pathlib.Path(__file__).resolve()),
                            source_script_file_name=str(pathlib.Path(__file__).resolve()),
                            file_create_date=datetime.datetime.today())
    
    # deriving filename
    if args.export=='FULL' and (args.modalities==ALL_MODALITIES):
        filename = os.path.join(args.datafolder, '%s-%s.FULL.nwb' % (args.datafolder.split(os.path.sep)[-2],
                                                                     args.datafolder.split(os.path.sep)[-1]))
    elif (args.export=='LIGHTWEIGHT'):
        filename = os.path.join(args.datafolder, '%s-%s.LIGHTWEIGHT.nwb' % (args.datafolder.split(os.path.sep)[-2],
                                                                            args.datafolder.split(os.path.sep)[-1]))
    elif (args.export=='NIDAQ'):
        filename = os.path.join(args.datafolder, '%s-%s.NIDAQ.nwb' % (args.datafolder.split(os.path.sep)[-2],
                                                                            args.datafolder.split(os.path.sep)[-1]))
    elif args.export=='FROM_VISUALSTIM_SETUP':
        filename = os.path.join(args.datafolder, '%s-%s.nwb' % (args.datafolder.split(os.path.sep)[-2],
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
        if 'time_duration' not in VisualStim:
            VisualStim['time_duration'] = np.array(VisualStim['time_stop'])-np.array(VisualStim['time_start'])
        for key in ['time_start', 'time_stop', 'time_duration']:
            metadata[key] = VisualStim[key]
        success, metadata = realign_from_photodiode(NIdaq_data['analog'][0], metadata,
                                                    verbose=args.verbose)
        if success:
            print(metadata['time_start_realigned'])
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
                None_cond = (VisualStim[key]==None)
                if key in ['protocol_id', 'index']:
                    array = np.array(VisualStim[key])
                elif (type(VisualStim[key]) in [list, np.array]) and np.sum(None_cond)>0:
                    # need to remove the None elements
                    VisualStim[key][None_cond] = 0*VisualStim[key][~None_cond][0]
                    array = np.array(VisualStim[key], dtype=type(VisualStim[key][~None_cond][0]))
                else:
                    array = VisualStim[key]
                VisualStimProp = pynwb.TimeSeries(name=key,
                                                  data = array,
                                                  unit='NA',
                                                  timestamps=timestamps)
                nwbfile.add_stimulus(VisualStimProp)
        else:
            # TEMPORARY FOR TROUBLESHOOTING !!
            metadata['time_start_realigned'] = metadata['time_start']
            metadata['time_stop_realigned'] = metadata['time_stop']
            print(' /!\ Realignement unsuccessful /!\ ')
            print('       --> using the default time_start / time_stop values ')
    
        if args.verbose:
            print('=> Storing the photodiode signal [...]')
        photodiode = pynwb.TimeSeries(name='Photodiode-Signal',
                                      data = NIdaq_data['analog'][0],
                                      starting_time=0.,
                                      unit='[current]',
                                      rate=float(metadata['NIdaq-acquisition-frequency']))
        nwbfile.add_acquisition(photodiode)

        # if args.verbose:
        #     print('=> Storing the recorded frames [...]')
        # insure_ordered_frame_names(args.datafolder)
        # frames = np.sort(os.listdir(os.path.join(args.datafolder,'screen-frames')))
        # MOVIE = []
        # for fn in frames:
        #     im  = np.array(Image.open(os.path.join(args.datafolder,'screen-frames',fn))).mean(axis=-1)
        #     MOVIE.append(im.astype(np.uint8)[::8,::8]) # subsampling !
        # frame_timestamps = [0]
        # for x1, x2 in zip(metadata['time_start_realigned'], metadata['time_stop_realigned']):
        #     frame_timestamps.append(x1)
        #     frame_timestamps.append(x2)

        # frame_stimuli = pynwb.image.ImageSeries(name='visual-stimuli',
        #                                         data=np.array(MOVIE).astype(np.uint8),
        #                                         unit='NA',
        #                                         timestamps=np.array(frame_timestamps)[:len(MOVIE)])
        # nwbfile.add_stimulus(frame_stimuli)
        
    #################################################
    ####         FaceCamera Recording         #######
    #################################################

    if metadata['FaceCamera']:
        
        if args.verbose:
            print('=> Storing FaceCamera acquisition [...]')
        if ('raw_FaceCamera' in args.modalities):
            try:
                FC_FILES = os.listdir(os.path.join(args.datafolder, 'FaceCamera-imgs'))
                times = np.array([float(f.replace('.npy', '')) for f in FC_FILES])
                times = times-NIdaq_Tstart # converted to times relative to NIdaq start

                img = np.load(os.path.join(args.datafolder, 'FaceCamera-imgs', FC_FILES[0]))

                FC_SUBSAMPLING = build_subsampling_from_freq(args.FaceCamera_frame_sampling,
                                                          1./np.mean(np.diff(times)), len(FC_FILES), Nmin=3)
                def FaceCamera_frame_generator():
                    for i in FC_SUBSAMPLING:
                        yield np.load(os.path.join(args.datafolder, 'FaceCamera-imgs', FC_FILES[i])).astype(np.uint8)

                FC_dataI = DataChunkIterator(data=FaceCamera_frame_generator(),
                                             maxshape=(None, img.shape[0], img.shape[1]),
                                             dtype=np.dtype(np.uint8))
                FaceCamera_frames = pynwb.image.ImageSeries(name='FaceCamera',
                                                            data=FC_dataI,
                                                            unit='NA',
                                                            timestamps=times[FC_SUBSAMPLING])
                nwbfile.add_acquisition(FaceCamera_frames)

            except BaseException as be:
                print(be)
                FC_FILES = None
                print(' /!\ Problems with FaceCamera data /!\ ')
            

        #################################################
        ####         Pupil from FaceCamera        #######
        #################################################
        
        if 'Pupil' in args.modalities:
            
            if os.path.isfile(os.path.join(args.datafolder, 'pupil.npy')):
                
                data = np.load(os.path.join(args.datafolder, 'pupil.npy'),
                               allow_pickle=True).item()
                print(data.keys())
                
                for key in []:
                    PupilProp = pynwb.TimeSeries(name=key,
                                                 data = metadata[key],
                                                 unit='seconds',
                                                 timestamps=timestamps)
                    nwbfile.add_processing(PupilProp)
            else:
                print(' /!\ No processed pupil data found /!\ ')

                

                

                # then add the frames subsampled
                if FC_FILES is not None:
                    img = np.load(os.path.join(args.datafolder, 'FaceCamera-imgs', FC_FILES[0]))
                    x, y = np.meshgrid(np.arange(0,img.shape[0]), np.arange(0,img.shape[1]), indexing='ij')
                    cond = (x>=data['xmin']) & (x<=data['xmax']) & (y>=data['ymin']) & (y<=data['ymax'])

                    PUPIL_SUBSAMPLING = build_subsampling_from_freq(args.Pupil_frame_sampling,
                                                                    1./np.mean(np.diff(times)), len(FC_FILES), Nmin=3)
                    def Pupil_frame_generator():
                        for i in PUPIL_SUBSAMPLING:
                            yield np.load(os.path.join(args.datafolder, 'FaceCamera-imgs', FC_FILES[i])).astype(np.uint8)[cond].reshape(\
                                                                                            data['xmax']-data['xmin']+1, data['ymax']-data['ymin']+1)
            
                    PUC_dataI = DataChunkIterator(data=Pupil_frame_generator(),
                                                  maxshape=(None, data['xmax']-data['xmin']+1, data['ymax']-data['ymin']+1),
                                                  dtype=np.dtype(np.uint8))
                    Pupil_frames = pynwb.image.ImageSeries(name='Pupil',
                                                           data=PUC_dataI,
                                                           unit='NA',
                                                           timestamps=times[PUPIL_SUBSAMPLING])
                    nwbfile.add_acquisition(Pupil_frames)

        
            
    
        #################################################
        ####      Whisking from FaceCamera        #######
        #################################################
    
        if 'Whisking' in args.modalities:
            
            if os.path.isfile(os.path.join(args.datafolder, 'whisking.npy')):
                
                data = np.load(os.path.join(args.datafolder, 'whisking.npy'),
                               allow_pickle=True).item()
            

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
    
    Ca_data, Ca_folder = None, get_TSeries_folders(args.datafolder)
    if metadata['CaImaging'] and (len(Ca_folder)>0):
        Ca_subfolder = Ca_folder[0] # get Tseries folder
        try:
            CaFn = get_files_with_extension(Ca_subfolder, extension='.xml')[0] # get Tseries metadata
        except BaseException as be:
            print(be)
            print('\n /!\  Problem with the CA-IMAGING data in %s  /!\ ' % args.datafolder)
            raise Exception
        
        xml = bruker_xml_parser(CaFn) # metadata
        # CaImaging_timestamps = STEP_FOR_CA_IMAGING['onset']+xml['Ch1']['relativeTime']+\
        #     float(xml['settings']['framePeriod'])/2. # in the middle in-between two time stamps
        onset = (metadata['STEP_FOR_CA_IMAGING_TRIGGER']['onset'] if 'STEP_FOR_CA_IMAGING_TRIGGER' in metadata else 0)
        CaImaging_timestamps = onset+xml['Ch1']['relativeTime']+\
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

            i, dI = 0, int(args.CaImaging_frame_sampling/float(xml['settings']['framePeriod']))
            def Ca_frame_generator():
                while i<(Ca_data.shape[0]-dI):
                    i+=dI
                    yield Ca_data.data[i:i+dI, :, :].mean(axis=0).astype(np.uint8)
                    
            Ca_dataI = DataChunkIterator(data=Ca_frame_generator(),
                                         maxshape=(None, Ca_data.shape[1], Ca_data.shape[2]),
                                         dtype=np.dtype(np.uint8))
            if args.compression>0:
                Ca_dataC = H5DataIO(data=Ca_dataI, # with COMPRESSION
                                    compression='gzip',
                                    compression_opts=args.compression)
                image_series = pynwb.ophys.TwoPhotonSeries(name='CaImaging-TimeSeries',
                                                           dimension=[2],
                                                           data = Ca_dataC,
                                                           imaging_plane=imaging_plane,
                                                           unit='s',
                                                           timestamps = CaImaging_timestamps)
            else:
                image_series = pynwb.ophys.TwoPhotonSeries(name='CaImaging-TimeSeries',
                                                           dimension=[2],
                                                           data = Ca_dataI,
                                                           # data = Ca_data.data[:].astype(np.uint8),
                                                           imaging_plane=imaging_plane,
                                                           unit='s',
                                                           timestamps = CaImaging_timestamps)
        else:
            image_series = pynwb.ophys.TwoPhotonSeries(name='CaImaging-TimeSeries',
                                                       dimension=[2],
                                                       data = np.ones((2,2,2)),
                                                       imaging_plane=imaging_plane,
                                                       unit='s',
                                                       timestamps = np.arange(2))
        nwbfile.add_acquisition(image_series)

        if ('processed_CaImaging' in args.modalities) and os.path.isdir(os.path.join(Ca_subfolder, 'suite2p')):
            print('=> Adding the suite2p processing [...]')
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
    
    if Ca_data is not None:
        Ca_data.close()

    return filename
    
if __name__=='__main__':

    import argparse, os
    parser=argparse.ArgumentParser(description="""
    Building NWB file from mutlimodal experimental recordings
    """,formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-c', "--compression", type=int, default=0, help='compression level, from 0 (no compression) to 9 (large compression, SLOW)')
    parser.add_argument('-df', "--datafolder", type=str, default='')
    parser.add_argument('-rf', "--root_datafolder", type=str, default=os.path.join(os.path.expanduser('~'), 'DATA'))
    parser.add_argument('-m', "--modalities", nargs='*', type=str, default=ALL_MODALITIES)
    parser.add_argument('-d', "--day", type=str, default=datetime.datetime.today().strftime('%Y_%m_%d'))
    parser.add_argument('-t', "--time", type=str, default='')
    parser.add_argument('-e', "--export", type=str, default='FROM_VISUALSTIM_SETUP', help='export option [FULL / LIGHTWEIGHT / FROM_VISUALSTIM_SETUP]')
    parser.add_argument('-r', "--recursive", action="store_true")
    parser.add_argument('-v', "--verbose", action="store_true")
    parser.add_argument('-cafs', "--CaImaging_frame_sampling", default=0., type=float)
    parser.add_argument('-fcfs', "--FaceCamera_frame_sampling", default=0., type=float)
    parser.add_argument('-pfs', "--Pupil_frame_sampling", default=1., type=float)
    parser.add_argument('-sfs', "--Snout_frame_sampling", default=0.05, type=float)
    parser.add_argument("--silent", action="store_true")
    parser.add_argument('-lw', "--lightweight", action="store_true")
    parser.add_argument('-fvs', "--from_visualstim_setup", action="store_true")
    parser.add_argument('-ndo', "--nidaq_only", action="store_true")
    args = parser.parse_args()

    if not args.silent:
        args.verbose = True

    if args.export=='LIGHTWEIGHT' or args.lightweight:
        args.export='LIGHTWEIGHT'
        args.modalities = LIGHT_MODALITIES
    if args.nidaq_only:
        args.export='NIDAQ'
        args.modalities = ['VisualStim', 'Locomotion', 'Electrophy']        
    if args.from_visualstim_setup or (args.export=='FROM_VISUALSTIM_SETUP'):
        args.export='FROM_VISUALSTIM_SETUP'
        args.modalities = ['VisualStim', 'Locomotion', 'Electrophy', 'raw_FaceCamera', 'Pupil', 'Whisking']

    if args.time!='':
        args.datafolder = os.path.join(args.root_datafolder, args.day, args.time)
        
    if args.datafolder!='':
        if os.path.isdir(args.datafolder):
            if args.datafolder[-1]==os.path.sep:
                args.datafolder = args.datafolder[:-1]
            build_NWB(args)
        else:
            print('"%s" not a valid datafolder' % args.datafolder)
    elif args.root_datafolder!='':
        FOLDERS = [l for l in os.listdir(args.root_datafolder) if len(l)==8]
        for f in FOLDERS:
            args.datafolder = os.path.join(args.root_datafolder, f)
            try:
                build_NWB(args)
            except BaseException as e:
                print(e)
