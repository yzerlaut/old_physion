import os, sys, pathlib, shutil, time, datetime, tempfile
import numpy as np

import pynwb, time, ast
from hdmf.data_utils import DataChunkIterator
from hdmf.backends.hdf5.h5_utils import H5DataIO

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.IO.binary import BinaryFile
from assembling.IO.bruker_xml_parser import bruker_xml_parser
from assembling.saving import get_files_with_extension, get_TSeries_folders
from assembling.tools import build_subsampling_from_freq
from assembling.IO.suite2p_to_nwb import add_ophys_processing_from_suite2p

def append_to_NWB(args):

    io = pynwb.NWBHDF5IO(args.nwb_file, mode='a')
    nwbfile = io.read()

    if (not hasattr(args, 'datafolder')) or (args.datafolder==''):
        args.datafolder=os.path.dirname(args.nwb_file)
        
    add_ophys(nwbfile, args)

    io.write(nwbfile)
    io.close()

def add_ophys(nwbfile, args,
              metadata=None,
              with_raw_CaImaging=True,
              with_processed_CaImaging=True,
              Ca_Imaging_options={'Suite2P-binary-filename':'data.bin',
                                  'plane':0}):

    if metadata is None:
        metadata = ast.literal_eval(nwbfile.session_description)
    try:
        CaFn = get_files_with_extension(args.CaImaging_folder, extension='.xml')[0]# get Tseries metadata
    except BaseException as be:
        print(be)
        print('\n /!\  Problem with the CA-IMAGING data in %s  /!\ ' % args.datafolder)
        raise Exception
        
    xml = bruker_xml_parser(CaFn) # metadata

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

    if with_raw_CaImaging:
            
        if args.verbose:
            print('=> Storing Calcium Imaging data [...]')

        Ca_data = BinaryFile(Ly=int(xml['settings']['linesPerFrame']),
                             Lx=int(xml['settings']['pixelsPerLine']),
                             read_filename=os.path.join(args.CaImaging_folder,
                                        'suite2p', 'plane%i' % Ca_Imaging_options['plane'],
                                                        Ca_Imaging_options['Suite2P-binary-filename']))

        CA_SUBSAMPLING = build_subsampling_from_freq(args.CaImaging_frame_sampling,
                                                     float(xml['settings']['framePeriod']),
                                                     Ca_data.shape[0], Nmin=3)


        dI = int(args.CaImaging_frame_sampling/float(xml['settings']['framePeriod']))
        def Ca_frame_generator():
            for i in CA_SUBSAMPLING:
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
                                                       timestamps = CaImaging_timestamps[CA_SUBSAMPLING])
        else:
            image_series = pynwb.ophys.TwoPhotonSeries(name='CaImaging-TimeSeries',
                                                       dimension=[2],
                                                       data = Ca_dataI,
                                                       # data = Ca_data.data[:].astype(np.uint8),
                                                       imaging_plane=imaging_plane,
                                                       unit='s',
                                                       timestamps = CaImaging_timestamps[CA_SUBSAMPLING])
    else:
        image_series = pynwb.ophys.TwoPhotonSeries(name='CaImaging-TimeSeries',
                                                   dimension=[2],
                                                   data = np.ones((2,2,2)),
                                                   imaging_plane=imaging_plane,
                                                   unit='s',
                                                   timestamps = np.arange(2))
    nwbfile.add_acquisition(image_series)

    if with_processed_CaImaging and os.path.isdir(os.path.join(args.CaImaging_folder, 'suite2p')):
        print('=> Adding the suite2p processing [...]')
        add_ophys_processing_from_suite2p(os.path.join(args.CaImaging_folder, 'suite2p'), nwbfile,
                                          device=device,
                                          optical_channel=optical_channel,
                                          imaging_plane=imaging_plane,
                                          image_series=image_series)
    elif with_processed_CaImaging:
        print('\n /!\  no "suite2p" folder found in "%s"  /!\ ' % Ca_subfolder)

    return Ca_data

    
if __name__=='__main__':

    import argparse, os
    parser=argparse.ArgumentParser(description="""
    Building NWB file from mutlimodal experimental recordings
    """,formatter_class=argparse.RawTextHelpFormatter)
    # main
    parser.add_argument('-f', "--nwb_file", type=str, default='')
    parser.add_argument('-cf', "--CaImaging_folder", type=str, default='')
    # other
    parser.add_argument('-c', "--compression", type=int, default=0, help='compression level, from 0 (no compression) to 9 (large compression, SLOW)')
    parser.add_argument('-rf', "--root_datafolder", type=str, default=os.path.join(os.path.expanduser('~'), 'DATA'))
    parser.add_argument('-d', "--day", type=str, default=datetime.datetime.today().strftime('%Y_%m_%d'))
    parser.add_argument('-t', "--time", type=str, default='')
    parser.add_argument('-r', "--recursive", action="store_true")
    parser.add_argument('-v', "--verbose", action="store_true")
    parser.add_argument('-cafs', "--CaImaging_frame_sampling", default=0.5, type=float)
    parser.add_argument("--silent", action="store_true")
    args = parser.parse_args()

    # if not args.silent:
    #     args.verbose = True

    # if args.export=='LIGHTWEIGHT' or args.lightweight:
    #     args.export='LIGHTWEIGHT'
    #     args.modalities = LIGHT_MODALITIES
    # if args.nidaq_only:
    #     args.export='NIDAQ'
    #     args.modalities = ['VisualStim', 'Locomotion', 'Electrophy']        
    # if args.from_visualstim_setup or (args.export=='FROM_VISUALSTIM_SETUP'):
    #     args.export='FROM_VISUALSTIM_SETUP'
    #     args.modalities = ['VisualStim', 'Locomotion', 'Electrophy', 'raw_FaceCamera', 'Pupil', 'Whisking']

    # if args.time!='':
    #     args.datafolder = os.path.join(args.root_datafolder, args.day, args.time)
        
    # if args.datafolder!='':
    #     if os.path.isdir(args.datafolder):
    #         if args.datafolder[-1]==os.path.sep:
    #             args.datafolder = args.datafolder[:-1]
    #         build_NWB(args)
    #     else:
    #         print('"%s" not a valid datafolder' % args.datafolder)
    # elif args.root_datafolder!='':
    #     FOLDERS = [l for l in os.listdir(args.root_datafolder) if len(l)==8]
    #     for f in FOLDERS:
    #         args.datafolder = os.path.join(args.root_datafolder, f)
    #         try:
    #             build_NWB(args)
    #         except BaseException as e:
    #             print(e)
    append_to_NWB(args)
