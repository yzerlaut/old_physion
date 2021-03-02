import datetime, os
import numpy as np
from natsort import natsorted 

from pynwb import NWBFile
from pynwb.base import Images
from pynwb.image import GrayscaleImage
from pynwb.device import Device
from pynwb.ophys import OpticalChannel
from pynwb.ophys import TwoPhotonSeries
from pynwb.ophys import ImageSegmentation
from pynwb.ophys import RoiResponseSeries
from pynwb.ophys import Fluorescence
from pynwb import NWBHDF5IO


def add_ophys_processing_from_suite2p(save_folder, nwbfile, CaImaging_timestamps,
                                      device=None,
                                      optical_channel=None,
                                      imaging_plane=None,
                                      image_series=None):
    """ 
    adapted from suite2p/suite2p/io/nwb.py "save_nwb" function
    """

    plane_folders = natsorted([ f.path for f in os.scandir(save_folder) if f.is_dir() and f.name[:5]=='plane'])
    ops1 = [np.load(os.path.join(f, 'ops.npy'), allow_pickle=True).item() for f in plane_folders]
    if len(ops1)>1:
        multiplane = True
    else:
        multiplane = False

    ops = ops1[0]


    if device is None:
        device = nwbfile.create_device(
            name='Microscope', 
            description='My two-photon microscope',
            manufacturer='The best microscope manufacturer')
    if optical_channel is None:
        optical_channel = OpticalChannel(
            name='OpticalChannel', 
            description='an optical channel', 
            emission_lambda=500.)
    if imaging_plane is None:
        imaging_plane = nwbfile.create_imaging_plane(
            name='ImagingPlane',
            optical_channel=optical_channel,
            imaging_rate=ops['fs'],
            description='standard',
            device=device,
            excitation_lambda=600.,
            indicator='GCaMP',
            location='V1',
            grid_spacing=([2,2,30] if multiplane else [2,2]),
            grid_spacing_unit='microns')

    if image_series is None:
        # link to external data
        image_series = TwoPhotonSeries(
            name='TwoPhotonSeries', 
            dimension=[ops['Ly'], ops['Lx']],
            external_file=(ops['filelist'] if 'filelist' in ops else ['']), 
            imaging_plane=imaging_plane,
            starting_frame=[0], 
            format='external', 
            starting_time=0.0, 
            rate=ops['fs'] * ops['nplanes']
        )
        nwbfile.add_acquisition(image_series) # otherwise, were added

    # processing
    img_seg = ImageSegmentation()
    ps = img_seg.create_plane_segmentation(
        name='PlaneSegmentation',
        description='suite2p output',
        imaging_plane=imaging_plane,
        reference_images=image_series
    )
    ophys_module = nwbfile.create_processing_module(
        name='ophys', 
        description='optical physiology processed data'
    )
    ophys_module.add(img_seg)

    file_strs = ['F.npy', 'Fneu.npy', 'spks.npy']
    traces = []
    ncells_all = 0
    for iplane, ops in enumerate(ops1):
        if iplane==0:
            iscell = np.load(os.path.join(save_folder, 'plane%i' % iplane, 'iscell.npy'))
            for fstr in file_strs:
                traces.append(np.load(os.path.join(save_folder, 'plane%i' % iplane, fstr)))
        else:
            iscell = np.append(iscell, np.load(os.path.join(save_folder, 'plane%i' % iplane, 'iscell.npy')), axis=0)
            for i,fstr in enumerate(file_strs):
                traces[i] = np.append(traces[i], 
                                    np.load(os.path.join(save_folder, 'plane%i' % iplane, fstr)), axis=0) 

        stat = np.load(os.path.join(save_folder, 'plane%i' % iplane, 'stat.npy'), allow_pickle=True)
        ncells = len(stat)
        for n in range(ncells):
            if multiplane:
                pixel_mask = np.array([stat[n]['ypix'], stat[n]['xpix'], 
                                    iplane*np.ones(stat[n]['npix']), 
                                    stat[n]['lam']])
                ps.add_roi(voxel_mask=pixel_mask.T)
            else:
                pixel_mask = np.array([stat[n]['ypix'], stat[n]['xpix'], 
                                    stat[n]['lam']])
                ps.add_roi(pixel_mask=pixel_mask.T)
        ncells_all+=ncells

    ps.add_column('iscell', 'two columns - iscell & probcell', iscell)

    rt_region = ps.create_roi_table_region(
        region=list(np.arange(0, ncells_all)),
        description='all ROIs')

    # FLUORESCENCE (all are required)
    file_strs = ['F.npy', 'Fneu.npy', 'spks.npy']
    name_strs = ['Fluorescence', 'Neuropil', 'Deconvolved']

    for i, (fstr,nstr) in enumerate(zip(file_strs, name_strs)):
        roi_resp_series = RoiResponseSeries(
            name=nstr,
            data=traces[i],
            rois=rt_region,
            unit='lumens',
            timestamps=CaImaging_timestamps) # CRITICAL TO HAVE IT HERE FOR RE-ALIGNEMENT
        fl = Fluorescence(roi_response_series=roi_resp_series, name=nstr)
        ophys_module.add(fl)

    # BACKGROUNDS
    # (meanImg, Vcorr and max_proj are REQUIRED)
    bg_strs = ['meanImg', 'meanImgE', 'Vcorr', 'max_proj', 'meanImg_chan2']
    nplanes = ops['nplanes']
    for iplane in range(nplanes):
        images = Images('Backgrounds_%d'%iplane)
        for bstr in bg_strs:
            if bstr in ops:
                if bstr=='Vcorr' or bstr=='max_proj':
                    img = np.zeros((ops['Ly'], ops['Lx']), np.float32)
                    img[ops['yrange'][0]:ops['yrange'][-1], 
                        ops['xrange'][0]:ops['xrange'][-1]] = ops[bstr]
                else:
                    img = ops[bstr]
                images.add_image(GrayscaleImage(name=bstr, data=img))

        ophys_module.add(images)



