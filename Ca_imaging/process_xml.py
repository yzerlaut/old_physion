import sys, os, pathlib, shutil, glob, time

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from hardware_control.Bruker.xml_parser import bruker_xml_parser
from assembling.saving import from_folder_to_datetime, check_datafolder, get_files_with_given_exts

# folder = '/home/yann/DATA/2020.09.25/M_1/TSeries-25092020-200-00-001'

# fn = get_files_with_given_exts(dir=folder, EXTS=['xml'])[0]

ops0 = {'look_one_level_down': 0.0,
       'delete_bin': False,
       'mesoscan': False,
       # 'bruker': True,
       'h5py': [],
       'h5py_key': 'data',
       'move_bin': False,
       'nplanes': 1,
       'nchannels': 1,
       'functional_chan': 1,
       'tau': 1.0,
       'fs': 10.0,
       'force_sktiff': False,
       'frames_include': -1,
       'multiplane_parallel': 0.0,
       'preclassify': 0.0,
       'save_mat': False,
       'save_NWB': 0.0,
       'combined': 1.0,
       'aspect': 1.0,
       'do_bidiphase': False,
       'bidiphase': 0.0,
       'bidi_corrected': False,
       'do_registration': 1,
       'two_step_registration': 0.0,
       'keep_movie_raw': False,
       'nimg_init': 300,
       'batch_size': 500,
       'maxregshift': 0.1,
       'align_by_chan': 1,
       'reg_tif': False,
       'reg_tif_chan2': False,
       'subpixel': 10,
       'smooth_sigma_time': 0.0,
       'smooth_sigma': 4.0,
       'th_badframes': 1.0,
       'pad_fft': False,
       'nonrigid': True,
       'block_size': [128, 128],
       'snr_thresh': 1.2,
       'maxregshiftNR': 8.0,
       '1Preg': True,
       'spatial_hp': 42,
       'spatial_hp_reg': 42.0,
       'spatial_hp_detect': 25,
       'pre_smooth': 0.0,
       'spatial_taper': 40.0,
       'roidetect': True,
       'spikedetect': True,
       'sparse_mode': True,
       'diameter': 12,
       'spatial_scale': 0,
       'connected': True,
       'nbinned': 5000,
       'max_iterations': 20,
       'threshold_scaling': 1.0,
       'max_overlap': 0.75,
       'high_pass': 6.0,
       'use_builtin_classifier': False,
       'inner_neuropil_radius': 2,
       'min_neuropil_pixels': 350,
       'allow_overlap': False,
       'chan2_thres': 0.65,
       'baseline': 'maximin',
       'win_baseline': 60.0,
       'sig_baseline': 10.0,
       'prctile_baseline': 8.0,
       'neucoeff': 0.7}

def build_db(folder):
    db = {'data_path':[folder],
          'subfolders': [],
          'save_path0': folder,
          'fast_disk': folder,
          'input_format': 'bruker'}
    return db

def build_ops(folder):
    return ops

def build_command(folder):

    xml_file = os.path.join(folder, os.path.join(folder.split('/')[-1]+'.xml'))
    
    bruker_data = bruker_xml_parser(xml_file)
    print(bruker_data)
    ops = ops0.copy()
    
    # acquisition frequency
    ops['fs'] = 1./float(bruker_data['settings']['framePeriod'])
    # zoom
    zoom = bruker_data['settings']['opticalZoom']
    
    db = build_db(folder)
    for key in ['data_path', 'subfolders', 'save_path0',
                'fast_disk', 'input_format']:
        ops[key] = db[key]

    

    
    
if __name__=='__main__':
    
    folder = '/media/yann/Yann/2020_11_10/TSeries-11102020-1605-016'
    
    xml_file = os.path.join(folder, os.path.join(folder.split('/')[-1]+'.xml'))
    
    bruker_data = bruker_xml_parser(xml_file)
    freq = 1./float(bruker_data['settings']['framePeriod'])
    print()
    # print(freq)
    # print(fn.split(os.path.sep))
    build_command(folder)
