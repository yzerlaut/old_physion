import sys, os, pathlib, shutil, glob, time, subprocess
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from Ca_imaging.process_xml import build_suite2p_options
from misc.folders import python_path_suite2p_env

PREPROCESSING_SETTINGS = {
    'GCamp6s_1plane':{'cell_diameter':20, # in um
                      'tau':1.3,
                      'sparse_mode':False,
                      'connected':True,
                      'nonrigid':0,
                      'threshold_scaling':0.5,
                      'neucoeff': 0.7},
    'GCamp6s_1plane_A1':{'cell_diameter':20, # in um
                         'tau':1.3,
                         'nchannels':1,
                         'functional_chan':2,
                         'align_by_chan':2,
                         'sparse_mode':False,
                         'connected':True,
                         'nonrigid':0,
                         'threshold_scaling':0.5,
                         'neucoeff': 0.7},
    'INT_GCamp6s_1plane_A1':{'cell_diameter':20, # in um
                             'tau':1.3,
                            'nchannels':1,
                            'functional_chan':2,
                            'align_by_chan':2,
                            'sparse_mode':True,
                            'connected':True,
                             'anatomical_only': 2, # using the mean image only for ROI detection
                             'nonrigid':0,
                             'threshold_scaling':0.5,
                             'neucoeff': 0.7},
    'GCamp6s_5plane_A1':{'cell_diameter':20, # in um
                         'nplanes': 5,
                         'nchannels':1,
                         'functional_chan':2,
                         'align_by_chan':2,
                         'tau':1.3,
                         'sparse_mode':False,
                         'connected':True,
                         # 'nonrigid':0,
                         'threshold_scaling':0.5,
                         'neucoeff': 0.7},
    'INT_GCamp6s_5plane_A1':{'cell_diameter':20, # in um
                         'nplanes': 5,
                         'nchannels':1,
                         'functional_chan':2,
                         'align_by_chan':2,
                         'tau':1.3,
                         'sparse_mode':False,
                         'connected':True,
                         'nonrigid':0,
                         'anatomical_only': 2, # using the mean image only for ROI detection
                         'threshold_scaling':0.5,
                         'neucoeff': 0.7},
    'registration-only':{'do_registration': 1,
                         'nonrigid': False,
                         'roidetect':False}, 
    'GCamp6f_1plane':{'cell_diameter':20, # in um
                      'tau':0.7,
                      'sparse_mode':False,
                      'connected':True,
                      # 'nonrigid':0,
                      'threshold_scaling':0.5,
                      'neucoeff': 1.0},
    'GCamp6s_1plane_2chan':{'cell_diameter':20, # in um
                            'tau':1.3,
                            'nchannels':2,
                            'functional_chan':1,
                            'align_by_chan':2,
                            'sparse_mode':False,
                            # 'nonrigid':0,
                            'connected':True,
                            'threshold_scaling':0.5,
                            'neucoeff': 1.0},
    'NDNF+_1plane':{'cell_diameter':20, # in um
                    'sparse_mode':True,
                    'connected':True,
                    # 'nonrigid':0,
                    'threshold_scaling':0.5,
                    'neucoeff': 1.0},
}

def run_preprocessing(args):
    if args.remove_previous and (os.path.isdir(os.path.join(args.CaImaging_folder, 'suite2p'))):
        shutil.rmtree(os.path.join(args.CaImaging_folder, 'suite2p'))
    build_suite2p_options(args.CaImaging_folder, PREPROCESSING_SETTINGS[args.setting_key])
    cmd = '%s -m suite2p --db %s --ops %s &' % (python_path_suite2p_env,
                                     os.path.join(args.CaImaging_folder,'db.npy'),
                                     os.path.join(args.CaImaging_folder,'ops.npy'))
    subprocess.run(cmd, shell=True)
    

if __name__=='__main__':

    import argparse, os
    parser=argparse.ArgumentParser(description="""
    Launch preprocessing of Ca-Imaging data with Suite2P
    """,formatter_class=argparse.RawTextHelpFormatter)
    # main
    parser.add_argument('-cf', "--CaImaging_folder", type=str, default='./')
    descr = 'Available keys :\n'
    for s in PREPROCESSING_SETTINGS.keys():
        descr += ' - %s \n' % s
    parser.add_argument('-sk', "--setting_key", type=str, default='', help=descr)
    parser.add_argument('-v', "--verbose", action="store_true")
    parser.add_argument("--remove_previous", action="store_true")
    parser.add_argument("--silent", action="store_true")
    args = parser.parse_args()

    if os.path.isdir(str(args.CaImaging_folder)) and ('TSeries' in str(args.CaImaging_folder)):
        run_preprocessing(args)
        # print('--> preprocessing of "%s" done !' % args.CaImaging_folder)
    elif os.path.isdir(str(args.CaImaging_folder)):
        folders = [os.path.join(args.CaImaging_folder, f) for f in os.listdir(args.CaImaging_folder) if ('TSeries' in f)]
        for args.CaImaging_folder in folders:
            run_preprocessing(args)
    else:
        print('/!\ Need to provide a valid "TSeries" folder /!\ ')
        








