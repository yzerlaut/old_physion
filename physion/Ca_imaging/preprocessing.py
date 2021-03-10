import sys, os, pathlib, shutil, glob, time, subprocess
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from Ca_imaging.process_xml import build_suite2p_options

PREPROCESSING_SETTINGS = {
    'GCamp6f_1plane':{'cell_diameter':20, # in um
                      'sparse_mode':False,
                      'connected':True,
                      'threshold_scaling':0.8},
    'NDNF+_1plane':{'cell_diameter':20, # in um
                    'sparse_mode':True,
                    'connected':True,
                    'threshold_scaling':0.8},
}

def run_preprocessing(args):
    if args.remove_previous and (os.path.isdir(os.path.join(args.CaImaging_folder, 'suite2p'))):
        shutil.rmtree(os.path.join(args.CaImaging_folder, 'suite2p'))
    build_suite2p_options(args.CaImaging_folder, args.setting_key)
    cmd = 'suite2p --db %s/db.npy --ops %s/ops.npy &' % (f, f)
    subprocess.run(cmd, shell=True)
    

if __name__=='__main__':

    import argparse, os
    parser=argparse.ArgumentParser(description="""
    Building NWB file from mutlimodal experimental recordings
    """,formatter_class=argparse.RawTextHelpFormatter)
    # main
    parser.add_argument('-cf', "--CaImaging_folder", type=str, default='')
    parser.add_argument('-sk', "--setting_key", type=str, default='')
    parser.add_argument('-v', "--verbose", action="store_true")
    parser.add_argument("--remove_previous", action="store_true")
    parser.add_argument("--silent", action="store_true")
    args = parser.parse_args()

    if os.path.isdir(args.CaImaging_folder) and ('TSeries' in str(args.CaImaging_folder)):
        run_preprocessing(args)
        print('--> preprocessing of "%s" done !' % args.CaImaging_folder)
    else:
        print('/!\ Need to provide a valid "TSeries" folder /!\ ')
        
