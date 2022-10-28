import sys, os, pathlib, shutil, glob, time, subprocess
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from Ca_imaging.process_xml import build_suite2p_options
from misc.folders import python_path_suite2p_env

PREPROCESSING_SETTINGS = {
    'registration-only':{'do_registration': 1,
                         'nchannels':1,
                         'functional_chan':1,
                         'nonrigid': False,
                         'roidetect':False}, 
    'GCamp6s_1plane':{'cell_diameter':20, # in um
                     'tau':1.3,
                     'nchannels':1,
                     'functional_chan':1,
                     'align_by_chan':1,
                     'sparse_mode':False,
                     'connected':True,
                     'nonrigid':0,
                     'threshold_scaling':0.5,
                     'mask_threshold':0.3,
                     'neucoeff': 0.7}
    }

# multiplane imaging
for nplanes in [1, 3, 5, 7]:

    # for pyramidal cells
    PREPROCESSING_SETTINGS['GCamp6s_%iplane' % nplanes] = PREPROCESSING_SETTINGS['GCamp6s_1plane'].copy()
    PREPROCESSING_SETTINGS['GCamp6s_%iplane' % nplanes]['nplanes'] = nplanes

    # for interneurons
    PREPROCESSING_SETTINGS['INT_GCamp6s_%iplane' % nplanes] = PREPROCESSING_SETTINGS['GCamp6s_%iplane' % nplanes].copy()
    PREPROCESSING_SETTINGS['INT_GCamp6s_%iplane' % nplanes]['anatomical_only'] = 3
    PREPROCESSING_SETTINGS['INT_GCamp6s_%iplane' % nplanes]['high_pass'] = 1
    PREPROCESSING_SETTINGS['INT_GCamp6s_%iplane' % nplanes]['nchannels'] = 2 # ASSUMES tdTomato always !!

# dealing with the specifics of the A1 settings
for key in list(PREPROCESSING_SETTINGS.keys()):
    PREPROCESSING_SETTINGS[key+'_A1'] = PREPROCESSING_SETTINGS[key].copy()
    PREPROCESSING_SETTINGS[key+'_A1']['nchannels'] = 2
    PREPROCESSING_SETTINGS[key+'_A1']['functional_chan'] = 2
    PREPROCESSING_SETTINGS[key+'_A1']['align_by_chan'] = 2

    
print(PREPROCESSING_SETTINGS)

def run_preprocessing(args):
    if args.remove_previous and (os.path.isdir(os.path.join(args.CaImaging_folder, 'suite2p'))):
        shutil.rmtree(os.path.join(args.CaImaging_folder, 'suite2p'))
    build_suite2p_options(args.CaImaging_folder, PREPROCESSING_SETTINGS[args.setting_key])
    cmd = '%s -m suite2p --db "%s" --ops "%s" &' % (python_path_suite2p_env,
                                     os.path.join(args.CaImaging_folder,'db.npy'),
                                     os.path.join(args.CaImaging_folder,'ops.npy'))
    print(cmd)
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
        








