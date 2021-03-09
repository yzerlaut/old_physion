import sys, os, pathlib, shutil, glob, time
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import get_TSeries_folders
from Ca_imaging.process_xml import build_suite2p_options
import subprocess

PREPROCESSING_SETTINGS = {
    'GCamp6f_1plane':{'cell_diameter':20, # in um
                      'sparse_mode':False,
                      'connected':True,
                      'threshold_scaling':0.8},
    'NDNF+_1plane':{},
}

# folder = sys.argv[-1]

# if 'TSeries' in str(folder):
#     build_suite2p_options(folder)
    
# if os.path.isdir(folder):
#     for f in get_TSeries_folders(folder):
#         build_suite2p_options(f)
#         if os.path.isdir(os.path.join(f, 'suite2p')):
#             shutil.rmtree(os.path.join(f, 'suite2p'))
#         cmd = 'suite2p --db %s/db.npy --ops %s/ops.npy &' % (f, f)
#         subprocess.run(cmd, shell=True)
# else:
#     print('/!\ Need to provide a valid folder /!\ ')
        



