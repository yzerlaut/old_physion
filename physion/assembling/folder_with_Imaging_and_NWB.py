import os, sys, subprocess, shutil
import numpy as np

folder = sys.argv[1]

Imaging_folders = np.sort([os.path.join(folder, f) for f in os.listdir(folder) if 'TSeries' in f])
NWB_files = np.sort([os.path.join(folder, f) for f in os.listdir(folder) if '.nwb' in f])

def run(caf, nwb):
    cmd = """
    /home/yann.zerlaut/miniconda3/envs/physion/bin/python /home/yann.zerlaut/work/physion/physion/assembling/add_ophys.py -f %s -cf %s; echo 'done' """ % (nwb, caf)
    p = subprocess.Popen(cmd, shell=True)

for caf, nwb in zip(Imaging_folders, NWB_files):
    print(caf, nwb)
    run(caf, nwb)
