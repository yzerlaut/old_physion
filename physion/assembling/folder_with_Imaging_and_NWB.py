import os, sys, subprocess, shutil
import numpy as np

# df = sys.argv[1] # datafolder
# caf = sys.argv[2] # ca folder
nvs = sys.argv[3] # new visual stim

def run(rdf, rcaf):
    df = os.path.join(sys.argv[1], rdf)
    caf = os.path.join(sys.argv[2], rcaf)
    if not os.path.isfile(os.path.join(df, 'visual-stim-old.npy')) and os.path.isfile(os.path.join(df, 'visual-stim.npy')):
        shutil.move(os.path.join(df, 'visual-stim.npy'), os.path.join(df, 'visual-stim-old.npy'))
    shutil.copy(nvs, os.path.join(df, 'visual-stim.npy'))
    nwbfile = os.path.join(df.split('/')[0],df.split('/')[0]+'-'+df.split('/')[1]+'.nwb')
    cmd = """
    /home/yann.zerlaut/miniconda3/envs/physion/bin/python /home/yann.zerlaut/work/physion/physion/assembling/build_NWB.py -df %s --lightweight; /home/yann.zerlaut/miniconda3/envs/physion/bin/python /home/yann.zerlaut/work/physion/physion/assembling/add_ophys.py -f %s -cf %s; echo 'done' """ % (df, nwbfile, caf)
    p = subprocess.Popen(cmd, shell=True)

for df, caf in zip(\
        np.sort(os.listdir(sys.argv[1])),
        np.sort(os.listdir(sys.argv[2]))):
    run(df, caf)
