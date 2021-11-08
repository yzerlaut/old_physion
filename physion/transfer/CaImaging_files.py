import sys, os, pathlib, shutil, glob, time
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import get_TSeries_folders

source_folder1 = sys.argv[1]
destination_folder = sys.argv[2]

FILES = ['F', 'Fneu', 'iscell', 'ops', 'spks', 'stat', 'redcell']

for sf in os.listdir(source_folder1):
    source_folder = os.path.join(source_folder1, sf)
    print('---', source_folder)
    if os.path.isdir(source_folder):
        print('TSeries folders: ', get_TSeries_folders(source_folder))
        for f in get_TSeries_folders(source_folder):
            if os.path.isdir(os.path.join(f, 'suite2p', 'plane0')):
                print('copying files from "%s"' % f)
                new_folder = os.path.join(destination_folder, f.split(os.path.sep)[-1], 'suite2p', 'plane0')
                pathlib.Path(new_folder).mkdir(parents=True, exist_ok=True)
                for bfn in FILES:
                    old = os.path.join(f, 'suite2p', 'plane0', bfn+'.npy')
                    new = os.path.join(new_folder, bfn+'.npy')
                    shutil.copyfile(old, new)
                old = os.path.join(f, f.split(os.path.sep)[-1]+'.xml')
                new_folder = os.path.join(destination_folder, f.split(os.path.sep)[-1])
                new = os.path.join(new_folder, f.split(os.path.sep)[-1]+'.xml')
                shutil.copyfile(old,new)
        





