import sys, os, pathlib, shutil, glob, time
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from hardware_control.Bruker.xml_parser import bruker_xml_parser
from assembling.saving import get_TSeries_folders
from Ca_imaging.process_xml import build_suite2p_options
import subprocess

folder = sys.argv[-1]

if os.path.isdir(folder):
    for f in get_TSeries_folders(folder):
        build_suite2p_options(f)
        shutil.rmtree(os.path.join(f, 'suite2p'))
        cmd = 'suite2p --db %s/db.npy --ops %s/ops.npy &' % (f, f)
        subprocess.run(cmd, shell=True)
else:
    print('/!\ Need to provide a valid folder /!\ ')
        



