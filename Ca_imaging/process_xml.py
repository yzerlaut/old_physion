import sys, os, pathlib, shutil, glob, time

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from hardware_control.Bruker.xml_parser import bruker_xml_parser
from assembling.saving import from_folder_to_datetime, check_datafolder, get_files_with_given_exts

folder = '/home/yann/DATA/2020.09.25/M_1/TSeries-25092020-200-00-001'

fn = get_files_with_given_exts(dir=folder, EXTS=['xml'])[0]

print(bruker_xml_parser(fn))


    
