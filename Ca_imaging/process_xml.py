import sys, os, pathlib, shutil, glob, time

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from hardware_control.Bruker.xml_parser import bruker_xml_parser
from assembling.saving import from_folder_to_datetime, check_datafolder, get_files_with_given_exts

# folder = '/home/yann/DATA/2020.09.25/M_1/TSeries-25092020-200-00-001'

# fn = get_files_with_given_exts(dir=folder, EXTS=['xml'])[0]

# print(bruker_xml_parser(fn))


    
if __name__=='__main__':
    
    folder = '/media/yann/Yann/2020_11_10/TSeries-11102020-1605-016'
    
    xml_file = os.path.join(folder, os.path.join(folder.split('/')[-1],'xml'))
    
    print(xml_file)
    # print(fn.split(os.path.sep))
