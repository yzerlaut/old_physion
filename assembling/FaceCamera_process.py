import os, sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from hardware_control.Bruker.xml_parser import bruker_xml_parser
from assembling.saving import get_files_with_extension

def list_TSeries_folder(folder):
    folders = [os.path.join(folder, d) for d in sorted(os.listdir(folder)) if ((d[:7]=='TSeries') and os.path.isdir(os.path.join(folder, d)))]
    return folders


if __name__=='__main__':

    fn = '/home/yann/DATA/2020_11_04'
    for bdf in list_TSeries_folder(fn):
        xml = get_files_with_extension(bdf, extension='.xml')[0]
        print(bruker_xml_parser(xml))
