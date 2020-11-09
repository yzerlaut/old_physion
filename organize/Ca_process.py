import os, sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from hardware_control.Bruker.xml_parser import bruker_xml_parser
from assembling.saving import get_files_with_extension

def list_TSeries_folder(folder):
    folders = [os.path.join(folder, d) for d in sorted(os.listdir(folder)) if ((d[:7]=='TSeries') and os.path.isdir(os.path.join(folder, d)))]
    return folders

def StartTime_to_day_seconds(StartTime):

    Hour = int(StartTime[0:2])
    Min = int(StartTime[3:5])
    Seconds = float(StartTime[6:])

    return 60*60*Hour+60*Min+Seconds

def build_Ca_filelist(folder):
    
    CA_FILES = {'Bruker_folder':[], 'Bruker_file':[],
                'StartTime':[], 'EndTime':[]}
    for bdf in list_TSeries_folder(folder):
        fn = get_files_with_extension(bdf, extension='.xml')[0]
        xml = bruker_xml_parser(fn)
        if len(xml['Ch1']['relativeTime'])>0:
            CA_FILES['Bruker_folder'].append(bdf)
            CA_FILES['Bruker_file'].append(fn)
            start = StartTime_to_day_seconds(xml['StartTime'])
            CA_FILES['StartTime'].append(start)
            CA_FILES['EndTime'].append(start+xml['Ch1']['relativeTime'][-1])
            
    return CA_FILES

if __name__=='__main__':

    folder = '/home/yann/DATA/2020_11_04'
    CA_FILES = build_Ca_filelist(folder)

    print(CA_FILES)
