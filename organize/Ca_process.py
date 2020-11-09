import os, sys, pathlib
import numpy as np
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from hardware_control.Bruker.xml_parser import bruker_xml_parser
from assembling.saving import get_files_with_extension, list_dayfolder

def list_TSeries_folder(folder):
    folders = [os.path.join(folder, d) for d in sorted(os.listdir(folder)) if ((d[:7]=='TSeries') and os.path.isdir(os.path.join(folder, d)))]
    return folders

def stringdatetime_to_date(s):

    Month, Day, Year = s.split('/')[0], s.split('/')[1], s.split('/')[2][:4]

    if len(Month)==1:
        Month = '0'+Month
    if len(Day)==1:
        Day = '0'+Day

    return '%s_%s_%s' % (Year, Month, Day)

def stringdatetime_to_time(s):

    Hour, Min, Seconds = int(s.split(':')[0][-2:]), int(s.split(':')[1]), int(s.split(':')[2][:2])
    if 'PM' in s:
        Hour += 12
    
    return '%s-%s-%s' % (Hour, Min, Seconds)

def StartTime_to_day_seconds(StartTime):

    Hour = int(StartTime[0:2])
    Min = int(StartTime[3:5])
    Seconds = float(StartTime[6:])

    return 60*60*Hour+60*Min+Seconds

def build_Ca_filelist(folder):
    
    CA_FILES = {'Bruker_folder':[], 'Bruker_file':[], 'date':[],
                'StartTime':[], 'EndTime':[], 'absoluteTime':[]}
    for bdf in list_TSeries_folder(folder):
        fn = get_files_with_extension(bdf, extension='.xml')[0]
        xml = bruker_xml_parser(fn)
        print(fn)
        if len(xml['Ch1']['relativeTime'])>0:
            CA_FILES['date'].append(stringdatetime_to_date(xml['date'])
            CA_FILES['datetime'].append(xml['date'])
            CA_FILES['Bruker_folder'].append(bdf)
            CA_FILES['Bruker_file'].append(fn)
            start = StartTime_to_day_seconds(xml['StartTime'])
            CA_FILES['StartTime'].append(start)
            CA_FILES['EndTime'].append(start+xml['Ch1']['relativeTime'][-1])
            CA_FILES['absoluteTime'].append(xml['Ch1']['absoluteTime'])

    return CA_FILES

if __name__=='__main__':

    folder = '/home/yann/DATA/2020_11_03'
    CA_FILES = build_Ca_filelist(folder)

    """
    PROTOCOL_LIST = list_dayfolder(folder)

    import datetime
    for pfolder in PROTOCOL_LIST:
        metadata = np.load(os.path.join(pfolder, 'metadata.npy'), allow_pickle=True).item()
        try:
            tstart = np.load(os.path.join(pfolder, 'NIdaq.start.npy'))[0]
            print(pfolder)
            print(datetime.timedelta(seconds=tstart))
        except FileNotFoundError:
            pass
        # print(metadata.keys())
    for i in range(len(CA_FILES['absoluteTime'])):
        print(CA_FILES['datetime'][i])


    """
