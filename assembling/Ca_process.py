import os, sys, pathlib, shutil, time
import numpy as np
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from hardware_control.Bruker.xml_parser import bruker_xml_parser
from assembling.saving import get_files_with_extension, list_dayfolder, check_datafolder, get_TSeries_folders

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
    
    CA_FILES = {'Bruker_folder':[], 'Bruker_file':[],
                'date':[], 'protocol':[],
                'StartTime':[], 'EndTime':[], 'absoluteTime':[]}
    for bdf in get_TSeries_folders(folder):
        fn = get_files_with_extension(bdf, extension='.xml')[0]
        try:
            xml = bruker_xml_parser(fn)
            if len(xml['Ch1']['relativeTime'])>0:
                CA_FILES['date'].append(stringdatetime_to_date(xml['date']))
                CA_FILES['Bruker_folder'].append(bdf)
                CA_FILES['Bruker_file'].append(fn)
                start = StartTime_to_day_seconds(xml['StartTime'])
                CA_FILES['StartTime'].append(start+xml['Ch1']['relativeTime'][0])
                CA_FILES['EndTime'].append(start+xml['Ch1']['relativeTime'][-1])
                CA_FILES['absoluteTime'].append(start+xml['Ch1']['absoluteTime'])
                CA_FILES['protocol'].append('')
        except BaseException as e:
            print(e)
            print(100*'-')
            print('Problem with file: "%s"' % fn)
            print(100*'-')

    return CA_FILES


def find_matching_data(PROTOCOL_LIST, CA_FILES,
                       min_protocol_duration=10, # seconds
                       verbose=True):

    for pfolder in PROTOCOL_LIST:
        metadata = np.load(os.path.join(pfolder, 'metadata.npy'), allow_pickle=True).item()
        if not 'true_tstart' in metadata:
            check_datafolder(pfolder)
            metadata = np.load(os.path.join(pfolder, 'metadata.npy'), allow_pickle=True).item()
        # time array for comparison
        times = np.arange(int(metadata['true_tstart']), int(metadata['true_tstop']))
        # insuring the good day
        day = pfolder.split(os.path.sep)[-2]
        day_cond = (np.array(CA_FILES['date'])==day)
        if len(times)>min_protocol_duration and (np.sum(day_cond)>0):
            # then we loop over Ca-imaging files to find the overlap
            for ica in np.arange(len(CA_FILES['StartTime']))[day_cond]:
                times2 = np.arange(int(CA_FILES['StartTime'][ica]), int(CA_FILES['EndTime'][ica]))

                if (len(np.intersect1d(times, times2))>min_protocol_duration) and verbose:
                    print('------------')
                    print(times[0], times[-1])
                    print(times2[0], times2[-1])
                    print(pfolder)
                    print(CA_FILES['absoluteTime'][ica][0])
                    print(CA_FILES['Bruker_folder'][ica])
                    CA_FILES['protocol'][ica] = pfolder


    return CA_FILES


SUITE2P_FILES = ['Fneu.npy',
                 'F.npy',
                 'iscell.npy',
                 'ops.npy',
                 'spks.npy',
                 'stat.npy']

def transfer_analyzed_data(CA_FILES):

    for ica in range(len(CA_FILES['StartTime'])):
        if CA_FILES['protocol'][ica]!='':
            new_folder = os.path.join(CA_FILES['protocol'][ica], 'Ca-imaging')
            pathlib.Path(new_folder).mkdir(parents=True, exist_ok=True)
            i=0
            folder = os.path.join(CA_FILES['Bruker_folder'][ica], 'suite2p', 'plane%i' % i)
            while os.path.isdir(folder):
                for f in SUITE2P_FILES:
                    if os.path.isfile(os.path.join(folder,f)):
                        shutil.copyfile(os.path.join(folder,f), os.path.join(new_folder, f))
                print('succesfully copied: %s to %s' % (os.path.join(folder,f), os.path.join(new_folder,f)))
                # adding the time array
                np.save(os.path.join(new_folder,'times.npy'), np.array(CA_FILES['absoluteTime'][ica]))
                # loop increment
                i+=1
                folder = os.path.join(CA_FILES['Bruker_folder'][ica], 'suite2p', 'plane%i' % i)
                


if __name__=='__main__':

    import argparse, os
    parser=argparse.ArgumentParser(description="transfer interface",
                       formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-rfCa', "--root_datafolder_Calcium", type=str,
                        default=os.path.join(os.path.expanduser('~'), 'DATA'))
    parser.add_argument('-rfVis', "--root_datafolder_Visual", type=str,
                        default=os.path.join(os.path.expanduser('~'), 'DATA'))
    parser.add_argument('-d', "--day", type=str,
                        default='')
    parser.add_argument('-wt', "--with_transfer", action="store_true")
    parser.add_argument('-v', "--verbose", action="store_true")
    args = parser.parse_args()

    if args.day!='':
        ca_folder = os.path.join(args.root_datafolder_Calcium, args.day)
        vis_folder = os.path.join(args.root_datafolder_Visual, args.day)
    else:
        ca_folder = args.root_datafolder_Calcium
        vis_folder = args.root_datafolder_Visual
        
    CA_FILES = build_Ca_filelist(ca_folder)
    
    if args.day!='':
        PROTOCOL_LIST = list_dayfolder(os.path.join(vis_folder, args.day))
    else: # loop over days
        PROTOCOL_LIST = []
        for day in os.listdir(vis_folder):
            PROTOCOL_LIST += list_dayfolder(os.path.join(vis_folder, day))
        print(PROTOCOL_LIST)
    CA_FILES = find_matching_data(PROTOCOL_LIST, CA_FILES,
                                  verbose=args.verbose)

    # if args.with_transfer:
    #     transfer_analyzed_data(CA_FILES)
