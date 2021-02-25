import os, sys, pathlib, time, datetime
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from physion.assembling.IO.bruker_xml_parser import bruker_xml_parser
from physion.assembling.saving import get_files_with_extension, get_TSeries_folders
from physion.behavioral_monitoring.locomotion import compute_position_from_binary_signals
from physion.analysis.read_NWB import read as read_NWB


def compute_locomotion(binary_signal, acq_freq=1e4,
                       speed_smoothing=10e-3, # s
                       t0=0):

    A = binary_signal%2
    B = np.round(binary_signal/2, 0)

    return compute_position_from_binary_signals(A, B,
                                                smoothing=int(speed_smoothing*acq_freq))


def build_subsampling_from_freq(subsampled_freq=1.,
                                original_freq=1.,
                                N=10, Nmin=3):
    """

    """
    if original_freq==0:
        print('  /!\ problem with original sampling freq /!\ ')
        
    if subsampled_freq==0:
        SUBSAMPLING = np.linspace(0, N-1, Nmin).astype(np.int)
    elif subsampled_freq>=original_freq:
        SUBSAMPLING = np.arange(0, N) # meaning all samples !
    else:
        SUBSAMPLING = np.arange(0, N, max([int(subsampled_freq/original_freq),Nmin]))

    return SUBSAMPLING


def load_FaceCamera_data(imgfolder, t0=0, verbose=True):

    times = np.array([float(f.replace('.npy', '')) for f in os.listdir(imgfolder) if f.endswith('.npy')])
    times = times[np.argsort(times)]-t0
    FILES = np.array([f for f in os.listdir(imgfolder) if f.endswith('.npy')], dtype=str)[np.argsort(times)]
    nframes = len(times)
    Lx, Ly = np.load(os.path.join(imgfolder, FILES[0])).shape
    if verbose:
        print('Sampling frequency: %.1f Hz' % (1./np.diff(times).mean()))
    return times, FILES, nframes, Lx, Ly


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
                'date':[], 'protocol':[],'StartTimeString':[],
                'StartTime':[], 'EndTime':[], 'absoluteTime':[]}
    
    for bdf in get_TSeries_folders(folder):
        fn = get_files_with_extension(bdf, extension='.xml')[0]
        try:
            xml = bruker_xml_parser(fn)
            if len(xml['Ch1']['relativeTime'])>0:
                CA_FILES['date'].append(stringdatetime_to_date(xml['date']))
                CA_FILES['Bruker_folder'].append(bdf)
                CA_FILES['Bruker_file'].append(fn)
                CA_FILES['StartTimeString'].append(xml['StartTime'])
                start = StartTime_to_day_seconds(xml['StartTime'])
                CA_FILES['StartTime'].append(start+xml['Ch1']['absoluteTime'][0])
                CA_FILES['EndTime'].append(start+xml['Ch1']['absoluteTime'][-1])
                CA_FILES['protocol'].append('')
        except BaseException as e:
            print(e)
            print(100*'-')
            print('Problem with file: "%s"' % fn)
            print(100*'-')

    return CA_FILES

def find_matching_CaImaging_data(cls, filename, CaImaging_root_folder,
                                 min_protocol_duration=10, # seconds
                                 verbose=True):

    success, folder = False, ''
    CA_FILES = build_Ca_filelist(CaImaging_root_folder)

    read_NWB(cls, filename)
    Tstart = cls.metadata['NIdaq_Tstart']
    st = datetime.datetime.fromtimestamp(Tstart).strftime('%H:%M:%S.%f')
    true_tstart = StartTime_to_day_seconds(st)
    true_duration = cls.tlim[1]-cls.tlim[0]
    true_tstop = true_tstart+true_duration
    times = np.arange(int(true_tstart), int(true_tstop))
    
    day = datetime.datetime.fromtimestamp(Tstart).strftime('%Y_%m_%d')
    print(day)
    # first insuring the good day in the CA FOLDERS
    day_cond = (np.array(CA_FILES['date'])==day)
    if len(times)>min_protocol_duration and (np.sum(day_cond)>0):
        # then we loop over Ca-imaging files to find the overlap
        for ica in np.arange(len(CA_FILES['StartTime']))[day_cond]:
            times2 = np.arange(int(CA_FILES['StartTime'][ica]),
                               int(CA_FILES['EndTime'][ica]))
            if (len(np.intersect1d(times, times2))>min_protocol_duration):
                success, folder = True, CA_FILES['Bruker_folder'][ica]
                percent_overlap = 100.*len(np.intersect1d(times, times2))/len(times)
                print(50*'-')
                print(' => matched to %s with %.1f %% overlap' % (folder,
                                                                  percent_overlap))
                print(50*'-')
    cls.io.close()
    return success, folder

class nothing:
    def __init__(self):
        self.name = 'nothing'
        
if __name__=='__main__':

    fn = '/media/yann/Yann/2021_02_16/15-41-13/2021_02_16-15-41-13.nwb'
    CA_FOLDER = '/home/yann/DATA/'
    cls = nothing()
    success, folder = find_matching_CaImaging_data(cls, fn, CA_FOLDER)

    # times, FILES, nframes, Lx, Ly = load_FaceCamera_data(folder+'FaceCamera-imgs')
    # Tstart = np.load(folder+'NIdaq.start.npy')[0]










