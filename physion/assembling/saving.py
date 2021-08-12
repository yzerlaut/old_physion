import datetime, os, string, pathlib, json, tempfile
import numpy as np

def day_folder(root_folder):
    return os.path.join(root_folder, datetime.datetime.now().strftime("%Y_%m_%d"))

def second_folder(day_folder):
    return os.path.join(day_folder, datetime.datetime.now().strftime("%H-%M-%S"))

def create_day_folder(root_folder):
    df = day_folder(root_folder)
    pathlib.Path(df).mkdir(parents=True, exist_ok=True)
    return day_folder(root_folder)

def create_second_folder(day_folder):
    pathlib.Path(second_folder(day_folder)).mkdir(parents=True, exist_ok=True)
    
def generate_filename_path(root_folder,
                           filename = '', extension='txt',
                           with_screen_frames_folder=False,
                           with_FaceCamera_frames_folder=False,
                           with_microseconds=False):

    Day_folder = day_folder(root_folder)
    Second_folder = second_folder(Day_folder)
    
    if not os.path.exists(Day_folder):
        print('creating the folder "%s"' % Day_folder)
        pathlib.Path(Day_folder).mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(Second_folder):
        print('creating the folder "%s"' % Second_folder)
        pathlib.Path(Second_folder).mkdir(parents=True, exist_ok=True)

    if with_screen_frames_folder:
        pathlib.Path(os.path.join(Second_folder, 'screen-frames')).mkdir(parents=True, exist_ok=True)

    if with_FaceCamera_frames_folder:
        pathlib.Path(os.path.join(Second_folder, 'FaceCamera-imgs')).mkdir(parents=True, exist_ok=True)
        
    if not extension.startswith('.'):
        extension='.'+extension
    
    return os.path.join(Second_folder, filename+extension)


def list_dayfolder(day_folder):
    folders = [os.path.join(day_folder, d) for d in sorted(os.listdir(day_folder)) if ((d[0] in string.digits) and (len(d)==8) and os.path.isdir(os.path.join(day_folder, d)) and os.path.isfile(os.path.join(day_folder, d, 'metadata.npy')) and os.path.isfile(os.path.join(day_folder, d, 'NIdaq.npy')) and os.path.isfile(os.path.join(day_folder, d, 'NIdaq.start.npy')))]
    return folders


def last_datafolder_in_dayfolder(day_folder):
    
    folders = list_dayfolder(day_folder)

    if folders[-1][-1] in string.digits:
        return folders[-1]
    else:
        print('No datafolder found, returning "./" ')
        return './'


def get_files_with_extension(folder, extension='.txt',
                             recursive=False):
    FILES = []
    if recursive:
        for root, dirs, files in os.walk(folder):
            for f in files:
                if f.endswith(extension) and ('$RECYCLE.BIN' not in root):
                    FILES.append(os.path.join(root, f))
    else:
        for f in os.listdir(folder):
            if not type(f) is str:
                f = f.decode('ascii')
            if f.endswith(extension) and ('$RECYCLE.BIN' not in folder):
                FILES.append(os.path.join(folder, f))
    return FILES


def get_files_with_given_exts(dir='./', EXTS=['npz','abf','bin']):
    """  DEPRECATED, use the function above !!"""
    FILES = []
    for ext in EXTS:
        for file in os.listdir(dir):
            if file.endswith(ext):
                FILES.append(os.path.join(dir, file))
    return np.array(FILES)


def get_TSeries_folders(folder, frame_limit=-1, limit_to_subdirectories=False):
    
    """ get files of a given extension and sort them..."""
    FOLDERS = []
    if limit_to_subdirectories:
        FOLDERS = [f for f in next(os.walk(folder))[1] if ('TSeries' in str(f)) and (len(os.listdir(f))>frame_limit)]
    else:
        for root, subdirs, files in os.walk(folder):
            if 'TSeries' in root.split(os.path.sep)[-1] and len(files)>frame_limit:
                FOLDERS.append(os.path.join(folder, root))
            elif 'TSeries' in root.split(os.path.sep)[-1]:
                print('"%s" ignored' % root)
                print('   ----> data should be at least %i frames !' % frame_limit)
    return np.array(FOLDERS)

def insure_ordered_frame_names(df):
    # insuring nice order of screen frames
    filenames = os.listdir(os.path.join(df,'screen-frames'))
    if len(filenames)>0:
        nmax = np.max(np.array([len(fn) for fn in filenames]))
        for fn in filenames:
            n0 = len(fn)
            if n0<nmax:
                os.rename(os.path.join(df,'screen-frames', fn),
                          os.path.join(df,'screen-frames', fn.replace('frame', 'frame'+'0'*(nmax-n0))))

def insure_ordered_FaceCamera_picture_names(df):
    # insuring nice order of screen frames
    filenames = os.listdir(os.path.join(df,'FaceCamera-imgs'))
    if len(filenames)>0:
        nmax = np.max(np.array([len(fn) for fn in filenames]))
        for fn in filenames:
            n0 = len(fn)
            if n0<nmax:
                os.rename(os.path.join(df,'FaceCamera-imgs',fn),
                          os.path.join(df,'FaceCamera-imgs','0'*(nmax-n0)+fn))
                

def from_folder_to_datetime(folder):

    s = folder.split(os.path.sep)[-2:]

    try:
        date = s[0].split('_')
        return date[2]+'/'+date[1]+'/'+date[0], s[1].replace('-', ':')
    except Exception:
        return '', folder

def folderName_to_daySeconds(datafolder):

    Hour = int(datafolder.split('-')[0][-2:])
    Min = int(datafolder.split('-')[1])
    Seconds = int(datafolder.split('-')[2][:2])

    return 60.*60.*Hour+60.*Min+Seconds
            
def computerTimestamp_to_daySeconds(t):

    s = str(datetime.timedelta(seconds=t))

    Hour = int(s.split(':')[0][-2:])
    Min = int(s.split(':')[1])
    Seconds = float(s.split(':')[2])
    
    return 60*60*Hour+60*Min+Seconds
    

def check_datafolder(df,
                     modalities=['Screen', 'Locomotion', 'Electrophy', 'Pupil', 'Calcium'],
                     verbose=True):

    if verbose:
        print('---> Checking the integrity of the datafolder [...] ')
        
    # should always be there
    print(os.path.join(df,'metadata.npy'))
    if os.path.isfile(os.path.join(df,'metadata.npy')):
        metadata = np.load(os.path.join(df,'metadata.npy'),
                           allow_pickle=True).item()

        if os.path.isfile(os.path.join(df, 'FaceCamera-times.npy')) and \
           os.path.isdir(os.path.join(df,'FaceCamera-imgs')):
            metadata['FaceCamera'] = True
        elif os.path.isfile(os.path.join(df, 'FaceCamera-times.npy')) and \
           os.path.isdir(os.path.join(df,'FaceCamera-compressed')):
            metadata['FaceCamera'] = True
        else:
            metadata['FaceCamera'] = False

        if os.path.isfile(os.path.join(df, 'NIdaq.npy')):
            metadata['NIdaq'] = True
            # dealing with TimeStamps !!
            if 'true_duration' not in metadata:
                try:
                    print('adding true "tstart" and "tstop" to metadata [...]')
                    tstart = np.load(os.path.join(df, 'NIdaq.start.npy'))[0]
                    tstart = dealWithVariableTimestamps(df, tstart)
                    metadata['true_tstart'] = tstart
                    data = np.load(os.path.join(df, 'NIdaq.npy'), allow_pickle=True).item()
                    duration = len(data['analog'][0,:])/metadata['NIdaq-acquisition-frequency']
                    metadata['true_duration'] = duration
                    metadata['true_tstop'] = tstart+duration
                    np.save(os.path.join(df,'metadata.npy'), metadata)
                    print('[ok] True duration= %.1fs' % metadata['true_duration'])
                except BaseException as e:
                    print('e')
                    print('\n'+100*'-'+'True start/stop undertermined for', df)
            else:
                dealWithVariableTimestamps(df, metadata['true_tstart'], verbose=verbose)
                print('True tstop= %.1fs' % metadata['true_tstop'])
                    
        else:
            metadata['NIdaq'] = False

        if os.path.isfile(os.path.join(df, 'visual-stim.npy')):
            metadata['VisualStim'] = True
            data = np.load(os.path.join(df, 'visual-stim.npy'), allow_pickle=True).item()
            for key, val in data.items():
                metadata[key] = val
        else:
            metadata['VisualStim'] = False

        if metadata['FaceCamera'] and os.path.isdir(os.path.join(df,'FaceCamera-imgs')):
            # insuring nice order of FaceCamera images
            filenames = os.listdir(os.path.join(df,'FaceCamera-imgs'))
            if len(filenames)>0:
                nmax = np.max(np.array([len(fn) for fn in filenames]))
                for fn in filenames:
                    n0 = len(fn)
                    if n0<nmax:
                        os.rename(os.path.join(df,'FaceCamera-imgs',fn),
                                  os.path.join(df,'FaceCamera-imgs','0'*(nmax-n0)+fn))

        if metadata['FaceCamera'] and os.path.isdir(os.path.join(df,'FaceCamera-compressed')):
            # insuring nice order of FaceCamera images
            filenames = os.listdir(os.path.join(df,'FaceCamera-compressed'))
            filenames.remove('metadata.npy')
            nmax1 = max([len(fn.split('imgs-')[1].split('.')[0].split('-')[0]) for fn in filenames])
            nmax2 = max([len(fn.split('imgs-')[1].split('.')[0].split('-')[1]) for fn in filenames])
            for fn in filenames[:-1]:
                n1 = fn.split('imgs-')[1].split('.')[0].split('-')[0]
                n2 = fn.split('imgs-')[1].split('.')[0].split('-')[1]
                if (len(n1)<nmax1) or (len(n2)<nmax2):
                    fn_full = os.path.join(df,'FaceCamera-compressed',fn)
                    os.rename(fn_full,
                              fn_full.replace('-'+n1+'-', '-'+'0'*(nmax1-len(n1))+n1+'-').replace('-'+n2+'.', '-'+'0'*(nmax2-len(n2))+n2+'.'))


        if metadata['VisualStim']:
            # insuring nice order of screen frames
            filenames = os.listdir(os.path.join(df,'screen-frames'))
            if len(filenames)>0:
                nmax = np.max(np.array([len(fn) for fn in filenames]))
                for fn in filenames:
                    n0 = len(fn)
                    if n0<nmax:
                        os.rename(os.path.join(df,'screen-frames', fn),
                                  os.path.join(df,'screen-frames', fn.replace('frame', 'frame'+'0'*(nmax-n0))))

        if verbose:
            print('[ok] datafolder checked !')
            
        return metadata
    else:
        print('Metadata file missing for "%s" ' % df)
        return {}
            
#########################################################
#### Dealing with root data folder
#########################################################

DFFN = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'master', 'data-folder.json') # DATA-FOLDER-FILENAME
    
def get_data_folder(root_datafolder):
    # if not existing we create the data-folder with tempdir
    if not os.path.isfile(DFFN):
        with open(DFFN, 'w') as fp:
            json.dump({"folder":'"%s"' % tempfile.gettempdir()}, fp)
    # now we can load
    with open(DFFN, 'r') as fp:
        data_folder = json.load(fp)['folder']
    # if not existing we re-write to temp folder
    if not os.path.isdir(data_folder): 
        with open(DFFN, 'w') as fp:
            json.dump({"folder":'"%s"' % tempfile.gettempdir()}, fp)
        data_folder = tempfile.gettempdir()
    return data_folder

def set_data_folder(df):
    with open(DFFN, 'w') as fp:
        json.dump({"folder":df}, fp)

#########################################################
#### NPZ files
#########################################################

def save_dict(filename, data):

    if '.npz' not in filename:
        print('/!\ The filename need to have the "npz" extension')
        print('        ------- renamed to:', filename+'.npz')
        np.savez(filename+'.npz', **data)
    else:
        np.savez(filename, **data)

def load_dict(filename):

    data = np.load(filename, allow_pickle=True)
    output = {}
    for key in data.files:
        output[key] = data[key]

        if type(output[key]) is np.ndarray:
            try:
                output[key] = output[key].item()
            except ValueError:
                pass
        
    return output


if __name__=='__main__':

    # print(list_dayfolder('/home/yann/DATA/2020_11_03'))
    # import tempfile
    # data_folder = tempfile.gettempdir()
    # print(last_datafolder_in_dayfolder(day_folder(data_folder)))
    # print(filename_with_datetime('', folder='./', extension='.npy'))
    # print(filename_with_datetime('', folder='./', extension='npy'))
    
    # fn = generate_filename_path('/home/yann/DATA/', 'visual-stim.npz')
    # print(fn)
    
    import argparse, os
    parser=argparse.ArgumentParser(description="transfer interface",
                       formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-rf', "--root_datafolder", type=str,
                        default=os.path.join(os.path.expanduser('~'), 'DATA'))
    parser.add_argument('-d', "--day", type=str,
                        default='2020_11_03')
    parser.add_argument('-wt', "--with_transfer", action="store_true")
    parser.add_argument('-v', "--verbose", action="store_true")
    args = parser.parse_args()

    for pfolder in list_dayfolder(os.path.join(args.root_datafolder, args.day)):
        check_datafolder(pfolder)
        
