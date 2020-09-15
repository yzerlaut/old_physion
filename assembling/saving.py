import datetime, os, string, pathlib, json, tempfile
import numpy as np

def day_folder(root_folder):
    return os.path.join(root_folder, datetime.datetime.now().strftime("%Y_%m_%d"))

def second_folder(day_folder):
    return os.path.join(day_folder, datetime.datetime.now().strftime("%H-%M-%S"))

def create_day_folder(root_folder):
    pathlib.Path(day_folder(root_folder)).mkdir(parents=True, exist_ok=True)

def create_second_folder(day_folder):
    pathlib.Path(second_folder(day_folder)).mkdir(parents=True, exist_ok=True)
    
def generate_filename_path(root_folder,
                           filename = '', extension='txt',
                           with_screen_frames_folder=False,
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
        
    if not extension.startswith('.'):
        extension='.'+extension
    
    return os.path.join(Second_folder, filename+extension)

def list_dayfolder(day_folder):
    folders = [os.path.join(day_folder, d) for d in sorted(os.listdir(day_folder)) if ((d[0] in string.digits) and os.path.isdir(os.path.join(day_folder, d)))]
    return folders
    
def last_datafolder_in_dayfolder(day_folder):
    
    folders = list_dayfolder(day_folder)

    if folders[-1][-1] in string.digits:
        return folders[-1]
    else:
        print('No datafolder found, returning "./" ')
        return './'

def from_folder_to_datetime(folder):

    s = folder.split(os.path.sep)[-2:]

    date = s[0].split('_')
    return date[2]+'/'+date[1]+'/'+date[0], s[1].replace('-', ':')

def check_datafolder(df):
    
    check = {}

    if os.path.isfile(os.path.join(df, 'FaceCamera-times.npy')) and \
       os.path.isdir(os.path.join(df,'FaceCamera-imgs')):
        check['FaceCamera'] = True
    else:
        check['FaceCamera'] = False

    if os.path.isfile(os.path.join(df, 'NIdaq.npy')):
        check['NIdaq'] = True
    else:
        check['NIdaq'] = False

    if os.path.isfile(os.path.join(df, 'visual-stim.npz')) and \
       os.path.isdir(os.path.join(df,'screen-frames')):
        check['visual-stim'] = True
    else:
        check['visual-stim'] = False

        
    if check['FaceCamera']:
        # insuring nice order of FaceCamera images
        filenames = os.listdir(os.path.join(df,'FaceCamera-imgs'))
        nmax = max([len(fn) for fn in filenames])
        for fn in filenames:
            n0 = len(fn)
            if n0<nmax:
                os.rename(os.path.join(df,'FaceCamera-imgs',fn),
                          os.path.join(df,'FaceCamera-imgs','0'*(nmax-n0)+fn))
                

    if check['visual-stim']:
        # insuring nice order of screen frames
        filenames = os.listdir(os.path.join(df,'screen-frames'))
        nmax = max([len(fn) for fn in filenames])
        for fn in filenames:
            n0 = len(fn)
            if n0<nmax:
                os.rename(os.path.join(df,'screen-frames', fn),
                          os.path.join(df,'screen-frames', fn.replace('frame', 'frame'+'0'*(nmax-n0))))

    return check
            
#########################################################
#### Dealing with root data folder
#########################################################

DFFN = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'master', 'data-folder.json') # DATA-FOLDER-FILENAME
    
def get_data_folder():
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

    import tempfile
    data_folder = tempfile.gettempdir()
    print(last_datafolder_in_dayfolder(day_folder(data_folder)))
    # print(filename_with_datetime('', folder='./', extension='.npy'))
    # print(filename_with_datetime('', folder='./', extension='npy'))
    
    # fn = generate_filename_path('/home/yann/DATA/', 'visual-stim.npz')
    # print(fn)
