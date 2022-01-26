import os

python_path = 'python'


possible_conda_dir_lists = [os.path.join(os.path.expanduser('~'), 'miniconda3'),
                            os.path.join(os.path.expanduser('~'), 'anaconda3'),
                            os.path.join(os.path.expanduser('~'), '.conda'),
                            os.path.join(os.path.expanduser('~'), 'appdata', 'continuum', 'anaconda3')]
                       
def check_path(env='physion'):
    i, success, path = 0, False, python_path
    while (not success) and (i<len(possible_conda_dir_lists)):
        new_path = os.path.join(possible_conda_dir_lists[i], 'envs', env)
        if os.path.isdir(new_path):
            success = True
            if (os.name=='nt'):
                path = os.path.join(new_path, 'python.exe')
            else:
                path = os.path.join(new_path, 'bin', 'python')
        i+=1
    return path

if (os.name=='nt') and os.path.isdir(os.path.join(os.path.expanduser('~'), '.conda', 'envs', 'acquisition')):
    print('acq setting')
    python_path = os.path.join(os.path.expanduser('~'), '.conda', 'envs', 'acquisition', 'python.exe')
else:
    python_path = check_path('physion')

        
python_path_suite2p_env = check_path('suite2p')
print(python_path)
print(python_path_suite2p_env)

    
FOLDERS = {
    '~/DATA':os.path.join(os.path.expanduser('~'), 'DATA'),
    '~/UNPROCESSED':os.path.join(os.path.expanduser('~'), 'UNPROCESSED'),
    '~/CURATED':os.path.join(os.path.expanduser('~'), 'CURATED')
}

if os.name=='nt':
    FOLDERS['D-drive'] = 'D:\\'
    FOLDERS['E-drive'] = 'E:\\'
    FOLDERS['F-drive'] = 'F:\\'
    FOLDERS['G-drive'] = 'G:\\'
else:
    FOLDERS['storage-curated'] = '/media/yann/DATADRIVE1/CURATED/'
    FOLDERS['storage-DATA'] = '/media/yann/DATADRIVE1/DATA/'
    FOLDERS['10.0.0.1:curated'] = 'yann@10.0.0.1:/media/yann/DATADRIVE1/CURATED'
    FOLDERS['MsWin-DATA'] = '/media/yann/Windows/Users/yann.zerlaut/DATA'
    FOLDERS['usb (YANN)'] = '/media/yann/YANN/'
    FOLDERS['usb (Yann)'] = '/media/yann/Yann/'
    # FOLDERS['MsWin-data'] = '/media/yann/Windows/home/yann/DATA/'
    # FOLDERS['MsWin-cygwin'] = '/media/yann/Windows/Users/yann.zerlaut/DATA/'
    FOLDERS['usb (code)'] = '/media/yann/CODE_YANN/'
    FOLDERS['10.0.0.1:~/DATA'] = 'yann@10.0.0.1:/home/yann/DATA/'
    FOLDERS['10.0.0.2:~/DATA'] = 'yann@10.0.0.2:/home/yann/DATA/'


    
