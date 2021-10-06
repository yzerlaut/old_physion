import os

python_path = 'python'
if os.path.isdir(os.path.join(os.path.expanduser('~'), 'miniconda3', 'envs', 'physion')):
    if os.name=='nt':
        python_path = os.path.join(os.path.expanduser('~'), 'miniconda3', 'envs', 'physion', 'python.exe')
    else:
        python_path = os.path.join(os.path.expanduser('~'), 'miniconda3', 'envs', 'physion', 'bin', 'python')
elif os.path.isdir(os.path.join(os.path.expanduser('~'), 'anaconda3', 'envs', 'physion')):
    if os.name=='nt':
        python_path = os.path.join(os.path.expanduser('~'), 'anaconda3', 'envs', 'physion', 'python.exe')
    else:
        python_path = os.path.join(os.path.expanduser('~'), 'anaconda3', 'envs', 'physion', 'bin', 'python')
        
python_path_suite2p_env = python_path
if os.path.isdir(os.path.join(os.path.expanduser('~'), 'miniconda3', 'envs', 'suite2p')):
    if os.name=='nt':
        python_path_suite2p_env = os.path.join(os.path.expanduser('~'), 'miniconda3', 'envs', 'suite2p', 'python.exe')
    else:
        python_path_suite2p_env = os.path.join(os.path.expanduser('~'), 'miniconda3', 'envs', 'suite2p', 'bin', 'python')
        
elif os.path.isdir(os.path.join(os.path.expanduser('~'), 'anaconda3', 'envs', 'suite2p')):
    if os.name=='nt':
        python_path_suite2p_env = os.path.join(os.path.expanduser('~'), 'anaconda3', 'envs', 'suite2p', 'python.exe')
    else:
        python_path_suite2p_env = os.path.join(os.path.expanduser('~'), 'anaconda3', 'envs', 'suite2p', 'bin', 'python')

    
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


    
