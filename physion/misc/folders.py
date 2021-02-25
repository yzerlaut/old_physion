import os

FOLDERS = {
    'home':os.path.join(os.path.expanduser('~'), 'DATA')
}

if os.name=='nt':
    FOLDERS['drive'] = 'F:\\'
else:
    FOLDERS['drive'] = '/media/yann/Yann/'
    FOLDERS['storage'] = '/media/yann/DATADRIVE1/DATA/'
    FOLDERS['MsWin-data'] = '/media/yann/Windows/home/yann/DATA/'
    FOLDERS['MsWin-cygwin'] = '/media/yann/Windows/Users/yann.zerlaut/DATA/'
