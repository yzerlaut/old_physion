import os

FOLDERS = {
    'processed':os.path.join(os.path.expanduser('~'), 'DATA', 'PROCESSED'),
    'data':os.path.join(os.path.expanduser('~'), 'DATA')
}

if os.name=='nt':
    FOLDERS['usb-drive'] = 'F:\\'
else:
    FOLDERS['usb-drive'] = '/media/yann/Yann/'
    FOLDERS['desktop-storage'] = '/media/yann/DATADRIVE1/DATA/'
    FOLDERS['MsWin-data'] = '/media/yann/Windows/home/yann/DATA/'
    FOLDERS['MsWin-cygwin'] = '/media/yann/Windows/Users/yann.zerlaut/DATA/'
