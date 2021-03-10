import os

FOLDERS = {
    'data':os.path.join(os.path.expanduser('~'), 'DATA'),
    'unprocessed':os.path.join(os.path.expanduser('~'), 'UNPROCESSED')
}

if os.name=='nt':
    FOLDERS['D-drive'] = 'D:\\'
    FOLDERS['E-drive'] = 'E:\\'
    FOLDERS['F-drive'] = 'F:\\'
    FOLDERS['G-drive'] = 'G:\\'
else:
    FOLDERS['usb-drive (Yann)'] = '/media/yann/Yann/'
    FOLDERS['desktop-storage'] = '/media/yann/DATADRIVE1/DATA/'
    # FOLDERS['MsWin-data'] = '/media/yann/Windows/home/yann/DATA/'
    # FOLDERS['MsWin-cygwin'] = '/media/yann/Windows/Users/yann.zerlaut/DATA/'
    FOLDERS['usb-drive (code)'] = '/media/yann/CODE_YANN/'
