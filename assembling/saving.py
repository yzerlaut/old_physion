import datetime, os
from pathlib import Path

def create_day_folder(root_folder):

    Path(os.path.join(root_folder,
                      datetime.datetime.now().strftime("%Y_%m_%d"))).mkdir(parents=True, exist_ok=True)
    
def generate_filename_path(root_folder,
                           extension='txt',
                           with_microseconds=False):

    day_folder = os.path.join(root_folder,
                              datetime.datetime.now().strftime("%Y_%m_%d"))
    if not os.path.exists(day_folder):
        print('creating the folder "%s"', day_folder)
        create_day_folder(day_folder)

    if not extension.startswith('.'):
        extension='.'+extension
        
    return os.path.join(day_folder, datetime.datetime.now().strftime("%H:%M:%S")+extension)
    

if __name__=='__main__':


    
    # print(filename_with_datetime('', folder='./', extension='.npy'))
    # print(filename_with_datetime('', folder='./', extension='npy'))
    
    create_day_folder('/home/yann/DATA/')
    fn = generate_filename_path('/home/yann/DATA/', 'visual-stim.npz')
    print(fn)
