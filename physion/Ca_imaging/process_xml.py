import sys, os, pathlib, shutil, glob, time
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.IO.bruker_xml_parser import bruker_xml_parser
from Ca_imaging.presets import ops0


def build_db(folder):
    print(folder)
    db = {'data_path':[folder],
          'subfolders': [],
          'save_path0': folder,
          'fast_disk': folder,
          'input_format': 'bruker'}
    return db

def build_ops(folder):
    return ops

def build_suite2p_options(folder,
                          settings_dict):
    
    if os.name=='nt':
        xml_file = os.path.join(folder, folder.split('/')[-1]+'.xml')
    else:
        xml_file = os.path.join(folder, folder.split(os.path.sep)[-1]+'.xml')

    bruker_data = bruker_xml_parser(xml_file)
    ops = ops0.copy()

    # acquisition frequency
    ops['fs'] = 1./float(bruker_data['settings']['framePeriod'])

    # hints for the size of the ROI
    um_per_pixel = float(bruker_data['settings']['micronsPerPixel']['XAxis'])
    ops['diameter'] = int(settings_dict['cell_diameter']/um_per_pixel) # in pixels (int 20um)
    ops['spatial_scale'] = int(settings_dict['cell_diameter']/6/um_per_pixel)

    # ops['tiff_list'] = [] 
    
    # all other keys here
    for key in settings_dict:
        if key in ops:
            ops[key] = settings_dict[key]
    
    db = build_db(folder)
    for key in ['data_path', 'subfolders', 'save_path0',
                'fast_disk', 'input_format']:
        ops[key] = db[key]

    np.save(os.path.join(folder,'db.npy'), db)
    np.save(os.path.join(folder,'ops.npy'), ops)

    

if __name__=='__main__':
    
    folder = sys.argv[-1] # '/media/yann/Yann/2020_11_10/TSeries-11102020-1605-016'
    
    xml_file = os.path.join(folder, os.path.join(folder.split('/')[-1]+'.xml'))
    
    bruker_data = bruker_xml_parser(xml_file)
    # print(freq)
    # print(fn.split(os.path.sep))

