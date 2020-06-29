import numpy as np
import os, sys, pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import day_folder, create_day_folder, generate_filename_path, load_dict

def get_list_of_datafiles(parent):

    list_of_folders = [os.path.join(day_folder(parent.data_folder), d)\
                       for d in os.listdir(day_folder(parent.data_folder)) if os.path.isdir(os.path.join(day_folder(parent.data_folder), d))]
    
    return [os.path.join(d,'visual-stim.npz')\
              for d in list_of_folders if os.path.isfile(os.path.join(d,'visual-stim.npz'))]

def get_multimodal_dataset(filename):

    data = load_dict(filename)
    # NIdaq files
    for i in ['start', 'end']:
        print(filename.replace('visual-stim.npz', 'NIdaq.%s.npy' % i))
        if os.path.isfile(filename.replace('visual-stim.npz', 'NIdaq.%s.npy' % i)):
            data['NIdaq-%s' % i] = np.load(filename.replace('visual-stim.npz', 'NIdaq.%s.npy' % i))
    i=0
    while os.path.isfile(filename.replace('visual-stim.npz', 'NIdaq.%s.npy' % i)):
        print(filename.replace('visual-stim.npz', 'NIdaq.%s.npy' % i))
        data['NIdaq-%s' % i] = np.load(filename.replace('visual-stim.npz', 'NIdaq.%s.npy' % i))
        i+=1

    return data


