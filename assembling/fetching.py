import numpy as np
import os, sys, pathlib
from PIL import Image

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import day_folder, create_day_folder, generate_filename_path, load_dict

def get_list_of_datafiles(data_folder):

    list_of_folders = [os.path.join(day_folder(data_folder), d)\
                       for d in os.listdir(day_folder(data_folder)) if os.path.isdir(os.path.join(day_folder(data_folder), d))]
    
    return [os.path.join(d,'visual-stim.npz')\
              for d in list_of_folders if os.path.isfile(os.path.join(d,'visual-stim.npz'))]

def get_multimodal_dataset(filename):

    # -- VISUAL STIMULATION -- #
    # presentation metadata
    data = load_dict(filename)
    # presented images (snapshot of last frame)
    data['images'] = []
    i=1
    while os.path.isfile(filename.replace('visual-stim.npz', 'frame%i.tiff' %i)):
        im = Image.open(filename.replace('visual-stim.npz', 'frame%i.tiff' %i))
        data['images'].append(np.array(im).mean(axis=2).T)
        i+=1

    # -- NI DAQ ACQUISTION -- #
    if os.path.isfile(filename.replace('visual-stim.npz', 'NIdaq.npy')):
        data['NIdaq'] = np.load(filename.replace('visual-stim.npz', 'NIdaq.npy'))

    # -- PTGREY CAMERA -- #

    # -- PTGREY CAMERA -- #
    
    return data


