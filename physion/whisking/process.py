import sys, os, pathlib, time
import numpy as np
from scipy.interpolate import interp1d

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from misc.progressBar import printProgressBar
from assembling.dataset import Dataset
from assembling.tools import load_FaceCamera_data
from pupil.outliers import replace_outliers
from pupil import roi


def load_folder(cls):
    """ see assembling/tools.py """
    cls.times, cls.FILES, cls.nframes, cls.Lx, cls.Ly = load_FaceCamera_data(cls.imgfolder)


def set_ROI_area(cls, roi_coords=None):

    if (roi_coords is None) and (cls.ROI is not None):
        roi_coords = cls.ROI.position(cls)

    if roi_coords is not None:

        mx, my, sx, sy = roi_coords
        fullimg = np.load(os.path.join(cls.imgfolder, cls.FILES[0]))
        
        cls.fullx, cls.fully = np.meshgrid(np.arange(fullimg.shape[0]),
                                             np.arange(fullimg.shape[1]),
                                             indexing='ij')

        cls.Nx, cls.Ny = sx+1, sy+1
                
        cls.zoom_cond = (cls.fullx>=mx) & (cls.fullx<=(mx+sx)) &\
            (cls.fully>=my) & (cls.fully<=my+sy)
                
    else:
        print('need to provide coords or to create ROI !!')


def load_ROI_data(cls, iframe1=0, iframe2=50):

    DATA = np.zeros((iframe2-iframe1, cls.Nx, cls.Ny))
    
    for frame in np.arange(iframe1, iframe2):
        fullimg = np.load(os.path.join(cls.imgfolder,
                                       cls.FILES[frame]))
        DATA[frame-iframe1,:,:] = fullimg[cls.zoom_cond].reshape(cls.Nx, cls.Ny)
    
    return DATA
    

    
if __name__=='__main__':

    import argparse, datetime

    parser=argparse.ArgumentParser()
    parser.add_argument('-df', "--datafolder", type=str,
            default='/home/yann/UNPROCESSED/2021_05_20/13-59-57/')
    parser.add_argument('-s', "--subsampling", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    from datavyz import ges as ge

    if args.debug:
        """
        snippet of code to design/debug the fitting algorithm
        
        ---> to be used with the "-Debug-" button of the GUI
        """

        args.imgfolder = os.path.join(args.datafolder, 'FaceCamera-imgs')
        args.data = np.load(os.path.join(args.datafolder,
                                         'whisking.npy'), allow_pickle=True).item()
        load_folder(args)
        set_ROI_area(args, roi_coords=args.data['ROI'])
        DATA = load_ROI_data(args, iframe1=0, iframe2=1000)
        
        
    else:
        print('add --debug option')
