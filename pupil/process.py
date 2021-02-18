import sys, os, pathlib
import numpy as np
import pyqtgraph as pg
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter
from analyz.workflow.shell import printProgressBar

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.dataset import Dataset
from pupil.outliers import replace_outliers
from pupil import roi

def ellipse_coords(xc, yc, sx, sy, n=50):
    t = np.linspace(0, 2*np.pi, n)
    return xc+np.cos(t)*sx/2, yc+np.sin(t)*sy/2

def circle_coords(xc, yc, s, n=50):
    t = np.linspace(0, 2*np.pi, n)
    return xc+np.cos(t)*s/2, yc+np.sin(t)*s/2

def inside_ellipse_cond(X, Y, xc, yc, sx, sy):
    return ( (X-xc)**2/(sx/2.)**2+\
             (Y-yc)**2/(sy/2.)**2 ) < 1

def inside_circle_cond(X, Y, xc, yc, s):
    return ( (X-xc)**2/(s/2.)**2+\
             (Y-yc)**2/(s/2.)**2 ) < 1

def ellipse_binary_func(X, Y, xc, yc, sx, sy):
    Z = np.zeros(X.shape)
    Z[inside_ellipse_cond(X, Y, xc, yc, sx, sy)] = 1
    return Z

def circle_binary_func(X, Y, xc, yc, s):
    Z = np.zeros(X.shape)
    Z[inside_circle_cond(X, Y, xc, yc, s)] = 1
    return Z

def ellipse_residual(coords, cls):
    """
    Residual function: 1/CorrelationCoefficient ! (after blanking)
    """
    im = ellipse_binary_func(cls.x, cls.y, *coords)
    im[~cls.fit_area] = 0
    return np.mean((cls.img-im)**2)

def circle_residual(coords, cls):
    """
    Residual function: 1/CorrelationCoefficient ! (after blanking)
    """
    im = circle_binary_func(cls.x, cls.y, *coords)
    im[~cls.fit_area] = 0
    return np.mean((cls.img-im)**2)
    
def perform_fit(cls,
                shape='ellipse',
                saturation=100,
                maxiter=100):

    im0 = cls.img
    im0[cls.img<cls.saturation] = 1
    im0[cls.img>=cls.saturation] = 0

    # center_of_mass
    c0 = [np.mean(cls.x[im0>0]),np.mean(cls.y[im0>0])]
    # std_of_mass
    s0 = [4*np.std(cls.x[im0>0]),4*np.std(cls.y[im0>0])]

    if shape=='ellipse':
        residual = ellipse_residual
        initial_guess = [c0[0], c0[1], s0[0], s0[1]]
    else: # circle
        residual = circle_residual
        initial_guess = [c0[0], c0[1], np.mean(s0)]

    res = minimize(residual, initial_guess,
                   args=(cls),
                   method='Nelder-Mead',
                   tol=1e-8, options={'maxiter':maxiter})

    return res.x, None, res.fun


def perform_loop(parent,
                 subsampling=1000,
                 shape='ellipse',
                 gaussian_smoothing=0,
                 saturation=100,
                 with_ProgressBar=False):

    temp = {} # temporary data in case of subsampling
    for key in ['frame', 'cx', 'cy', 'sx', 'sy', 'diameter', 'residual']:
        temp[key] = []
    
    if with_ProgressBar:
        printProgressBar(0, parent.nframes)

    init_fit_area(parent)
    for parent.cframe in list(range(parent.nframes-1)[::subsampling])+[parent.nframes-1]:
        # preprocess image
        preprocess(parent, with_reinit=False, gaussian_smoothing=gaussian_smoothing)
        coords, _, res = perform_fit(parent,
                                     saturation=saturation,
                                     shape=parent.Pshape)
        if parent.Pshape=='circle':
            coords = list(coords)+[coords[-1]] # form circle to ellipse
        temp['frame'].append(parent.cframe)
        temp['residual'].append(res)
        for key, val in zip(['cx', 'cy', 'sx', 'sy'], coords):
            temp[key].append(val)
        temp['diameter'].append(np.pi*coords[2]*coords[3])
        if with_ProgressBar:
            printProgressBar(parent.cframe, parent.nframes)
    if with_ProgressBar:
        printProgressBar(parent.nframes, parent.nframes)
        
    print('Pupil size calculation over !')

    return temp
    
def extract_boundaries_from_ellipse(ellipse, Lx, Ly):
    cx, cy, sx, sy = ellipse
    x,y = np.meshgrid(np.arange(0,Lx), np.arange(0,Ly), indexing='ij')
    ellipse = ((y - cy)**2 / (sy/2)**2 +
               (x - cx)**2 / (sx/2)**2) <= 1
    xmin, xmax = np.min(x[ellipse]), np.max(x[ellipse])
    ymin, ymax = np.min(y[ellipse]), np.max(y[ellipse])

    return {'xmin':xmin,
            'xmax':xmax,
            'ymin':ymin,
            'ymax':ymax}

def init_fit_area(cls,
                  fullimg=None,
                  ellipse=None,
                  reflectors=None):

    if fullimg is None:
        fullimg = np.load(os.path.join(cls.imgfolder,cls.FILES[0]))

    cls.fullx, cls.fully = np.meshgrid(np.arange(fullimg.shape[0]),
                                       np.arange(fullimg.shape[1]),
                                       indexing='ij')

    if ellipse is not None:
        bfe = extract_boundaries_from_ellipse(ellipse, cls.Lx, cls.Ly)
        cls.zoom_cond = ((cls.fullx>=bfe['xmin']) &\
                         (cls.fullx<=bfe['xmax'])) &\
                         (cls.fully>=bfe['ymin']) &\
                         (cls.fully<=bfe['ymax'])
        Nx, Ny = bfe['xmax']-bfe['xmin']+1, bfe['ymax']-bfe['ymin']+1
    else:
        cls.zoom_cond = ((cls.fullx>=np.min(cls.ROI.x[cls.ROI.ellipse])) &\
                         (cls.fullx<=np.max(cls.ROI.x[cls.ROI.ellipse])) &\
                         (cls.fully>=np.min(cls.ROI.y[cls.ROI.ellipse])) &\
                         (cls.fully<=np.max(cls.ROI.y[cls.ROI.ellipse])))
    
        Nx=np.max(cls.ROI.x[cls.ROI.ellipse])-\
            np.min(cls.ROI.x[cls.ROI.ellipse])+1
        Ny=np.max(cls.ROI.y[cls.ROI.ellipse])-\
            np.min(cls.ROI.y[cls.ROI.ellipse])+1
        
    cls.x, cls.y = cls.fullx[cls.zoom_cond].reshape(Nx,Ny),\
        cls.fully[cls.zoom_cond].reshape(Nx,Ny)
    cls.x -= cls.x[0,0]
    cls.y -= cls.y[0,0]

    if ellipse is not None:
        cls.fit_area = inside_ellipse_cond(cls.x, cls.y, *ellipse)
    else:
        cls.fit_area = cls.ROI.ellipse[cls.zoom_cond].reshape(Nx,Ny)

    if reflectors is None:
        reflectors = [r.extract_props() for r in cls.rROI]
    for r in reflectors:
        cls.fit_area = cls.fit_area & ~inside_ellipse_cond(cls.x, cls.y, *r)

def preprocess(cls, with_reinit=True, img=None, gaussian_smoothing=0):

    if (img is None) and (cls.FaceCamera is not None):
        img = cls.FaceCamera.data[cls.cframe,:,:]
    elif (img is None):
        img = np.load(os.path.join(cls.imgfolder, cls.FILES[cls.cframe]))
    else:
        img = img.copy()

    if with_reinit:
        init_fit_area(cls)
    cls.img = img[cls.zoom_cond].reshape(cls.x.shape)
    
    # first smooth
    if gaussian_smoothing>0:
        cls.img = gaussian_filter(cls.img, gaussian_smoothing)

    # # then threshold
    cls.img[~cls.fit_area] = cls.saturation
    cond = cls.img>=cls.saturation
    cls.img[cond] = cls.saturation

    return cls.img


def load_folder(cls):
    times = np.array([float(f.replace('.npy', '')) for f in os.listdir(cls.imgfolder) if f.endswith('.npy')])
    cls.times = times[np.argsort(times)]
    cls.FILES = np.array([f for f in os.listdir(cls.imgfolder) if f.endswith('.npy')])[np.argsort(times)]
    cls.nframes = len(cls.times)
    cls.Lx, cls.Ly = np.load(os.path.join(cls.imgfolder, cls.FILES[0])).shape


def load_ROI(cls, with_plot=True):

    saturation = cls.data['ROIsaturation']
    if hasattr(cls, 'sl'):
        cls.sl.setValue(int(saturation))
    cls.ROI = roi.sROI(parent=cls,
                       pos = roi.ellipse_props_to_ROI(cls.data['ROIellipse']))
    cls.rROI = []
    cls.reflectors = []
    if 'reflectors' in cls.data:
        for r in cls.data['reflectors']:
            cls.rROI.append(roi.reflectROI(len(cls.rROI),
                                            pos = roi.ellipse_props_to_ROI(r),
                                            moveable=True, parent=cls))
    
if __name__=='__main__':

    import argparse, datetime

    parser=argparse.ArgumentParser()
    parser.add_argument('-df', "--datafolder", type=str,default='')
    # parser.add_argument("--shape", default='ellipse')
    # parser.add_argument("--saturation", type=float, default=75)
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--subsampling", type=int, default=1000)
    # parser.add_argument("--ellipse", type=float, default=[], nargs=)
    # parser.add_argument("--gaussian_smoothing", type=float, default=0)
    # parser.add_argument('-df', "--datafolder", default='./')
    # parser.add_argument('-f', "--saving_filename", default='pupil-data.npy')
    parser.add_argument("-nv", "--non_verbose", help="decrease output verbosity", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument('-rf', "--root_datafolder", type=str, default=os.path.join(os.path.expanduser('~'), 'DATA'))
    parser.add_argument('-d', "--day", type=str, default=datetime.datetime.today().strftime('%Y_%m_%d'))
    parser.add_argument('-t', "--time", type=str, default='')
    
    args = parser.parse_args()

    if args.time!='':
        args.datafolder = os.path.join(args.root_datafolder, args.day,
                                       args.time)
    if not args.debug:
        if os.path.isfile(os.path.join(args.datafolder, 'pupil.npy')):
            args.imgfolder = os.path.join(args.datafolder, 'FaceCamera-imgs')
            load_folder(args)
            args.data = np.load(os.path.join(args.datafolder, 'pupil.npy'),
                           allow_pickle=True).item()
            init_fit_area(args,
                          fullimg=None,
                          ellipse=args.data['ROIellipse'],
                          reflectors=args.data['reflectors'])
            temp = perform_loop(args,
                    subsampling=args.subsampling,
                    shape=args.data['shape'],
                    gaussian_smoothing=args.data['gaussian_smoothing'],
                    with_ProgressBar=True)
            

        else:
            print('  /!\ "pupil.npy" file found, create one with the GUI  /!\ ')
            
        #     nwbfile = pynwb.NWBHDF5IO(args.datafile, 'r').read()
        #     args.FaceCamera = nwbfile.acquisition['FaceCamera']
        #     args.nframes, args.Lx, args.Ly = args.FaceCamera.data.shape

        #     # -- loop over frames
        #     print('\n Processing images to track pupil size and position in "%s"' % args.datafile)
        #     if not args.non_verbose:
        #         printProgressBar(0, args.nframes)
            
        #     for args.cframe in range(args.nframes):
        #         # preprocess image
        #         args.img = preprocess(args, ellipse=args.ellipse)
        #         coords, _, res = fit_pupil_size(args,
        #                                         reflectors=args.reflectors,
        #                                         maxiter=args.maxiter,
        #                                         shape=args.shape)
        #         data['diameter'][args.cframe] = np.pi*coords[2]*coords[3]
        #         for key, val in zip(['cx', 'cy', 'sx', 'sy'], coords):
        #             data[key][args.cframe] = val
        #         printProgressBar(args.cframe, args.nframes)
        #     printProgressBar(args.nframes, args.nframes)
        #     # adding min-max of picture
        #     data['xmin'], data['xmax'] = args.xmin, args.xmax
        #     data['ymin'], data['ymax'] = args.ymin, args.ymax
        #     data = replace_outliers(data) # dealing with outliers
        #     if not args.non_verbose:
        #         printProgressBar(len(args.times), args.nframes)
        #         print('Pupil size calculation over !')
        #         print('Processed data saved as:', args.datafile.replace('.nwb', '.pupil.npy'))
        #     # save analysis output
        #     np.save(args.datafile.replace('.nwb', '.pupil.npy'), data)
        # elif not os.path.isfile(args.datafile.replace('.nwb', '.pupil.npy')):
        #     print('Need to save ROIs for this datafolder !')
        # else:
        #     print("ERROR: provide a valid NWB datafile !")
    else:
        """
        snippet of code to design/debug the fitting algorithm
        
        ---> to be used with the "-Debug-" button of the GUI
        """
        from datavyz import ges as ge
        from analyz.IO.npz import load_dict

        # prepare data
        data = np.load('pupil.npy', allow_pickle=True).item()
        x, y = np.meshgrid(data['ximg'], data['yimg'], indexing='ij')

        fig, ax = ge.figure(figsize=(1.4,2), left=0, bottom=0, right=0, top=0)
        ge.image(data['img'], ax=ax)
        # ax.plot(*ellipse_coords(*data['ROIpupil']))
        ax.plot(*perform_fit(data['img'], x, y, data['reflectors'], shape='circle')[1])
        ge.show()
