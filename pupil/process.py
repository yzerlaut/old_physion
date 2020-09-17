import sys, os, pathlib
import numpy as np
import pyqtgraph as pg
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter
from analyz.workflow.shell import printProgressBar

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from assembling.saving import check_datafolder
from pupil.outliers import replace_outliers

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

def ellipse_residual(coords, x, y, img_no_reflect, reflector_cond):
    """
    Residual function: 1/CorrelationCoefficient ! (after blanking)
    """
    im = ellipse_binary_func(x, y, *coords)[~reflector_cond]
    if np.std(im)>0:
        return 1./np.abs(np.corrcoef(img_no_reflect,im)[0,1])
    else:
        return 1e3

def circle_residual(coords, x, y, img_no_reflect, reflector_cond):
    """
    Residual function: 1/CorrelationCoefficient ! (after blanking)
    """
    im = circle_binary_func(x, y, *coords)[~reflector_cond]
    if np.std(im)>0:
        return 1./np.abs(np.corrcoef(img_no_reflect,im)[0,1])
    else:
        return 1e3
    
def perform_fit(img, x, y, reflectors,
                shape='ellipse'):

    reflector_cond = np.zeros(x.shape, dtype=bool)
    for r in reflectors:
        reflector_cond = reflector_cond | inside_ellipse_cond(x, y, *r)
    img[reflector_cond] = 0

    img_no_reflect = img[~reflector_cond]

    # center_of_mass
    c0 = [np.sum(img*x)/np.sum(img),np.sum(img*y)/np.sum(img)]
    # std_of_mass
    s0 = [2*np.sqrt(np.sum(img*(x-c0[0])**2)/np.sum(img)),
          2*np.sqrt(np.sum(img*(y-c0[1])**2)/np.sum(img))]
    
    if shape=='ellipse':
        residual = ellipse_residual
        initial_guess = [c0[0], c0[1], s0[0], s0[1]]
        shape = ellipse_coords
    else: # circle
        residual = circle_residual
        initial_guess = [c0[0], c0[1], np.mean(s0)]
        shape = circle_coords
        
    res = minimize(residual, initial_guess,
                   args=(x, y, img_no_reflect, reflector_cond),
                   method='Nelder-Mead',
                   tol=1e-8, options={'maxiter':1000})
    return res.x, shape(*res.x), res.fun

def fit_pupil_size(parent, shape='circle',
                   reflectors=None,
                   ellipse=None):

    img = (parent.img.max()-parent.img)/(parent.img.max()-parent.img.min())
    x, y = np.meshgrid(parent.ximg, parent.yimg, indexing='ij')

    if reflectors is None: # means we can extract it from parent object
        reflectors = [r.extract_props() for r in parent.rROI]
    
    return perform_fit(img, x, y, reflectors, shape=shape)
    
    
def preprocess(cls, ellipse=None, img=None):

    # applying the ellipse mask
    if img is None:
        img = np.load(cls.filenames[cls.cframe]).copy()
    else:
        img = img.copy()

    if ellipse is not None:
        cx, cy, sx, sy = ellipse
        Lx, Ly = img.shape
        x,y = np.meshgrid(np.arange(0,Lx), np.arange(0,Ly), indexing='ij')
        ellipse = ((y - cy)**2 / (sy/2)**2 +
                    (x - cx)**2 / (sx/2)**2) <= 1
        img[~ellipse] = cls.saturation
        cls.xmin, cls.xmax = np.min(x[ellipse]), np.max(x[ellipse])
        cls.ymin, cls.ymax = np.min(y[ellipse]), np.max(y[ellipse])
        
    elif cls.ROI is not None:
        img[~cls.ROI.ellipse] = cls.saturation

        img = img[np.min(cls.ROI.x[cls.ROI.ellipse]):np.max(cls.ROI.x[cls.ROI.ellipse]):,\
                  np.min(cls.ROI.y[cls.ROI.ellipse]):np.max(cls.ROI.y[cls.ROI.ellipse])]

    # first smooth
    img = gaussian_filter(img, cls.gaussian_smoothing)
    
    # then threshold
    img[img>cls.saturation] = cls.saturation

    cls.img = img
    cls.ximg, cls.yimg = np.arange(cls.img.shape[0]), np.arange(cls.img.shape[1])

    return img

def build_temporal_subsampling(cls,
                               folders = [],
                               sampling_rate=None):
    """
    """
    if sampling_rate is None:
        cls.sampling_rate = float(cls.rateBox.text())
    else:
        cls.sampling_rate = sampling_rate

    if len(folders)>0: # means batch processing

        cls.iframes, cls.times = [], []
        cls.nframes, cls.filenames = 0, []
        last_time = 0
        for i, df in enumerate(folders):
            times = np.load(os.path.join(df, 'FaceCamera-times.npy'))
            t0, t = times[0], times[0]
            iframes = []
            while t<=times[-1]:
                it = np.argmin((times-t)**2)
                iframes.append(it)
                cls.times.append(last_time+t-t0)
                t+=1./cls.sampling_rate

            fns = np.array(sorted(os.listdir(os.path.join(df,
                                                        'FaceCamera-imgs'))))[iframes]
            cls.filenames = np.concatenate([cls.filenames,
                                      [os.path.join(df, 'FaceCamera-imgs', f) for f in fns]])
            last_time = cls.times[-1] # 
            
        cls.times, cls.PD = np.array(cls.times), np.zeros(len(cls.times))
        cls.nframes = len(cls.times)
        
    else: # single datafolder processing
        times = np.load(os.path.join(cls.datafolder, 'FaceCamera-times.npy'))
        t0, t, cls.iframes, cls.times = times[0], times[0], [], []
        while t<=times[-1]:
            it = np.argmin((times-t)**2)
            cls.iframes.append(it)
            cls.times.append(t-t0)
            t+=1./cls.sampling_rate
        cls.times = np.array(cls.times)
        if not hasattr(cls, 'PD'):
            cls.PD = np.zeros(len(cls.times))
        cls.nframes = len(cls.iframes)
        fns = np.array(sorted(os.listdir(os.path.join(cls.datafolder,
                                                      'FaceCamera-imgs'))))[cls.iframes]
        cls.filenames = np.array([os.path.join(cls.datafolder, 'FaceCamera-imgs', f) for f in fns])

if __name__=='__main__':

    import argparse

    parser=argparse.ArgumentParser()
    parser.add_argument("--shape", default='circle')
    parser.add_argument("--sampling_rate", type=float, default=5.)
    parser.add_argument("--saturation", type=float, default=75)
    # parser.add_argument("--ellipse", type=float, default=[], nargs=)
    parser.add_argument("--gaussian_smoothing", type=float, default=2)
    parser.add_argument('-df', "--datafolder", default='./')
    parser.add_argument('-f', "--saving_filename", default='pupil-data.npy')
    parser.add_argument("-nv", "--non_verbose", help="decrease output verbosity", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if not args.debug:
        if os.path.isdir(args.datafolder) and\
           os.path.isfile(os.path.join(args.datafolder, 'pupil-ROIs.npy')):
            # load ROI
            rois = np.load(os.path.join(args.datafolder, 'pupil-ROIs.npy'),allow_pickle=True).item()
            args.saturation = rois['ROIsaturation']
            args.reflectors = rois['reflectors']
            args.ellipse = rois['ROIellipse']
            # insure data ordering and build sampling
            check_datafolder(args.datafolder)
            build_temporal_subsampling(args, sampling_rate=args.sampling_rate)
            # initialize data
            data = dict(vars(args))
            for key in ['cx', 'cy', 'sx', 'sy', 'residual']:
                data[key] = np.zeros(args.nframes)
            data['times'] = args.times
            # -- loop over frames
            print('\n Processing images to track pupil size and position in "%s"' % args.datafolder)
            if not args.non_verbose:
                printProgressBar(0, args.nframes)
            for args.cframe in range(args.nframes):
                # preprocess image
                args.img = preprocess(args, ellipse=args.ellipse)
                data['xmin'], data['xmax'] = args.xmin, args.xmax
                data['ymin'], data['ymax'] = args.ymin, args.ymax
                coords, _, res = fit_pupil_size(args, reflectors=args.reflectors)
                data['cx'][args.cframe] = coords[0]
                data['cy'][args.cframe] = coords[1]
                data['sx'][args.cframe] = coords[2]
                if args.shape=='circle':
                    data['sy'][args.cframe] = coords[2]
                else:
                    data['sy'][args.cframe] = coords[3]
                data['residual'][args.cframe] = res
                if not args.non_verbose:
                    printProgressBar(args.cframe, args.nframes)
            data = replace_outliers(data) # dealing with outliers
            np.save(os.path.join(args.datafolder, args.saving_filename), data)
            if not args.non_verbose:
                printProgressBar(args.nframes, args.nframes)
                print('Pupil size calculation over !')
                print('Processed data saved as:', os.path.join(args.datafolder, args.saving_filename))
            # save analysis output
        elif not os.path.isfile(os.path.join(args.datafolder, 'pupil-ROIs.npy')):
            print('Need to save ROIs for this datafolder !')
        else:
            print("ERROR: provide a valid data folder with a !")
    else:
        """
        snippet of code to design/debug the fitting algorithm
        
        ---> to be used with the "-Debug-" button of the GUI
        """
        from datavyz import ges as ge
        from analyz.IO.npz import load_dict

        # prepare data
        data = np.load('pupil.npz')
        x, y = np.meshgrid(data['ximg'], data['yimg'], indexing='ij')

        fig, ax = ge.figure(figsize=(1.4,2), left=0, bottom=0, right=0, top=0)
        ge.image(data['img'], ax=ax)
        # ax.plot(*ellipse_coords(*data['ROIpupil']))
        ax.plot(*perform_fit(data['img'], x, y, data['reflectors'], shape='circle')[1])
        ge.show()
