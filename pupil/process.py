import sys, os
import numpy as np
import pyqtgraph as pg
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter

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

    return res.x, shape(*res.x)

def fit_pupil_size(parent, shape='circle'):

    img = (parent.img.max()-parent.img)/(parent.img.max()-parent.img.min())
    x, y = np.meshgrid(parent.ximg, parent.yimg, indexing='ij')

    reflectors = [r.extract_props() for r in parent.rROI]
    
    return perform_fit(img, x, y, reflectors, shape=shape)
    
    
def preprocess(cls, gaussian_smoothing=2):

    # applying the ellipse mask
    img = np.load(os.path.join(cls.datafolder, 'FaceCamera-imgs',
                               cls.filenames[cls.cframe])).copy()

    img[~cls.ROI.ellipse] = 255-cls.ROI.saturation

    img = img[np.min(cls.ROI.x[cls.ROI.ellipse]):np.max(cls.ROI.x[cls.ROI.ellipse]):,\
              np.min(cls.ROI.y[cls.ROI.ellipse]):np.max(cls.ROI.y[cls.ROI.ellipse])]

    # smooth
    img = gaussian_filter(img, gaussian_smoothing)
    # then threshold
    img[img>cls.ROI.saturation] = 255-cls.ROI.saturation

    cls.img = img
    cls.ximg, cls.yimg = np.arange(cls.img.shape[0]), np.arange(cls.img.shape[1])

    return img

def build_temporal_subsampling(cls):
    """
    """
    cls.sampling_rate = float(cls.rateBox.text())
    times = np.load(os.path.join(cls.datafolder, 'FaceCamera-times.npy'))
    t0, t, cls.iframes, cls.times = times[0], times[0], [], []
    while t<times[-1]:
        it = np.argmin((times-t)**2)
        cls.iframes.append(it)
        cls.times.append(t-t0)
        t+=1./cls.sampling_rate
    cls.times, cls.PD = np.array(cls.times), np.zeros(len(cls.times))
    cls.Pr1, cls.Pr2 = np.array(cls.times), np.zeros(len(cls.times))
    cls.nframes = len(cls.iframes)
    cls.filenames = np.array(sorted(os.listdir(os.path.join(cls.datafolder,
                                                            'FaceCamera-imgs'))))[cls.iframes]


def check_sanity(cls):

    filenames = os.listdir(os.path.join(cls.datafolder,'FaceCamera-imgs'))
    nmax = max([len(fn) for fn in filenames])
    for fn in filenames:
        n0 = len(fn)
        if n0<nmax:
            os.rename(os.path.join(cls.datafolder,'FaceCamera-imgs',fn),
                      os.path.join(cls.datafolder,'FaceCamera-imgs','0'*(nmax-n0)+fn))



if __name__=='__main__':

    import argparse

    
    parser=argparse.ArgumentParser()
    parser.add_argument("--shape", default='circle')
    parser.add_argument("--sampling_rate", type=float, default=10.)
    parser.add_argument("--data_folder", default='./')
    parser.add_argument("--saving_filename", default='pupil-data.hq.npy')
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    args = parser.parse_args()
    
    if os.path.isdir(args.datafolder):
        
        # insure data ordering and build sampling
        process.check_sanity(args)
        process.build_temporal_subsampling(args)

    else:
        print("ERROR: provide a valid data folder !")
