import sys
import numpy as np
import pyqtgraph as pg
from scipy.optimize import minimize

def fit_pupil_size(parent):
    parent.scatter = pg.ScatterPlotItem([0], [0], pen='k', symbol='+')
    parent.pROI.addItem(parent.scatter)
    parent.win.show()
    parent.show()
    

def prepare_fit(parent):

    np.savez('pupil.npz',
             **{'img':parent.img,
                'ximg':parent.ximg,
                'yimg':parent.yimg,
                'reflectors':[r.extract_props() for r in parent.rROI],
                'ROIpupil':parent.pupil.extract_props(),
                'ROIellipse':parent.ROI.extract_props()})
    parent.scatter = pg.ScatterPlotItem([0], [0], pen='k', symbol='+')
    parent.pROI.addItem(parent.scatter)
    return True


def ellipse_coords(xc, yc, sx, sy):
    t = np.linspace(0, 2*np.pi, 100)
    return xc+np.cos(t)*sx/2, yc+np.sin(t)*sy/2

def inside_ellipse_cond(X, Y, xc, yc, sx, sy):
    return ( (X-xc)**2/(sx/2.)**2+\
             (Y-yc)**2/(sy/2.)**2 ) < 1

def ellipse_binary_func(X, Y, xc, yc, sx, sy):
    Z = np.zeros(X.shape)
    Z[inside_ellipse_cond(X, Y, xc, yc, sx, sy)] = 1
    return Z

def build_grid(x, y, reflector_cond,
               bound_fraction_for_center=0.2,
               center_discretization=21, # better odd, to have the center position
               min_radius=5,
               radius_discretization=15):

    Dx, Dy = x.max()-x.min(), y.max()-y.min()
    xmin, xmax = x.min()+bound_fraction_for_center*Dx, x.max()-bound_fraction_for_center*Dx
    ymin, ymax = y.min()+bound_fraction_for_center*Dy, y.max()-bound_fraction_for_center*Dy

    Xc, Yc, S, Value = [], [], [], []
    COORDS = []  
    for xc in np.linspace(xmin, xmax, center_discretization):
        for yc in np.linspace(ymin, ymax, center_discretization):
            for s in np.linspace(min_radius,
                                 max([min_radius, np.min([yc-y.min(), y.max()-yc,
                                                          x.max()-xc, xc-x.min()])]),
                                 radius_discretization):
                COORDS.append([xc, yc, s, s])
    return np.array(COORDS), Value

def find_best_ellipse_on_GRID(img, x, y, reflector_cond):

    Xc, Yc, S, V = build_grid(x, y, reflector_cond)
    print('init')
    # Residuals = []
    # for xc, yc, s in zip(Xc, Yc, S):
    #     Residuals.append(np.sum(np.abs(ellipse_binary_func(x, y, xc, yc, s, s)[~reflector_cond]-\
    #                                    img[~reflector_cond])))

    imgr = img[~reflector_cond]
    ibest = np.argmin([np.sum((v-imgr)**2) for v in V])
    return Xc[ibest], Yc[ibest], S[ibest], S[ibest]

def residual(coords, x, y, img_no_reflect, reflector_cond):
    """
    Residual function: 1/CorrelationCoefficient ! (after blanking)
    """
    im = ellipse_binary_func(x, y, *coords)[~reflector_cond]
    if np.std(im)>0:
        return 1./np.abs(np.corrcoef(img_no_reflect,im)[0,1])
    else:
        return 1e3

def perform_fit(img, x, y, reflectors):

    reflector_cond = np.zeros(x.shape, dtype=bool)
    for r in data['reflectors']:
        reflector_cond = reflector_cond | inside_ellipse_cond(x, y, *r)
    img[reflector_cond] = 0

    img_no_reflect = img[~reflector_cond]

    # center_of_mass
    c0 = [np.sum(img*x)/np.sum(img),np.sum(img*y)/np.sum(img)]
    # std_of_mass
    s0 = [2*np.sqrt(np.sum(img*(x-c0[0])**2)/np.sum(img)),
          2*np.sqrt(np.sum(img*(y-c0[1])**2)/np.sum(img))]
    
    res = minimize(residual, [c0[0], c0[1], s0[0], s0[1]],
                   args=(x, y, img_no_reflect, reflector_cond),
                   method='Nelder-Mead',
                   tol=1e-8, options={'maxiter':1000})

    return res.x
    
if __name__=='__main__':
    
    from datavyz import ges as ge
    from analyz.IO.npz import load_dict

    # prepare data
    data = np.load('pupil.npz')
    img = (data['img'].max()-data['img'])/(data['img'].max()-data['img'].min())
    x, y = np.meshgrid(data['ximg'], data['yimg'], indexing='ij')

    
    # reflector condition
    reflector_cond = np.zeros(x.shape, dtype=bool)
    for r in data['reflectors']:
        reflector_cond = reflector_cond | inside_ellipse_cond(x, y, *r)
    img[reflector_cond] = 0

    img_no_reflect = img[~reflector_cond]
    
    fig, ax = ge.figure(figsize=(1.4,2), left=0, bottom=0, right=0, top=0)
    ge.image(img, ax=ax)
    ax.plot(*ellipse_coords(*data['ROIpupil']))
    ax.plot(*ellipse_coords(*perform_fit(img, x, y, data['reflectors'])))
    ge.show()
