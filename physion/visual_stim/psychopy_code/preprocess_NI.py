import os, sys
import numpy as np
from scipy.interpolate import RectBivariateSpline, interp2d

import itertools, string, sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from datavyz.images import load

def img_after_hist_normalization(img):
    """
    for NATURAL IMAGES:
    histogram normalization to get comparable images
    """
    print('Performing histogram normalization [...]')
    flat = np.array(1000*img.flatten(), dtype=int)

    cumsum = np.cumsum(np.histogram(flat, bins=np.arange(1001))[0])

    norm_cs = np.concatenate([(cumsum-cumsum.min())/(cumsum.max()-cumsum.min())*1000, [1]])
    new_img = np.array([norm_cs[f]/1000. for f in flat])

    return new_img.reshape(img.shape)


def adapt_to_screen_resolution(img, new_screen_size):

    print('Adapting image to chosen screen resolution [...]')
    
    old_X = np.arange(img.shape[0])
    old_Y = np.arange(img.shape[1])
    
    new_X = np.arange(new_screen_size[0])
    new_Y = np.arange(new_screen_size[1])

    new_img = np.zeros(new_screen_size)
    
    spline_approx = interp2d(old_X, old_Y, img.T, kind='linear')
    
    return spline_approx(new_X, new_Y).T

    
if __name__=='__main__':

    from datavyz import ge
    NI_directory = os.path.join(str(pathlib.Path(__file__).resolve().parents[1]), 'NI_bank')
    
    image_number = 0
    filename = os.listdir(NI_directory)[image_number]
    img = load(os.path.join(NI_directory, filename))

    SCREEN = {'width':20, 'height':12, 'Xd_max':1200, 'Yd_max':800}
    # rescaled_img = adapt_to_screen_resolution(img, SCREEN)
    rescaled_img = img_after_hist_normalization(img)


    ge.image(rescaled_img)
    ge.show()
