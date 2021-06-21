import os, sys, pathlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from PIL import Image


sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from assembling.IO.bruker_xml_parser import bruker_xml_parser

def build_from_bruker_tiffs(folder,
                            channel='Ch2',
                            threshold=255):


    xml_file = os.path.join(folder, os.path.join(folder.split('/')[-1]+'.xml'))
    bruker_data = bruker_xml_parser(xml_file)


    dx = float(bruker_data['settings']['micronsPerPixel']['XAxis'])

    img = np.array(Image.open(os.path.join(folder, bruker_data[channel]['tifFile'][0])))
    X, Y = np.meshgrid(dx*np.arange(img.shape[0]), dx*np.arange(img.shape[1]), indexing='ij')
    
    x, y = X.flatten(), Y.flatten()

    print('looping over tif files [...]')
    xs, ys, zs = [], [], []
    for z, img_file in zip(bruker_data[channel]['depth'], bruker_data[channel]['tifFile']):
        im = np.array(Image.open(os.path.join(folder, img_file))).flatten()
        cond = im>threshold
        zs = zs+list(np.ones(np.sum(cond))*z)
        xs = xs+list(x[cond])
        ys = ys+list(y[cond])



    for z, img_file in zip(bruker_data[channel]['depth'][::10], bruker_data[channel]['tifFile'][::10]):
        im = np.array(Image.open(os.path.join(folder, img_file)))
        fig, [ax,ax2] = plt.subplots(1, 2)
        ax.set_title('z=%.1f um' % z)
        ax2.set_title('thresholded', size=8)
        ax.imshow(im, cmap=plt.cm.viridis)#, origin='lower', extent=(0,0,dx*img.shape[0], dx*img.shape[1]))
        im[im<threshold] = 0
        im[im>=threshold] = 1
        pos = ax2.imshow(im, cmap=plt.cm.viridis)#, origin='lower', extent=(0,0,dx*img.shape[0], dx*img.shape[1]))
        cond = np.array(im).flatten()>threshold
        ax2.scatter(y[cond], x[cond], s=2, color='r')
        ax.axis('equal')
        ax2.axis('equal')
        plt.show()


    print('found %i supra-threshold dots'% len(xs))
    return xs, ys, zs
    

def plot(xs, ys, zs, ms=2):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xs, ys, zs, marker='o', color='r', s=ms)
    ax.set_xlabel('x (um)')
    ax.set_ylabel('y (um)')
    ax.set_zlabel('z (um)')    
    return fig, ax

    
if __name__=='__main__':
    
    folder = sys.argv[-1] # '/media/yann/Yann/2020_11_10/TSeries-11102020-1605-016'

    plot(*build_from_bruker_tiffs(folder, threshold=1400), ms=2)

    plt.show()











