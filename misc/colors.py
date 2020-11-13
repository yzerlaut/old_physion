import numpy as np
# from matplotlib.cm import hsv, viridis, viridis_r, copper, copper_r, cool, jet,\
#     PiYG, binary, binary_r, bone, Pastel1, Pastel2, Paired, Accent, Dark2, Set1, Set2,\
#     Set3, tab10, tab20, tab20b, tab20c
from matplotlib.cm import hsv


def build_colors_from_array(array,
                            discretization=10,
                            cmap='hsv'):

    if discretization<len(array):
        discretization = len(array)
    Niter = int(len(array)/discretization)

    colors = (array%discretization)/discretization +\
        (array/discretization).astype(int)/discretization**2

    return np.array(255*hsv(colors)).astype(int)
    

if __name__=='__main__':

    import pyqtgraph as pg
    print(build_colors_from_array(np.arange(5)))
    # pen = pg.mkPen(color=build_colors_from_array(np.arange(33))[0])
