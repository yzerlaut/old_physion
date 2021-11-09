import os, sys, pathlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))


def plot(args):

    fig = plt.figure()
    for i, label in enumerate(['in_vivo', 'in_vitro']):
        ax = fig.add_subplot(121+i, projection='3d')
        ax.set_title(label, fontsize=8)
        cell_file = os.path.join(getattr(args, label+'_folder'), 'cells.npy')
        if os.path.isfile(cell_file):
            cells = np.load(cell_file, allow_pickle=True).item()
            if label=='in_vitro':
                zlim = cells['zlim']-np.min(cells['z']) # before we modify cells['z']
                cells['z'] = cells['z']-np.min(cells['z'])
                ax.scatter(cells['x'], cells['z'], cells['y'], marker='o', color='r', s=20)
                ax.set_xlabel('x (um)')
                ax.set_ylabel('z (um)')
                ax.set_zlabel('y (um)')
                ax.set_xlim([0, cells['dx']*cells['nxpix']])
                ax.set_zlim([0, cells['dy']*cells['nypix']])
                ax.set_ylim(zlim)
            else:
                ax.scatter(cells['x'], cells['y'], cells['z'], marker='o', color='r', s=20)
                ax.set_xlabel('x (um)')
                ax.set_ylabel('y (um)')
                ax.set_zlabel('z (um)')
                ax.set_xlim([0, cells['dx']*cells['nxpix']])
                ax.set_ylim([0, cells['dy']*cells['nypix']])
                ax.set_zlim(cells['zlim'])

    return fig


if __name__=='__main__':
    
    from misc.colors import build_dark_palette
    import tempfile, argparse, os
    parser=argparse.ArgumentParser(description="Algorithm for cell location matching",
                       formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-f1', "--in_vivo_folder", type=str,
                        default = '/home/yann/DATA/CaImaging/SSTcre_GCamp6s/Z-Stacks/In-Vivo-Mouse2')
    parser.add_argument('-f2', "--in_vitro_folder", type=str,
                        default = '/home/yann/DATA/CaImaging/SSTcre_GCamp6s/Z-Stacks/In-Vitro-Mouse2/Test SSTMarcel-912')
    parser.add_argument('-v', "--verbose", action="store_true")
    
    args = parser.parse_args()

    plot(args)
    plt.show()
    # app = QtWidgets.QApplication(sys.argv)
    # build_dark_palette(app)
    # main = MainWindow(app,
    #                   args=args)
    # sys.exit(app.exec_())
