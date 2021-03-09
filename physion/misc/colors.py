import numpy as np
# from matplotlib.cm import hsv, viridis, viridis_r, copper, copper_r, cool, jet,\
#     PiYG, binary, binary_r, bone, Pastel1, Pastel2, Paired, Accent, Dark2, Set1, Set2,\
#     Set3, tab10, tab20, tab20b, tab20c
from matplotlib.cm import hsv
from PyQt5 import QtGui, QtCore
import pyqtgraph as pg


def build_colors_from_array(array,
                            discretization=10,
                            cmap='hsv'):

    if discretization<len(array):
        discretization = len(array)
    Niter = int(len(array)/discretization)

    colors = (array%discretization)/discretization +\
        (array/discretization).astype(int)/discretization**2

    return np.array(255*hsv(colors)).astype(int)


def build_dark_palette(app):

    app.setStyle("Fusion")

    # Now use a palette to switch to dark colors:
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(53, 53, 53))
    # palette.setColor(QtGui.QPalette.Background, QtGui.QColor(53, 53, 53))
    # palette.setColor(QtGui.QPalette.PlaceholderText, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    # palette.setColor(QtGui.QPalette.Foreground, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(25, 25, 25))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.black)
    palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(50, 50, 50))
    palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
    # palette.setColor(QtGui.QPalette.Link, QtGui.QColor(42, 130, 218))
    # palette.setColor(QtGui.QPalette.Link, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Link, QtGui.QColor(200, 200, 200))
    # palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(150, 150, 150))
    palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)
    app.setPalette(palette)


if __name__=='__main__':

    import pyqtgraph as pg
    print(build_colors_from_array(np.arange(5)))
    # pen = pg.mkPen(color=build_colors_from_array(np.arange(33))[0])


    
