import sys
import os
import shutil
import time
import numpy as np
from PyQt5 import QtGui, QtCore
import pyqtgraph as pg
from pyqtgraph import GraphicsScene
import pims
from scipy.stats import zscore, skew
from matplotlib import cm

colors = np.array([[0,200,50],[180,0,50],[40,100,250],[150,50,150]])

import process

def extract_ellipse_props(ROI):
    """ extract ellipse props from ROI (NEED TO RECENTER !)"""
    xcenter = ROI.pos()[1]+ROI.size()[1]/2.
    ycenter = ROI.pos()[0]+ROI.size()[0]/2.
    sy, sx = ROI.size()
    return xcenter, ycenter, sx, sy

def ellipse_props_to_ROI(coords):
    """ re-translate to ROI props"""
    x0 = coords[0]-coords[2]/2
    y0 = coords[1]-coords[3]/2
    return x0, y0, coords[2], coords[3]

class reflectROI():
    def __init__(self, wROI, moveable=True,
                 parent=None, pos=None,
                 yrange=None, xrange=None, ellipse=None):
        # which ROI it belongs to
        self.wROI = wROI # can have many reflections
        self.color = (0.0,0.0,0.0)
        self.moveable = moveable
        
        if pos is None:
            view = parent.pROI.viewRange()
            imx = (view[0][1] + view[0][0]) / 2
            imy = (view[1][1] + view[1][0]) / 2
            dx = (view[0][1] - view[0][0]) / 4
            dy = (view[1][1] - view[1][0]) / 4
            dx = np.minimum(dx, parent.Ly*0.4)
            dy = np.minimum(dy, parent.Lx*0.4)
            imx = imx - dx / 2
            imy = imy - dy / 2
        else:
            imy, imx, dy, dx = pos
            self.yrange=yrange
            self.xrange=xrange
            self.ellipse=ellipse
        self.draw(parent, imy, imx, dy, dx)
        # self.ROI.sigRegionChangeFinished.connect(lambda: self.position(parent))
        # self.ROI.sigClicked.connect(lambda: self.position(parent))
        self.ROI.sigRemoveRequested.connect(lambda: self.remove(parent))

    def draw(self, parent, imy, imx, dy, dx):
        roipen = pg.mkPen(self.color, width=3,
                          style=QtCore.Qt.SolidLine)
        self.ROI = pg.EllipseROI(
            [imx, imy], [dx, dy],
            movable = self.moveable,
            pen=roipen, removable=self.moveable
        )
        self.ROI.handleSize = 8
        self.ROI.handlePen = roipen
        self.ROI.addScaleHandle([1, 0.5], [0., 0.5])
        self.ROI.addScaleHandle([0.5, 0], [0.5, 1])
        self.ROI.setAcceptedMouseButtons(QtCore.Qt.LeftButton)
        parent.pROI.addItem(self.ROI)

    def remove(self, parent):
        parent.pROI.removeItem(self.ROI)
        parent.win.show()
        parent.show()

    def position(self, parent):
        pass

    def extract_props(self):
        return extract_ellipse_props(self.ROI)
    
class pupilROI():
    def __init__(self, moveable=True,
                 parent=None, pos=None,
                 yrange=None, xrange=None, ellipse=None):
        self.color = (255.0,0.0,0.0)
        self.moveable = moveable
        
        if pos is None:
            view = parent.pROI.viewRange()
            imx = (view[0][1] + view[0][0]) / 2
            imy = (view[1][1] + view[1][0]) / 2
            dx = (view[0][1] - view[0][0]) / 5
            dy = (view[1][1] - view[1][0]) / 5
            dx = np.minimum(dx, parent.Ly*0.4)
            dy = np.minimum(dy, parent.Lx*0.4)
            imx = imx - dx / 2
            imy = imy - dy / 2
        else:
            imy, imx, dy, dx = pos
            self.yrange=yrange
            self.xrange=xrange
            self.ellipse=ellipse
        self.draw(parent, imy, imx, dy, dx)
        self.ROI.sigRemoveRequested.connect(lambda: self.remove(parent))

    def draw(self, parent, imy, imx, dy, dx):
        roipen = pg.mkPen(self.color, width=3,
                          style=QtCore.Qt.SolidLine)
        self.ROI = pg.EllipseROI(
            [imx, imy], [dx, dy],
            movable = self.moveable,
            pen=roipen, removable=self.moveable
        )
        self.ROI.handleSize = 8
        self.ROI.handlePen = roipen
        self.ROI.addScaleHandle([1, 0.5], [0., 0.5])
        self.ROI.addScaleHandle([0.5, 0], [0.5, 1])
        self.ROI.setAcceptedMouseButtons(QtCore.Qt.LeftButton)
        parent.pROI.addItem(self.ROI)

    def remove(self, parent):
        parent.pROI.removeItem(self.ROI)
        parent.win.show()
        parent.show()

    def position(self, parent):
        pass

    def extract_props(self):
        return extract_ellipse_props(self.ROI)
    

class sROI():
    def __init__(self, moveable=False,
                 parent=None, saturation=None, color=None, pos=None,
                 yrange=None, xrange=None,
                 ivid=None, pupil_sigma=None):
        if saturation is None:
            self.saturation = 0
        else:
            self.saturation = saturation
        self.moveable = moveable
        if color is None:
            self.color = np.maximum(0, np.minimum(255, colors[0]+np.random.randn(3)*70))
            self.color = tuple(self.color)
        else:
            self.color = color
            
        if pos is None:
            pos = int(3*parent.Lx/8), int(3*parent.Ly/8), int(parent.Lx/4), int(parent.Ly/4)
        self.draw(parent, *pos)
        
        self.moveable = moveable
        self.ROI.sigRegionChangeFinished.connect(lambda: self.position(parent))
        self.ROI.sigClicked.connect(lambda: self.position(parent))
        self.ROI.sigRemoveRequested.connect(lambda: self.remove(parent))
        self.position(parent)

    def draw(self, parent, imy, imx, dy, dx):
        roipen = pg.mkPen(self.color, width=3,
                          style=QtCore.Qt.SolidLine)
        self.ROI = pg.EllipseROI(
            [imx, imy], [dx, dy],
            pen=roipen, removable=self.moveable)
        self.ROI.handleSize = 8
        self.ROI.handlePen = roipen
        self.ROI.addScaleHandle([1, 0.5], [0., 0.5])
        self.ROI.addScaleHandle([0.5, 0], [0.5, 1])
        self.ROI.setAcceptedMouseButtons(QtCore.Qt.LeftButton)
        parent.p0.addItem(self.ROI)

    def position(self, parent):

        cx, cy, sx, sy = self.extract_props()
        
        xrange = np.arange(parent.Lx).astype(np.int32)
        yrange = np.arange(parent.Ly).astype(np.int32)
        ellipse = np.zeros((xrange.size, yrange.size), np.bool)
        self.x,self.y = np.meshgrid(np.arange(0,parent.Lx), np.arange(0,parent.Ly),
                                    indexing='ij')
        ellipse = ((self.y - cy)**2 / (sy/2)**2 +
                    (self.x - cx)**2 / (sx/2)**2) <= 1
        self.ellipse = ellipse
        parent.ROIellipse = self.extract_props()
        # parent.sl[1].setValue(parent.saturation * 100 / 255)
        self.plot(parent)

    def remove(self, parent):
        parent.p0.removeItem(self.ROI)
        parent.pROIimg.clear()
        if parent.scatter is not None:
            parent.pROI.removeItem(parent.scatter)
        parent.win.show()
        parent.show()


    def plot(self, parent):

        # get saturation level
        self.saturation = 255-parent.saturation
        
        process.preprocess(parent)
        
        parent.reflector.setEnabled(False)
        parent.reflector.setEnabled(True)
        
        parent.pROIimg.setImage(parent.img)
        parent.pROIimg.setLevels([0, self.saturation])
        # parent.pROI.setRange(xRange=(0,img.shape[0]), yRange=(0, img.shape[1]), padding=0.0)
        parent.win.show()
        parent.show()

    def plot_simple(self, parent):

        # get saturation level
        self.saturation = 255-parent.saturation

        # applying the ellipse mask
        img = parent.fullimg.copy()
        img[~self.ellipse] = 255-self.saturation
        
        img = img[np.min(self.x[self.ellipse]):np.max(self.x[self.ellipse]):,\
                  np.min(self.y[self.ellipse]):np.max(self.y[self.ellipse])]
        
        # smooth
        img = gaussian_filter(img, 2)
        # then threshold
        img[img>self.saturation] = 255-self.saturation
        parent.pROIimg.setImage(img)
        parent.pROIimg.setLevels([0, self.saturation])
        parent.img = img
        
    def extract_props(self):
        return extract_ellipse_props(self.ROI)
    
