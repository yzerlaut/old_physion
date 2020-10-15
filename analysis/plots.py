import numpy as np
import pyqtgraph as pg
from PyQt5 import QtGui, QtCore
import os, sys, pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

pupilpen = pg.mkPen((255,0,0), width=3, style=QtCore.Qt.SolidLine)

def scale_and_position(self, y, value=None, i=0):
    if value is None:
        value=y
    return shift(self, i)+\
        self.settings['increase-factor']**i*\
        (value-y.min())/(y.max()-y.min())

def shift(self, i):
    return self.settings['blank-space']*i+\
        np.sum(np.power(self.settings['increase-factor'], np.arange(i)))

    
def raw_data_plot(self, tzoom,
                  plot_update=True,
                  with_images=False,
                  with_scatter=False):

    iplot = 0
    scatter = []
    
    ## -------- Screen --------- ##
    pen = pg.mkPen(color=self.settings['colors']['Screen'])
    if self.Screen is not None:
        # first photodiode signal
        cond = (self.Screen.photodiode.t>=tzoom[0]) & (self.Screen.photodiode.t<=tzoom[1])
        isampling = max([1,int(len(self.Screen.photodiode.t[cond])/self.settings['Npoints'])])
        y = scale_and_position(self, self.Screen.photodiode.val[cond][::isampling], i=0)
        if plot_update:
            self.plot.plot(self.Screen.photodiode.t[cond][::isampling], y, pen=pen)
        if with_images:
            tt, im = self.Screen.grab_frame(self.time,
                                            with_time=True,
                                            force_previous_time=True)
            self.pScreenimg.setImage(im)
            scatter.append((tt, 0))
            
        if with_scatter:
            itime = np.argmin((self.Screen.photodiode.t[cond]-\
                               self.time)**2)
            val = scale_and_position(self, y,
                    value=self.Screen.photodiode.val[cond][itime], i=0)
            scatter.append((self.Screen.photodiode.t[cond][itime], val))
    else:
        y = shift(self,0)+np.zeros(2)
        self.plot.plot([tzoom[0], tzoom[1]],y, pen=pen)


    ## -------- Locomotion --------- ##
    pen = pg.mkPen(color=self.settings['colors']['Locomotion'])
    if self.Locomotion is not None:
        cond = (self.Locomotion.t>=tzoom[0]) & (self.Locomotion.t<=tzoom[1])
        isampling = max([1, int(len(self.Locomotion.t[cond])/self.settings['Npoints'])])
        y = scale_and_position(self, self.Locomotion.val[cond][::isampling], i=1)
        if plot_update:
            self.plot.plot(self.Locomotion.t[cond][::isampling], y, pen=pen)
        if with_scatter:
            itime = np.argmin((self.Locomotion.t[cond]-self.time)**2)
            val = scale_and_position(self, y, value=self.Locomotion.val[cond][itime], i=1)
            scatter.append((self.Screen.photodiode.t[cond][itime], val))
    else:
        y = shift(self,1)+np.zeros(2)
        self.plot.plot([tzoom[0], tzoom[1]],y, pen=pen)

    ## -------- Face --------- ##
    if self.Face is not None:
        im_face = self.Face.grab_frame(self.time)
        self.pFaceimg.setImage(im_face)
        
    
    ## -------- Pupil --------- ##
    pen = pg.mkPen(color=self.settings['colors']['Pupil'])
    if self.Pupil is not None and self.Pupil.processed is not None:
        # time-varying diameter
        pt = self.Pupil.processed['times']
        cond = (pt>=tzoom[0]) & (pt<=tzoom[1])
        isampling = max([1,int(len(self.Pupil.processed['diameter'][cond])/self.settings['Npoints'])])
        y = scale_and_position(self,
            self.Pupil.processed['diameter'][cond][::isampling], i=2)
        if plot_update:
            self.plot.plot(pt[cond][::isampling], y,pen=pen)
        if with_images:
            # im_face = self.Face.grab_frame(self.time) # already loaded above
            self.pFaceimg.setImage(im_face)
            plot_pupil(self, im_face)
        if with_scatter:
            self.ipt = np.argmin((pt[cond]-self.time)**2) # used later
            val = scale_and_position(self, y,
                value=self.Pupil.processed['diameter'][cond][self.ipt],
                                     i=2)
            scatter.append((pt[cond][self.ipt], val))
        else:
            self.ipt = 0
    else:
        y = shift(self, 2)+np.zeros(2)
        self.plot.plot([tzoom[0], tzoom[1]], y, pen=pen)


    ## -------- Electrophy --------- ##
    pen = pg.mkPen(color=self.settings['colors']['Electrophy'])
    if self.Electrophy is not None:
        cond = (self.Electrophy.t>=tzoom[0]) & (self.Electrophy.t<=tzoom[1])
        isampling = max([1,int(len(self.Electrophy.t[cond])/self.settings['Npoints'])])
        y = scale_and_position(self, self.Electrophy.val[cond][::isampling], i=3)
        if plot_update:
            self.plot.plot(self.Electrophy.t[cond][::isampling], y, pen=pen)
        if with_scatter:
            itime = np.argmin((self.Electrophy.t[cond]-self.time)**2)
            val = scale_and_position(self, y, value=self.Electrophy.val[cond][itime], i=3)
            scatter.append((self.Electrophy.t[cond][itime], val))
    else:
        y = shift(self,3)+np.zeros(2)
        self.plot.plot([tzoom[0], tzoom[1]],y, pen=pen)

    ## -------- Calcium --------- ##
    pen = pg.mkPen(color=self.settings['colors']['Calcium'])
    if self.Calcium is not None:
        pass

    if self.Screen is not None:
        # if visual-stim we highlight the stim periods
        icond = np.argwhere((self.Screen.time_start>tzoom[0]-\
                             self.metadata['presentation-duration']) & \
                            (self.Screen.time_stop<tzoom[1]+\
                             self.metadata['presentation-duration'])).flatten()

        if hasattr(self, 'StimFill') and self.StimFill is not None:
            for x in self.StimFill:
                self.plot.removeItem(x)

        X, Y = [], []
        if len(icond)>0:
            # for i in range(max([0,icond[0]-1]),
            #                min([icond[-1]+1,len(self.Screen.time_stop)])):
            self.StimFill = []
            for i in icond:
                t0 = self.Screen.time_start[i]
                t1 = self.Screen.time_stop[i]
                self.StimFill.append(self.plot.plot([t0, t1], [0, 0],
                                fillLevel=y.max(), brush=(150,150,150,80)))

    if with_scatter and hasattr(self, 'scatter'):
        self.plot.removeItem(self.scatter)
        self.scatter.setData([s[0] for s in scatter],
                             [s[1] for s in scatter],
                             size=10, brush=pg.mkBrush(255,255,255))
        self.plot.addItem(self.scatter)

    self.plot.setRange(xRange=tzoom, yRange=[0,y.max()], padding=0.0)
    
    self.frameSlider.setValue(int(self.settings['Npoints']*(self.time-tzoom[0])/(tzoom[1]-tzoom[0])))
    
    self.plot.show()


def plot_pupil(self, img):

    if self.Pupil.xmin is None:
        x,y = np.meshgrid(np.arange(0,img.shape[0]),
                          np.arange(0,img.shape[1]), indexing='ij')
        cx, cy, sx, sy = self.Pupil.ROIellipse
        ellipse = ((y - cy)**2 / (sy/2)**2 +
                   (x - cx)**2 / (sx/2)**2) <= 1
        self.Pupil.xmin = np.min(x[ellipse])
        self.Pupil.xmax = np.max(x[ellipse])
        self.Pupil.ymin = np.min(y[ellipse])
        self.Pupil.ymax = np.max(y[ellipse])
    
    cropped_img = img[self.Pupil.xmin:self.Pupil.xmax,
                      self.Pupil.ymin:self.Pupil.ymax]

    self.pPupilimg.setImage(cropped_img)

    if self.Pupil.processed is not None:
        if self.PupilROI is not None:
            self.pPupil.removeItem(self.PupilROI)
        mx = self.Pupil.processed['cx'][self.ipt] - self.Pupil.xmin
        my = self.Pupil.processed['cy'][self.ipt] - self.Pupil.ymin
        sx = self.Pupil.processed['sx-corrected'][self.ipt]
        sy = self.Pupil.processed['sy-corrected'][self.ipt]
        
        self.PupilROI = pg.EllipseROI([my-sy/2., mx-sx/2.], [2*sy, 2*sx],
                                      pen=pupilpen)
        self.pPupil.addItem(self.PupilROI)
