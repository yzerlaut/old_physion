import numpy as np
import pyqtgraph as pg
from PyQt5 import QtGui, QtCore
import os, sys, pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

pupilpen = pg.mkPen((255,0,0), width=3, style=QtCore.Qt.SolidLine)

def scale_and_position(self, y, value=None, i=0):
    if value is None:
        value=y
    ymin, ymax = y.min(), y.max()
    if ymin<ymax:
        return shift(self, i)+\
            self.settings['increase-factor']**i*\
            (value-ymin)/(ymax-ymin)
    else:
        return shift(self, i)+value

def shift(self, i):
    return self.settings['blank-space']*i+\
        np.sum(np.power(self.settings['increase-factor'], np.arange(i)))

def convert_time_to_index(time, nwb_quantity):
    if nwb_quantity.timestamps is not None:
        return np.arange(nwb_quantity.timestamps.shape[0])[nwb_quantity.timestamps[:]>=time][0]
    elif nwb_quantity.starting_time is not None:
        t = time-nwb_quantity.starting_time
        dt = 1./nwb_quantity.rate
        imax = nwb_quantity.data.shape[0]-1 # maybe shift to -1 to handle images
        return max([1, min([int(t/dt), imax-1])]) # then we add +1 / -1 in the visualization
    else:
        return 0

def convert_index_to_time(index, nwb_quantity):
    """ index can be an array """
    return nwb_quantity.starting_time+index/nwb_quantity.rate


def raw_data_plot(self, tzoom,
                  plot_update=True,
                  with_images=False,
                  with_scatter=False):

    iplot = 0
    scatter = []
    
    ## -------- Screen --------- ##
    pen = pg.mkPen(color=self.settings['colors']['Screen'])

    if 'Photodiode-Signal' in self.nwbfile.acquisition:
        i1 = convert_time_to_index(tzoom[0], self.nwbfile.acquisition['Photodiode-Signal'])+1
        i2 = convert_time_to_index(tzoom[1], self.nwbfile.acquisition['Photodiode-Signal'])-1
        isampling = np.unique(np.linspace(i1, i2, self.settings['Npoints'], dtype=int))
        y = scale_and_position(self,self.nwbfile.acquisition['Photodiode-Signal'].data[isampling], i=iplot)
        iplot+=1
        if plot_update:
            self.plot.plot(convert_index_to_time(isampling, self.nwbfile.acquisition['Photodiode-Signal']), y, pen=pen)

    if 'visual-stimuli' in self.nwbfile.stimulus:
        
        i0 = convert_time_to_index(self.time, self.nwbfile.stimulus['visual-stimuli'])-1
        self.pScreenimg.setImage(self.nwbfile.stimulus['visual-stimuli'].data[i0])
        if hasattr(self, 'ScreenFrameLevel'):
            self.plot.removeItem(self.ScreenFrameLevel)
        self.ScreenFrameLevel = self.plot.plot(self.nwbfile.stimulus['visual-stimuli'].timestamps[i0]*np.ones(2), [0, y.max()], pen=pen, linewidth=0.5)


    # ## -------- Locomotion --------- ##
    pen = pg.mkPen(color=self.settings['colors']['Locomotion'])
    if 'Running-Speed' in self.nwbfile.acquisition:
        i1 = convert_time_to_index(tzoom[0], self.nwbfile.acquisition['Running-Speed'])+1
        i2 = convert_time_to_index(tzoom[1], self.nwbfile.acquisition['Running-Speed'])-1
        isampling = np.unique(np.linspace(i1, i2, self.settings['Npoints'], dtype=int))
        y = scale_and_position(self,self.nwbfile.acquisition['Running-Speed'].data[isampling], i=iplot)
        iplot+=1
        if plot_update:
            self.plot.plot(convert_index_to_time(isampling, self.nwbfile.acquisition['Running-Speed']), y, pen=pen)
            
    # if self.Locomotion is not None:
    #     cond = (self.Locomotion.t>=tzoom[0]) & (self.Locomotion.t<=tzoom[1])
    #     isampling = max([1, int(len(self.Locomotion.t[cond])/self.settings['Npoints'])])
    #     y = scale_and_position(self, self.Locomotion.val[cond][::isampling], i=iplot)
    #     iplot+=1
    #     if plot_update:
    #         self.plot.plot(self.Locomotion.t[cond][::isampling], y, pen=pen)
    #     if with_scatter:
    #         itime = np.argmin((self.Locomotion.t[cond]-self.time)**2)
    #         val = scale_and_position(self, y, value=self.Locomotion.val[cond][itime], i=iplot)
    #         scatter.append((self.Screen.photodiode.t[cond][itime], val))
    # else:
    #     y = shift(self,1)+np.zeros(2)
    #     self.plot.plot([tzoom[0], tzoom[1]],y, pen=pen)

    # ## -------- Face --------- ##
    # if self.Face is not None:
    #     im_face = self.Face.grab_frame(self.time)
    #     self.pFaceimg.setImage(im_face)
        
    
    # ## -------- Pupil --------- ##
    # pen = pg.mkPen(color=self.settings['colors']['Pupil'])
    # if self.Pupil is not None and self.Pupil.processed is not None:
    #     # time-varying diameter
    #     pt = self.Pupil.processed['times']
    #     cond = (pt>=tzoom[0]) & (pt<=tzoom[1])
    #     isampling = max([1,int(len(self.Pupil.processed['diameter'][cond])/self.settings['Npoints'])])
    #     y = scale_and_position(self,
    #                            self.Pupil.processed['diameter'][cond][::isampling], i=iplot)
    #     iplot+=1
    #     if plot_update:
    #         self.plot.plot(pt[cond][::isampling], y,pen=pen)
    #     if with_images:
    #         # im_face = self.Face.grab_frame(self.time) # already loaded above
    #         self.pFaceimg.setImage(im_face)
    #         plot_pupil(self, im_face)
    #     if with_scatter:
    #         self.ipt = np.argmin((pt[cond]-self.time)**2) # used later
    #         val = scale_and_position(self, y,
    #                                  value=self.Pupil.processed['diameter'][cond][self.ipt],i=iplot)
    #         scatter.append((pt[cond][self.ipt], val))
    #     else:
    #         self.ipt = 0
    # else:
    #     y = shift(self, 2)+np.zeros(2)
    #     self.plot.plot([tzoom[0], tzoom[1]], y, pen=pen)


    # ## -------- Electrophy --------- ##
    # pen = pg.mkPen(color=self.settings['colors']['Electrophy'])
    # if self.Electrophy is not None:
    #     cond = (self.Electrophy.t>=tzoom[0]) & (self.Electrophy.t<=tzoom[1])
    #     isampling = max([1,int(len(self.Electrophy.t[cond])/self.settings['Npoints'])])
    #     y = scale_and_position(self, self.Electrophy.val[cond][::isampling], i=iplot)
    #     iplot+=1
    #     if plot_update:
    #         self.plot.plot(self.Electrophy.t[cond][::isampling], y, pen=pen)
    #     if with_scatter:
    #         itime = np.argmin((self.Electrophy.t[cond]-self.time)**2)
    #         val = scale_and_position(self, y, value=self.Electrophy.val[cond][itime], i=iplot)
    #         scatter.append((self.Electrophy.t[cond][itime], val))
    # else:
    #     y = shift(self,3)+np.zeros(2)
    #     self.plot.plot([tzoom[0], tzoom[1]],y, pen=pen)

    # ## -------- Calcium --------- ##
    pen = pg.mkPen(color=self.settings['colors']['CaImaging'])
    if 'CaImaging-TimeSeries' in self.nwbfile.acquisition:
        i0 = convert_time_to_index(self.time, self.nwbfile.acquisition['CaImaging-TimeSeries'])
        self.pCaimg.setImage(self.nwbfile.acquisition['CaImaging-TimeSeries'].data[i0])
        if hasattr(self, 'CaFrameLevel'):
            self.plot.removeItem(self.CaFrameLevel)
        self.CaFrameLevel = self.plot.plot(self.nwbfile.acquisition['CaImaging-TimeSeries'].timestamps[i0]*np.ones(2), [0, y.max()], pen=pen, linewidth=0.5)
        
    # if self.CaImaging is not None:
    #     print(len(self.CaImaging.t))
    #     print(self.CaImaging.Firing.shape)
    #     cond = (self.CaImaging.t>=tzoom[0]) & (self.CaImaging.t<=tzoom[1])
    #     isampling = max([1,int(len(self.CaImaging.t[cond])/self.settings['Npoints'])])
    #     for n in range(self.CaImaging.Firing.shape[0]):
    #         y = scale_and_position(self, self.CaImaging.Firing[n,cond][::isampling], i=iplot)+.1*n
    #         # y = scale_and_position(self, self.CaImaging.Fluo[n,cond][::isampling], i=iplot)+.1*n 
    #         if plot_update:
    #             self.plot.plot(self.CaImaging.t[cond][::isampling], y, pen=pen)
    #     iplot+=1
    #     # if with_scatter:
    #     #     itime = np.argmin((self.CaImaging.t[cond]-self.time)**2)
    #     #     val = scale_and_position(self, y, value=self.Electrophy.val[cond][itime], i=iplot)
    #     #     scatter.append((self.Electrophy.t[cond][itime], val))
    # else:
    #     y = shift(self,3)+np.zeros(2)
    #     self.plot.plot([tzoom[0], tzoom[1]],y, pen=pen)
            

    if ('time_start_realigned' in self.nwbfile.stimulus) and ('time_stop_realigned' in self.nwbfile.stimulus):
        
        # if visual-stim we highlight the stim periods
        icond = np.argwhere((self.nwbfile.stimulus['time_start_realigned'].data[:]>tzoom[0]-10) & \
                            (self.nwbfile.stimulus['time_stop_realigned'].data[:]<tzoom[1]+10)).flatten()

        if hasattr(self, 'StimFill') and self.StimFill is not None:
            for x in self.StimFill:
                self.plot.removeItem(x)

        X, Y = [], []
        if len(icond)>0:
            self.StimFill = []
            # for i in icond:
            for i in range(max([0,icond[0]-1]),
                           min([icond[-1]+1,self.nwbfile.stimulus['time_stop_realigned'].data.shape[0]-1])):
                t0 = self.nwbfile.stimulus['time_start_realigned'].data[i]
                t1 = self.nwbfile.stimulus['time_stop_realigned'].data[i]
                self.StimFill.append(self.plot.plot([t0, t1], [0, 0],
                                fillLevel=y.max(), brush=(150,150,150,80)))

    # if with_scatter and hasattr(self, 'scatter'):
    #     self.plot.removeItem(self.scatter)
    #     self.scatter.setData([s[0] for s in scatter],
    #                          [s[1] for s in scatter],
    #                          size=10, brush=pg.mkBrush(255,255,255))
    #     self.plot.addItem(self.scatter)

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
