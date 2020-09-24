import numpy as np
import pyqtgraph as pg
from PyQt5 import QtGui
from PIL import Image
import os

def scale_and_position(self, y, i=0):
    return shift(self, i)++\
        self.settings['increase-factor']**i*\
        (y-y.min())/(y.max()-y.min())

def shift(self, i):
    return self.settings['blank-space']*i+\
        np.sum(np.power(self.settings['increase-factor'], np.arange(i)))

def raw_data_plot(self, tzoom):

    iplot = 0
    scatter = []
    
    ## -------- Screen --------- ##
    pen = pg.mkPen(color=self.settings['colors']['Screen'])
    if self.Screen is not None:
        # first photodiode signal
        cond = (self.Screen['times']>=tzoom[0]) & (self.Screen['times']<=tzoom[1])
        isampling = max([1,int(len(self.Screen['times'][cond])/self.settings['Npoints'])])
        y = scale_and_position(self, self.Screen['photodiode'][cond][::isampling], i=0)
        self.plot.plot(self.Screen['times'][cond][::isampling], y, pen=pen)
        itime = np.argmin((self.Screen['times'][cond]-self.time)**2)
        scatter.append((self.Screen['times'][cond][itime], y[itime]))
    else:
        self.plot.plot([tzoom[0], tzoom[1]], shift(self, 0)+np.zeros(2), pen=pen)

    ## -------- Locomotion --------- ##
    pen = pg.mkPen(color=self.settings['colors']['Locomotion'])
    if self.Locomotion is not None:
        cond = (self.Locomotion['times']>=tzoom[0]) & (self.Locomotion['times']<=tzoom[1])
        isampling = max([int(len(self.Locomotion['times'][cond])/self.settings['Npoints'])])
        y = scale_and_position(self, self.Locomotion['trace'][cond][::isampling], i=1)
        itime = np.argmin((self.Locomotion['times'][cond]-self.time)**2)
        scatter.append((self.Locomotion['times'][cond][itime], y[itime]))
        self.plot.plot(self.Locomotion['times'][cond][::isampling], y, pen=pen)
    else:
        self.plot.plot([tzoom[0], tzoom[1]], shift(self, 1)+np.zeros(2), pen=pen)


    ## -------- Pupil --------- ##
    pen = pg.mkPen(color=self.settings['colors']['Pupil'])
    if self.Pupil is not None:
        # time-varying diameter
        cond = (self.Pupil['times']>=tzoom[0]) & (self.Pupil['times']<=tzoom[1])
        isampling = max([1,int(len(self.Pupil['times'][cond])/self.settings['Npoints'])])
        y = scale_and_position(self, self.Pupil['diameter'][cond][::isampling], i=2)
        self.plot.plot(self.Pupil['times'][cond][::isampling], y,pen=pen)
        itime = np.argmin((self.Pupil['times'][cond]-self.time)**2)
        scatter.append((self.Pupil['times'][cond][itime], y[itime]))
    else:
        self.plot.plot([tzoom[0], tzoom[1]], shift(self, 2)+np.zeros(2), pen=pen)


    ## -------- Electrophy --------- ##
    pen = pg.mkPen(color=self.settings['colors']['Electrophy'])
    if self.Electrophy is not None:
        # time-varying diameter
        cond = (self.Electrophy['times']>=tzoom[0]) & (self.Electrophy['times']<=tzoom[1])
        isampling = max([1,int(len(self.Electrophy['times'][cond])/self.settings['Npoints'])])
        y = scale_and_position(self, self.Electrophy['trace'][cond][::isampling], i=3)
        itime = np.argmin((self.Electrophy['times'][cond]-self.time)**2)
        scatter.append((self.Electrophy['times'][cond][itime], y[itime]))
        self.plot.plot(self.Electrophy['times'][cond][::isampling], y, pen=pen)
    else:
        self.plot.plot([tzoom[0], tzoom[1]],shift(self,3)+np.zeros(2), pen=pen)

    ## -------- Calcium --------- ##
    pen = pg.mkPen(color=self.settings['colors']['Calcium'])
    if self.Calcium is not None:
        # time-varying diameter
        cond = (self.Pupil['times']>=tzoom[0]) & (self.Pupil['times']<=tzoom[1])
        isampling = max([1,int(len(self.Pupil['times'][cond])/self.settings['Npoints'])])
        y = self.Pupil['diameter'][cond][::isampling]
        self.plot.plot(self.Pupil['times'][cond][::isampling],
                       scale_and_position(self, y, i=4), pen=pen)
    else:
        self.plot.plot([tzoom[0], tzoom[1]], shift(self,4)+np.zeros(2), pen=pen)

    if self.Screen is not None:
        # if visual-stim we highlight the stim periods
        icond = np.argwhere((self.metadata['time_start_realigned']>tzoom[0]-\
                             self.metadata['presentation-duration']) & \
                            (self.metadata['time_stop_realigned']<tzoom[1]+\
                             self.metadata['presentation-duration'])).flatten()
        if len(icond)>0:
            for i in range(max([0,icond[0]-1]),
                           min([icond[-1]+1,len(self.metadata['time_stop_realigned'])])):
                t0 = self.metadata['time_start_realigned'][i]
                t1 = self.metadata['time_stop_realigned'][i]
                self.plot.plot([t0, t1], [0, 0], fillLevel=y.max(), brush=(150,150,150,100))
        
    self.scatter.setData([s[0] for s in scatter], [s[1] for s in scatter],
                         size=10, brush=pg.mkBrush(255,255,255))
    self.plot.addItem(self.scatter)

    self.plot.setRange(xRange=tzoom, yRange=[0,y.max()], padding=0.0)
    
    self.frameSlider.setValue(int(self.settings['Npoints']*(self.time-tzoom[0])/(tzoom[1]-tzoom[0])))
    
    self.plot.show()

def update_images(self, time):

    # pupil 
    iframe = min([len(self.Pupil['imgs'])-1,np.argmin((self.Pupil['times']-time)**2)+1])
    self.pFaceimg.setImage(np.load(self.Pupil['imgs'][iframe]))

    # screen
    
    # self.currentTime.setText('%.2f' % float(self.times[self.cframe]))

    # self.jump_to_frame()
    
