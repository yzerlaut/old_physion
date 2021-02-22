import numpy as np
import pyqtgraph as pg
from PyQt5 import QtGui, QtCore
import os, sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from pupil import roi

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

def convert_time_to_index(time, nwb_quantity, axis=0):
    if nwb_quantity.timestamps is not None:
        cond = nwb_quantity.timestamps[:]>=time
        if np.sum(cond)>0:
            return np.arange(nwb_quantity.timestamps.shape[0])[cond][0]
        else:
            return nwb_quantity.timestamps.shape[axis]-1
    elif nwb_quantity.starting_time is not None:
        t = time-nwb_quantity.starting_time
        dt = 1./nwb_quantity.rate
        imax = nwb_quantity.data.shape[axis]-1 # maybe shift to -1 to handle images
        return max([1, min([int(t/dt), imax-1])]) # then we add +1 / -1 in the visualization
    else:
        return 0

def convert_index_to_time(index, nwb_quantity):
    """ index can be an array """
    return nwb_quantity.starting_time+index/nwb_quantity.rate


def raw_data_plot(self, tzoom,
                  plot_update=True,
                  with_images=False,
                  with_roi=False,
                  with_scatter=False):

    iplot = 0
    scatter = []
    self.plot.clear()
    
    ## -------- Screen --------- ##
    
    if 'Photodiode-Signal' in self.nwbfile.acquisition:
        
        i1 = convert_time_to_index(tzoom[0], self.nwbfile.acquisition['Photodiode-Signal'])+1
        i2 = convert_time_to_index(tzoom[1], self.nwbfile.acquisition['Photodiode-Signal'])-1
        if self.no_subsampling:
            isampling = np.arange(i1, i2)
        else:
            isampling = np.unique(np.linspace(i1, i2, self.settings['Npoints'], dtype=int))
        y = scale_and_position(self,self.nwbfile.acquisition['Photodiode-Signal'].data[isampling], i=iplot)
        iplot+=1
        self.plot.plot(convert_index_to_time(isampling, self.nwbfile.acquisition['Photodiode-Signal']), y,
                       pen=pg.mkPen(color=self.settings['colors']['Screen']))

    if 'visual-stimuli' in self.nwbfile.stimulus:
        
        i0 = convert_time_to_index(self.time, self.nwbfile.stimulus['visual-stimuli'])-1
        self.pScreenimg.setImage(self.nwbfile.stimulus['visual-stimuli'].data[i0])
        self.pScreenimg.setLevels([0,255])
        if hasattr(self, 'ScreenFrameLevel'):
            self.plot.removeItem(self.ScreenFrameLevel)
        # self.ScreenFrameLevel = self.plot.plot(self.nwbfile.stimulus['visual-stimuli'].timestamps[i0]*np.ones(2), [0, y.max()],
        #                                        pen=pg.mkPen(color=self.settings['colors']['Screen']), linewidth=0.5)


    ## -------- Locomotion --------- ##
    
    if 'Running-Speed' in self.nwbfile.acquisition:
        
        i1 = convert_time_to_index(tzoom[0], self.nwbfile.acquisition['Running-Speed'])+1
        i2 = convert_time_to_index(tzoom[1], self.nwbfile.acquisition['Running-Speed'])-1
        if self.no_subsampling:
            isampling = np.arange(i1, i2)
        else:
            isampling = np.unique(np.linspace(i1, i2, self.settings['Npoints'], dtype=int))
        y = scale_and_position(self,self.nwbfile.acquisition['Running-Speed'].data[isampling], i=iplot)
        iplot+=1
        self.plot.plot(convert_index_to_time(isampling, self.nwbfile.acquisition['Running-Speed']), y,
                       pen=pg.mkPen(color=self.settings['colors']['Locomotion']))
            

    ## -------- FaceCamera and Pupil-Size --------- ##
    
    pen = pg.mkPen(color=self.settings['colors']['Whisking'])
    if 'FaceCamera' in self.nwbfile.acquisition:
        i0 = convert_time_to_index(self.time, self.nwbfile.acquisition['FaceCamera'])
        self.pFaceimg.setImage(self.nwbfile.acquisition['FaceCamera'].data[i0])
        if hasattr(self, 'FaceCameraFrameLevel'):
            self.plot.removeItem(self.FaceCameraFrameLevel)
        self.FaceCameraFrameLevel = self.plot.plot(self.nwbfile.acquisition['FaceCamera'].timestamps[i0]*np.ones(2),
                                                   [0, y.max()], pen=pen, linewidth=0.5)
        

    pen = pg.mkPen(color=self.settings['colors']['Pupil'])
    if 'Pupil' in self.nwbfile.acquisition:
        i0 = convert_time_to_index(self.time, self.nwbfile.acquisition['Pupil'])
        img = self.nwbfile.acquisition['Pupil'].data[i0]
        self.pPupilimg.setImage(img)
        self.pPupilimg.setLevels([img.min(),img.max()])
        if hasattr(self, 'PupilFrameLevel'):
            self.plot.removeItem(self.PupilFrameLevel)
        self.PupilFrameLevel = self.plot.plot(self.nwbfile.acquisition['Pupil'].timestamps[i0]*np.ones(2),
                                              [0, y.max()], pen=pen, linewidth=0.5)
        t_pupil_frame = self.nwbfile.acquisition['Pupil'].timestamps[i0]
    else:
        t_pupil_frame = None
        
    if 'Pupil' in self.nwbfile.processing:

        i1 = convert_time_to_index(self.tzoom[0], self.nwbfile.processing['Pupil'].data_interfaces['cx'])
        i2 = convert_time_to_index(self.tzoom[1], self.nwbfile.processing['Pupil'].data_interfaces['cx'])

        img = self.nwbfile.acquisition['Pupil'].data[i0]
        self.pPupilimg.setImage(np.array(255/np.exp(1.)*(1.-np.exp(1.-img/255.)), dtype=int))
        y = scale_and_position(self,
                               self.nwbfile.processing['Pupil'].data_interfaces['sx'].data[i1:i2]*\
                               self.nwbfile.processing['Pupil'].data_interfaces['sy'].data[i1:i2],
                               i=iplot)
        iplot+=1
        self.plot.plot(self.nwbfile.processing['Pupil'].data_interfaces['cx'].timestamps[i1:i2], y, pen=pen)

        coords = []
        if hasattr(self, 'fit'):
            self.fit.remove(self)

        if t_pupil_frame is not None:
            i0 = convert_time_to_index(t_pupil_frame, self.nwbfile.processing['Pupil'].data_interfaces['cx'])
            for key in ['cx', 'cy', 'sx', 'sy']:
                coords.append(self.nwbfile.processing['Pupil'].data_interfaces[key].data[i0])
            self.fit = roi.pupilROI(moveable=False,
                                    parent=self,
                                    color=(125, 0, 0),
                                    pos = roi.ellipse_props_to_ROI(coords))
            

    # ## -------- Electrophy --------- ##
    if ('Electrophysiological-Signal' in self.nwbfile.acquisition):
        i1 = convert_time_to_index(tzoom[0], self.nwbfile.acquisition['Electrophysiological-Signal'])+1
        i2 = convert_time_to_index(tzoom[1], self.nwbfile.acquisition['Electrophysiological-Signal'])-1
        if self.no_subsampling:
            isampling = np.arange(i1, i2)
        else:
            isampling = np.unique(np.linspace(i1, i2, self.settings['Npoints'], dtype=int))
        y = scale_and_position(self,self.nwbfile.acquisition['Electrophysiological-Signal'].data[isampling], i=iplot)
        iplot+=1
        self.plot.plot(convert_index_to_time(isampling, self.nwbfile.acquisition['Electrophysiological-Signal']), y,
                       pen=pg.mkPen(color=self.settings['colors']['Electrophy']))


    # ## -------- Calcium --------- ##
    pen = pg.mkPen(color=self.settings['colors']['CaImaging'])
    if (self.time==0) and ('ophys' in self.nwbfile.processing):
        self.pCaimg.setImage(self.nwbfile.processing['ophys'].data_interfaces['Backgrounds_0'].images[self.CaImaging_bg_key][:]) # plotting the mean image
    elif 'CaImaging-TimeSeries' in self.nwbfile.acquisition:
        i0 = convert_time_to_index(self.time, self.nwbfile.acquisition['CaImaging-TimeSeries'])
        self.pCaimg.setImage(self.nwbfile.acquisition['CaImaging-TimeSeries'].data[i0])
        if hasattr(self, 'CaFrameLevel'):
            self.plot.removeItem(self.CaFrameLevel)
        self.CaFrameLevel = self.plot.plot(self.nwbfile.acquisition['CaImaging-TimeSeries'].timestamps[i0]*np.ones(2), [0, y.max()], pen=pen, linewidth=0.5)
        
    if ('ophys' in self.nwbfile.processing) and with_roi:
        if hasattr(self, 'ROIscatter'):
            self.pCa.removeItem(self.ROIscatter)
        self.ROIscatter = pg.ScatterPlotItem()
        X, Y = [], []
        for ir in self.roiIndices:
            indices = np.arange(self.pixel_masks_index[ir], self.pixel_masks_index[ir+1])
            X += [self.pixel_masks[ii][1] for ii in indices]
            Y += [self.pixel_masks[ii][0] for ii in indices]
        self.ROIscatter.setData(X, Y, size=1, brush=pg.mkBrush(0,255,0))
        self.pCa.addItem(self.ROIscatter)

    if ('ophys' in self.nwbfile.processing) and (self.roiIndices is not None):
        i1 = convert_time_to_index(self.tzoom[0], self.Neuropil, axis=1)
        i2 = convert_time_to_index(self.tzoom[1], self.Neuropil, axis=1)
        if self.roiPick.text()=='sum':
            y = scale_and_position(self, getattr(self, self.CaImaging_key).data[self.validROI_indices[self.roiIndices],i1:i2].sum(axis=0), i=iplot)
            nrnp = scale_and_position(self, getattr(self, self.CaImaging_key).data[self.validROI_indices[self.roiIndices],i1:i2].sum(axis=0),
                                      value=self.Neuropil.data[self.validROI_indices[self.roiIndices],i1:i2].sum(axis=0), i=iplot)
            tt = np.linspace(np.max([self.tlim[0], self.tzoom[0]]), np.min([self.tlim[1], self.tzoom[1]]), len(y)) # TEMPORARY
            self.plot.plot(tt, y, pen=pg.mkPen(color=(0,250,0), linewidth=1))
            self.plot.plot(tt, nrnp, pen=pg.mkPen(color=(255,255,255), linewidth=0.2))
        else:
            for n, ir in enumerate(self.roiIndices):
                y = scale_and_position(self, getattr(self, self.CaImaging_key).data[self.validROI_indices[ir],i1:i2], i=iplot)+n
                nrnp = scale_and_position(self, getattr(self, self.CaImaging_key).data[self.validROI_indices[ir],i1:i2],
                                          value=self.Neuropil.data[self.validROI_indices[ir],i1:i2], i=iplot)+n
                tt = np.linspace(np.max([self.tlim[0], self.tzoom[0]]), np.min([self.tlim[1], self.tzoom[1]]), len(y)) # TEMPORARY
                self.plot.plot(tt, y, pen=pg.mkPen(color=(0,250,0), linewidth=1))
                self.plot.plot(tt, nrnp, pen=pg.mkPen(color=(255,255,255), linewidth=0.2))
        iplot += 1

    # if self.CaImaging is not None:
    #     print(len(self.CaImaging.t))
    #     print(self.CaImaging.Firing.shape)
    #     cond = (self.CaImaging.t>=tzoom[0]) & (self.CaImaging.t<=tzoom[1])
    #     isampling = max([1,int(len(self.CaImaging.t[cond])/self.settings['Npoints'])])
    #     for n in range(self.CaImaging.Firing.shape[0]):
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