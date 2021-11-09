# Pupil size



## Material and Methods

### Pupil tracking

We monitored the pupil dilation in the mouse’s  eye contralateral to the screen presentation. We used a USB camera (Flea3 FL3-U3-13Y3M, FLIR Integrated Imaging Solutions Inc.) equipped with a 10X lens (CFZOOM 13-130mm, Edmund Optics) and XX nm long-pass filter (FGL780M, Thorlabs). The camera was positioned XX cm from the left side of the mouse’s face and  was illuminated by 6 infrared LEDs. The visible light from the LCD monitor was sufficient to allow an appropriate dynamic range for state-dependent pupillary fluctuations (Neske et al., 2019). Video capture from the camera was controlled by custom-written *python* software (using the *simple_pyspin* API, see https://pypi.org/project/simple-pyspin/). Camera frames were 1280x960 pixels and acquired at a 20 Hz frame rate. To synchronize camera frames with other data acquired during the experiment, we stored digital time stamps in-between each frame acquisition. The time associated to a given frame for later analysis was then the center of the two surrounding time stamps.
We estimated the pupil diameter from the frames using state-of-the-art processing (Reimer et al., 2014; McGinley et al 2015; Vinck et al., 2014, Neske et al., 2019, Stringer et al., 2019). Briefly, we selected the eye's ROI, we introduced a saturation level to get a nearly-binary image (below saturatoin corresponding to the pupil) and we fitted a 2D ellipse on this processed image (using the *minimize* function of *scipy.optimize*, 'Nelder-Mead' method). A few frames (XX percent, mostly due to blinking) did not enable an accurate fitting of the pupil properties. Those outliers in the fitting output were identified with the following critera: the product of the fit residual and the ellipse properties should deviate by more than the product of twice their standard deviations. Those values were replaced by nearest-neighbor interpolation.

