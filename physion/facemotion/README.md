# Whisker Pad Motion / Whisking

## Material and Methods

### Whisking tracking

The camera on the animal's face used for pupil monitoring was also used to detect whisking periods. We analyzed the part of the image corresponding to the left whisker pad of the mouse. The "whisking" signal was computed as follows. At the time corresponding to the frame of index $n$, we computed the motion image (the difference between frame $n+1$ and frame $n$) and we computed the mean square root of the resulting image's pixel values. A few grooming event (<1% of recording time) where the paw motion led to very high values of "whisking" signals were discarded by clipping the "whisking" signal below a threshold value corresponding to grooming events. This threshold was manually set for each recording using a custom GUI. 