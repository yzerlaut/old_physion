# Assembling pipeline

*procedures to pre-process and assemble physiological recordings to produce multi-modal datasets*

## Purpose

The aim is to assemble the following elements:

- Presentation times of visual stimuli, see [Visual Stimulation](../visual_stim/README.md)
- Electrophysiological data, see [Electrophysiology](../electrophy/README.md)
- two-photon imaging data, see [Calcium imaging](../Ca-imaging/README.md)
- FLIR-Camera for the animal face data (pupil tracking, whisker pad movement,...),  see [Behavioral monitoring](../behavioral_monitoring/README.md)
- Webcam data for an overall view of the experimental rig over time,  see [Behavioral monitoring](../behavioral_monitoring/README.md) (for now, not stored, just real-time visualization)
- Roto-encoder data to track the movement of the animal

## Strategy

All elements send signals to the NI-daq ! We launch a (clocked !) continuous recording on the NI-daq and we realign from those signals.

The NI-daq receives:

1. The photodiode signal form the screen (taken from from the right-bottom corner of the screen)
2. The aperture time of the FLIR-camera (TO BE DONE, for now, using computer timestamps)
3. The aperture time of the two-photon microscope
4. The rotoencoder input

N.B. The alignement of the webcam data (and for now FLIR-camera) are made thanks to computer timestamps in-between every frame grabbed from the camera. They are re-aligned to the recording thanks to a computer time-stamp of the NIdaq start.

## Data

A data folder corresponding to one protocol is stored within the root data folder (e.g. `C:\\Users\yann.zerlaut\DATA\`) with as a date-time folder structure, i.e. as `C:\\Users\yann.zerlaut\DATA\2020-04-24\14_51_08\`

By running:
```
dir C:\\Users\yann.zerlaut\DATA\2020-04-24\14_51_08\
```
One can see that is made of the following elements:

```
...
```

## Code

### Finding visual-stimulation onset

Base on the photodiode signal (that is a bit noisy), we integrate it over time (after having substracted the baseline defined by the peak of the signal distribution) for each episode. We determine the onset when the integral passes a threshold. This threshold in the integral corresponds to the signal settling at the maximum for 5ms. The onset is then the time of the crossing minus those 5 seconds.
```
python assembling/fetching.py
```

### Other assembling procedures

The scripts doing the assembling is in the file [fetching.py](./fetching.py).





