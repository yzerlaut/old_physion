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
2. The rotoencoder input

For the two-photon microscope

2. The aperture time of the FLIR-camera (TO BE DONE, for now, using computer timestamps)
3. The aperture time of the two-photon microscope

N.B. The alignement of the webcam data (and for now FLIR-camera) are made thanks to computer timestamps in-between every frame grabbed from the camera. They are re-aligned to the recording thanks to a computer time-stamp of the NIdaq start (see `NIdaq.Tstart.npy` file in the data).

## Tracking visual-stimulation onset from the photodiode signal

Base on the photodiode signal (that is a bit noisy), we integrate it over time (after having substracted the baseline defined by the peak of the signal distribution) for each episode. We determine the onset when the integral passes a threshold. This threshold in the integral corresponds to the signal settling at the maximum for 5ms. The onset is then the time of the crossing minus those 5 seconds.
```
python assembling/fetching.py
```

## Resampling realigned data after onset-tracking

Because the time onset is not fixed from trial to trial (it it determined by the onset-tracking procedure above),there is the need to resample the data to make them comparable from trial-to-trial (e.g. to compute a trial-average).
This is performs as follows (see the implementation in (../analysis/trial_averaging.py)[../analysis/trial_averaging.py]):
We build a fixed time vector (e.g. `t = np.linspace(-1, 6, 1e3)` for a stimulus of duration 5s) and we want to interpolate the single-trial response on this time vector by looping over the onsets of the full experiment. We used the function `interp1d` of the `interpolate` module of `scipy` to build a linear interpolation of each single-trial response.
The time step of the re-sampling depended on the recording modality, we use 33ms for the Calcium-Imaging data and 10 ms for the Electrophysiological data.






