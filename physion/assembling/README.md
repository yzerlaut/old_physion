# Assembling pipeline

*procedures to pre-process and assemble physiological recordings to produce multi-modal datasets*

## Purpose

The aim is to assemble the following elements:

- Presentation times of visual stimuli, see [Visual Stimulation](../visual_stim/README.md)
- Electrophysiological data, see [Electrophysiology](../electrophy/README.md)
- two-photon imaging data, see [Calcium imaging](../Ca-imaging/README.md)
- FLIR-Camera for the animal face data (pupil tracking, whisker pad movement,...),  see [Behavioral monitoring](../behavioral_monitoring/README.md)
- Roto-encoder data to track the movement of the animal
- [+Possibly] Webcam data for an overall view of the experimental rig over time, see [Behavioral monitoring](../behavioral_monitoring/README.md) (not stored now, just real-time visualization)

## Strategy

All elements send signals to the NI-daq ! We launch a (clocked !) continuous recording on the NI-daq and we realign from those signals.

The NI-daq receives:

1. The photodiode signal form the screen (taken from from the right-bottom corner of the screen)
2. The rotoencoder input
3. The electrophysiological signal

Those quantities are therefore naturally synchronized to the NIdaq.

For the Two-Photon data and the FLIR camera, we use computer time stamps:
- For the two-photon data, we use the "relativeTime" (with respect to the NIdaq-triggered recording) available from the `Prairie` software in the xml metadata of a "TSerie", see the example [Bruker-xml file](../Ca_imagingBruker_xml/TSeries-190620-250-00-002.xml) and the [Bruker-xml parser script](./IO/bruker_xml_parser.py)
- For the FLIR Camera, we generate computer time stamps in between every frame acquisition and we associate to a frame the time in the middle of those two time stamps, see our [layer on top of the FLIR API](../hardware_control/FLIRcamera/recording.py) generating the `FaceCamera-times.npy` file in the data folders. Those digital time stamps are realigned to the NIdaq time thanks to a computer time-stamp generated at the NIdaq start (see `NIdaq.Tstart.npy` file in the datafolder).

## Assembling Steps

In absence of Calcium Imaging, one can jump directly to step 4.


1. We match the calcium imaging data to the NIdaq datafiles and move them to the same directory.
   Let's say that both data are located in `/media/yann/DATADRIVE/` (the script will loop recursively), we run:
   ```
   python assembling/move_CaImaging_folders.py --root_datafolder_Visual /media/yann/DATADRIVE/ --root_datafolder_Calcium /media/yann/DATADRIVE/ --with_transfer
   ```

2. we pre-process the Calcium imaging data
   ```
   python Ca_imaging/preprocessing.py --root_datafolder /media/yann/DATADRIVE/
   ```

3. We manually check the Calcium data 

4. We build a multimodal `NWB` file for each recording.
   For a given recording:
   ```
   python assembling/build_NWB.py /media/yann/DATADRIVE/2020_12_11/ FULL --recursive
   ```
   ```
   python assembling/build_NWB.py /media/yann/DATADRIVE/2020_12_11 FULL --recursive
   ```

Note that steps 1. and 2. are interchangeable as the analysis files will be move together with the raw datafiles.



## Tracking visual-stimulation onset from the photodiode signal

Base on the photodiode signal (that is a bit noisy), we integrate it over time (after having substracted the baseline defined by the peak of the signal distribution) for each episode. We determine the onset when the integral passes a threshold. This threshold in the integral corresponds to the signal settling at the maximum for 5ms. The onset is then the time of the crossing minus those 5 seconds.

The [script performing the realignement](./realign_from_photodiode.py) can be test with:
```
python assembling/realign_from_photodiode.py EXAMPLE_DATAFOLDER
```
it shows up a few of the episodes with the quantities used for the realignement, such as:

<p align="center">
  <img src="../docs/realignement-from-photodiode.png"/>
</p>


This was obtained with: `python assembling/realign_from_photodiode.py /media/yann/Yann/2020_11_10/16-59-49/`


## Resampling realigned data after onset-tracking

Because the time onset is not fixed from trial to trial (it it determined by the onset-tracking procedure above),there is the need to resample the data to make them comparable from trial-to-trial (e.g. to compute a trial-average).
This is performs as follows (see the implementation in (../analysis/trial_averaging.py)[../analysis/trial_averaging.py]):
We build a fixed time vector (e.g. `t = np.linspace(-1, 6, 1e3)` for a stimulus of duration 5s) and we want to interpolate the single-trial response on this time vector by looping over the onsets of the full experiment. We used the function `interp1d` of the `interpolate` module of `scipy` to build a linear interpolation of each single-trial response.
The time step of the re-sampling depended on the recording modality, we use 33ms for the Calcium-Imaging data and 10 ms for the Electrophysiological data.






