# Assembling pipeline

*procedures to pre-process and assemble multi-modal recordings to produce a coherent physiological dataset*

## Purpose

The aim is to assemble the following elements:

- Electrophysiological data, see [Electrophysiology](electrophy/README.md)
- two-photon imaging data, see [Calcium imaging](Ca-imaging/README.md)
- FLIR-Camera data,  see [Behavioral monitoring](behavioral-montoring/README.md)

## Strategy

All elements send signals to the NI-daq ! We launch a continuous recording (clocked !) on the NI-daq and we realign from those signals.

The NI-daq receives:

1. The photodiode signal (taken from from the right-bottom corner of the screen)
2. The aperture time of the FLIR-camera
3. The aperture time of the two-photon microscope

## Code

The script doing the assembly is []





