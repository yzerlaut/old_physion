# Visual stimulation

## Configuration/Installation

Go through the steps of the [README](../README.md) of the repository.

First install a python distribution from: https://www.anaconda.com/products/individual

The stimulus presentation relies on the [PsychoPy module](https://www.psychopy.org).  The custom code for the GUI and the set of stimulus lies in the present repository: ["psychopy_code" folder](./psychopy_code/) and ["gui" folder](./gui/).

A few examples, that were used to design and calibrate the stimuli, can be found in the [psychopy_code/demo_stim.py](./psychopy_code/demo_stim.py)

## Running the visual stimulation program

Open the Anaconda prompt and run:

```
python visual_stim\gui\main.py
```

There is a `"demo"` mode to adjust and build the protocols.

<p align="center">
  <img src="../doc/gui-visual-stim.png"/>
</p>


## Screen settings and calibration

[...] to be done

The visual space covered by the screen is described by the following parameters:
- Angular height of the visual field
- Angular width of the visual field
- Field center from antero-posterior axis

## Set of stimuli

The set of stimuli implemented can be visualized in the GUI (with the parameters of each stimulus type).

They are documented in the [file of default parameter](./default_params.py).

## Realign physiological recordings

see the [Assembling module](../assembling/README.md)

