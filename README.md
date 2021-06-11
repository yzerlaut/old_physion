<div><img src="https://github.com/yzerlaut/physion/raw/master/doc/physion.png" alt="physion logo" width="35%" align="right" style="margin-left: 10px"></div>

# physion -- Vision Physiology

> *Code for experimental setups and analysis pipelines to study cellular and network physiology in visual cortical circuits*

--------------------

The software is organized into several modules to perform the acquisition, the preprocessing, the visualization and the analysis of multimodal recordings (see [Documentation below](README.md#modules-and-documentation)).

### Software screenshot

<p align="center">
  <img src="doc/screenshot.jpg"/>
</p>

--------------------

## Install

1. Install a python distribution for scientific analysis:

   [get the latest Miniconda distribution](https://docs.conda.io/en/latest/miniconda.html) or [the full Anaconda distribution](https://www.anaconda.com/products/individual)
   
2. For a minimal install, run the following in the [Anaconda prompt](https://docs.anaconda.com/anaconda/user-guide/getting-started/#write-a-python-program-using-anaconda-prompt-or-terminal):

   ```
   conda install pip numpy scipy pyqtgraph matplotlib jupyter
   conda install -c conda-forge pynwb
   ```

   then either (for git users):
   ```
   git clone https://github.com/yzerlaut/physion
   ```
   or:
   ```
   pip install git+https://github.com/yzerlaut/physion
   ```

   For a complete install, see the instructions in [Performing multimodal recordings](physion/exp/README.md#full-install-for-experimental-setups)
   
## Getting started

After the installation, the program is the launched from the [Anaconda prompt](https://docs.anaconda.com/anaconda/user-guide/getting-started/#write-a-python-program-using-anaconda-prompt-or-terminal) by typing:
   ```
   python -m physion
   ```

## Modules and documentation

The different modules of the software are documented in the following links:

- [Visual stimulation](physion/visual_stim/README.md) -- relying on [PsychoPy](https://psychopy.org)
- [Performing multimodal recordings](physion/exp/README.md)
- [Electrophysiology](physion/electrophy/README.md)
- [Calcium imaging](physion/Ca_imaging/README.md) -- forked from [Suite2P](https://github.com/MouseLand/suite2p)
- [Pupil tracking](physion/pupil/README.md)
- [Behavioral monitoring](physion/behavioral_monitoring/README.md) -- adapted from [FaceMap](https://github.com/MouseLand/facemap)
- [Assembling pipeline](physion/assembling/README.md)
- [Hardware control](physion/hardware_control/README.md)
- [Visualization](physion/dataviz/README.md) -- relying on the excellent [PyQtGraph](http://pyqtgraph.org/)
- [Analysis](physion/analysis/README.md)

## Troubleshooting / Issues

Use the dedicated [Issues](https://github.com/yzerlaut/physion/issues) interface of Github.