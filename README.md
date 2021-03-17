<div><img src="https://github.com/yzerlaut/physion/raw/master/doc/physion.png" alt="physion logo" width="35%" align="right" style="margin-left: 10px"></div>

# physion -- Vision Physiology

> *Code for experimental setups and analysis pipelines to study cellular and network physiology in visual cortical circuits*

--------------------

<p align="center">
  <img src="docs/screenshot.jpg"/>
</p>

--------------------

## Install

1. Install a python distribution for scientific analysis:
   [get the latest Miniconda distribution](https://docs.conda.io/en/latest/miniconda.html) or [the full Anaconda distribution](https://www.anaconda.com/products/individual)
   
2. For a minimal install, run the following in the [Anaconda prompt](https://docs.anaconda.com/anaconda/user-guide/getting-started/#write-a-python-program-using-anaconda-prompt-or-terminal):
   ```
   conda install numpy 
   conda install scipy
   conda install matplotlib
   conda install pyqtgraph
   conda install xmltodict
   pip install git+https://github.com/yzerlaut/physion
   ```
   For a complete install, see the instructions in [Performing multimodal recordings](physion/exp/README.md)

## Getting started

3. The program is the launched from the [Anaconda prompt](https://docs.anaconda.com/anaconda/user-guide/getting-started/#write-a-python-program-using-anaconda-prompt-or-terminal) by typing:
   ```
   python -m physion
   ```

## Documentation and modules

The different modules of the software are documented in the following links:

- [Visual stimulation](physion/visual_stim/README.md)
- [Performing multimodal recordings](physion/exp/README.md)
- [Electrophysiology](physion/electrophy/README.md)
- [Calcium imaging](physion/Ca_imaging/README.md) -- forked from [Suite2P](https://github.com/MouseLand/suite2p)
- [Pupil tracking](physion/pupil/README.md)
- [Behavioral monitoring](physion/behavioral_monitoring/README.md) -- adapted from [FaceMap](https://github.com/MouseLand/facemap)
- [Assembling pipeline](physion/assembling/README.md)
- [Hardware control](physion/hardware_control/README.md)
- [Visualization](physion/dataviz/README.md)
- [Analysis](physion/analysis/README.md)

## How do I get set up ?

#### 1) Get a scientific python distribution


#### 2) Download this repository

Either:
- For contributors (requires a github account). Fork this repository and clone your own fork.
- For `git` users, clone the repository with `git clone https://github.com/yzerlaut/cortical-physio-icm`
- For others, download the [zip archive](https://github.com/yzerlaut/cortical-physio-icm/archive/master.zip)

#### 3) Install dependencies

Open the Anaconda prompt (or the UNIX shell) and use `pip` to install the dependencies:

```
pip install -r requirements.txt
```

Example of the full process from the Anaconda prompt:

<p align="center">
  <img src="doc/install-instructions.png"/>
</p>


## Running the program

The master program for experiments is launched from the "Anaconda prompt" by typing:
```
python -m physion
```

<p align="center">
  <img src="doc/gui-master.png"/>
</p>


You can now pick the specific module that you want to use.


It loads by default the `protocols` and `configurations` stored in [master/protocols/](master/protocols/) and [master/configs/](master/configs/) respectively. Store your protocols and recordings configurations there and you will be able to pick them from the GUI.

## Preparing protocols and configuring experiments

Go to the individual modules for the details about the settings of protocols:
- [Visual stimulation](visual_stim/README.md)
- [Electrophysiology](electrophy/README.md)
- [Calcium imaging](Ca-imaging/README.md)
- [Behavioral monitoring](behavioral_monitoring/README.md)

## Pre-processing data (alignement, pre-processing, ... )

The master program to pre-process datafiles is launched with:
```
python master\preprocessing.py
```

## Analyzing experiments

The master program to configure experiments is launched with:
```
python master\analysis.py
```

All included analysis should be implemented annd described within the [Analysis module](analysis/README.md)

We also showcase a few analysis as [Jupyter Notebooks](https://jupyter.org/) in the different modules. They can be launched by opening the Anaconda prompt (or the UNIX shell) and by running (taking the example of "Ca-imaging/preprocessing_demo.ipynb"):
```
jupyter notebook Ca-imaging/preprocessing_demo.ipynb
```
You can then run cells and play with the code in your browser.

## Troubleshooting / Issues

Use the dedicated [Issues](https://github.com/yzerlaut/cortical-physio-icm/issues) interface of Github.

