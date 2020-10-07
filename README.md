# Tools and pipelines for cortical physiology

*Code for experimental setups and analysis pipelines to study cellular and network physiology in sensory cortices*

## Modules

- [Visual stimulation](visual_stim/README.md)
- [Electrophysiology](electrophy/README.md)
- [Calcium imaging](Ca_imaging/README.md)
- [Pupil tracking](pupil/README.md)
- [Behavioral monitoring](behavioral_monitoring/README.md)
- [Hardware control](hardware_control/README.md)
- [Assembling pipeline](assembling/README.md)
- [Analysis](analysis/README.md)

## How do I get set up ?


#### 1) Get a scientific python distribution

Install a python distribution for scientific analysis, get [the latest Anaconda distribution](https://www.anaconda.com/products/individual)

#### 2) Download this repository

Either:
- For `git` users & contributors (requires a github account). Fork this repository and clone your own fork.
- For others, download the [zip archive](https://github.com/yzerlaut/cortical-physio-icm/archive/master.zip)

#### 3) Install dependencies

Open the Anaconda prompt (or the UNIX shell) and use `pip` to install the dependencies:
```
pip install -r requirements.txt
```

The installation instructions for the experimental setup are described in a dedicated README


## Running experiments

The master program for experiments is launched with:
```
python master\exp.py
```

<p align="center">
  <img src="doc/gui-master.png"/>
</p>

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

