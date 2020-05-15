# Tools and pipelines for cortical physiology

> Code for experimental setups and analysis pipelines to study cellular and network physiology in sensory cortices 

## How to get set up ?

#### 1) Get a scientific python distribution

Install a python distribution for scientific analysis.

We recommend the Anaconda distribution, get it here: https://www.anaconda.com/products/individual

#### 2) Download this repository

Three options:

- For `git` users & contributors (requires a github account). Fork this repository and clone your own fork.
  
- For other `git` users, clone the repository with:
  ```
  git clone https://github.com/yzerlaut/cortical-physio-icm.git
  ```

- For others, download the [zip archive](https://github.com/yzerlaut/cortical-physio-icm/archive/master.zip)

#### 3) Install dependencies (optional)

Open the Anaconda prompt (or the UNIX shell) and use `pip` to install the dependencies:
```
pip install psychopy # for visual stimulation (but see https://www.psychopy.org/download.html)
pip install neo # to load electrophysiological data
pip install git+https://github.com/yzerlaut/datavyz # for custom data visualization
```

#### 4) Open the notebooks and run the code

Open the Anaconda prompt (or the UNIX shell) and run):
```
jupyter notebook Ca-imaging/preprocessing_demo.ipynb
```
You can then run cells and play with the code in your browser.

## Analysis of Calcium imaging

#### Registration and Cell detection

Use [Suite2P](https://github.com/MouseLand/suite2p), see instructions in the documentation at [http://mouseland.github.io/suite2p](http://mouseland.github.io/suite2p)

#### Preprocessing of Calcium signals

The preprocessing step are illustrated in the [demo notebook](https://github.com/yzerlaut/cortical-physio-icm/blob/master/Ca-imaging/preprocessing_demo.ipynb)

#### Use Principal Component Analysis to find patterns of population activity

The use of PCA is shown in the [demo notebook](https://github.com/yzerlaut/cortical-physio-icm/blob/master/Ca-imaging/PCA_demo.ipynb)

## Visual stimulation
	
### Set of stimuli

[...]

### Realign physiological recordings

[...]

