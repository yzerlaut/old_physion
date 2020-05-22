# Tools and pipelines for cortical physiology

*Code for experimental setups and analysis pipelines to study cellular and network physiology in sensory cortices*

## Modules

- [Electrophysiology](electrophy/README.md)
- [Calcium imaging](Ca-imaging/README.md)
- [Visual stimulation](visual-stim/README.md)
- [Behavioral monitoring](behavioral-montoring/README.md)
- ...

## How do I get set up ?

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

## Troubleshooting / Issues

Use the dedicated [https://github.com/yzerlaut/cortical-physio-icm/issues](Issues) interface of Github.

