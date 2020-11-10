# Interface for experimental protocols

## GUI

The interface is minimal:

<p align="center">
  <img src="../doc/exp.png"/>
</p>

- It loads the `VisualStim` protocols stored in [/exp/protocols](b)
Select the desired modalities and "init" and "launch" recordings.


## Machinery

Using spearate `threads` for the different processes using the `multiprocessing` module.

- The camera is launched on an independent process and we interact by sending "record"/"stop" signals.

- 