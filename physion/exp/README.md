# Interface for experimental protocols

## Full install for experimental setups

```
pip install psychopy
```

## GUI

The interface is minimal:

<p align="center">
  <img src="../doc/exp.png"/>
</p>

- It loads the `VisualStim` protocols stored in [/exp/protocols]()
Select the desired modalities and "init" and "launch" recordings.


## Under the hood

Using separate `threads` for the different processes using the `multiprocessing` module, we interact with those threads by sending "record"/"stop" signals. The different threads are:

- The NIdaq recording

- THe visual stimulation

- The FLIR-camera
