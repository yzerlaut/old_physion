# National Instruments DAQ cards

*control of National Instruments DAQ cards for data acquisition and stimulation*

## Reference

- https://nidaqmx-python.readthedocs.io/en/latest/

But the `python` API isn't so well documented... This is a translation of the `C` API that is well documented.

There are a few example scripts:
- https://github.com/ni/nidaqmx-python/tree/master/nidaqmx_examples

But the present code is actually based on the material available in the tests:

- https://github.com/ni/nidaqmx-python/tree/master/nidaqmx/tests

In particular the file:

- https://github.com/ni/nidaqmx-python/blob/master/nidaqmx/tests/test_stream_analog_readers_writers.py