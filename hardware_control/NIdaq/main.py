import nidaqmx, time
import numpy as np

from nidaqmx.utils import flatten_channel_string
from nidaqmx.constants import Edge, WAIT_INFINITELY
from nidaqmx.stream_readers import (
    AnalogSingleChannelReader, AnalogMultiChannelReader)
from nidaqmx.stream_writers import (
    AnalogSingleChannelWriter, AnalogMultiChannelWriter)

import sys, os, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from hardware_control.NIdaq.config import find_x_series_devices, find_m_series_devices, get_analog_input_channels, get_analog_output_channels

class Acquisition:

    def __init__(self,
                 dt=1e-3,
                 Nchannel_in=2,
                 max_time=10,
                 buffer_time=0.5,
                 filename='data.npy',
                 device=None,
                 verbose=False, ):

        self.running = True
        self.dt = dt
        self.max_time = max_time
        self.sampling_rate = 1./self.dt
        self.buffer_size = int(buffer_time/self.dt)
        self.Nchannel_in = Nchannel_in
        self.data = np.zeros((Nchannel_in, 1), dtype=np.float64)
        self.filename = filename
        
        try:
            self.device = find_x_series_devices()[0]
        except BaseException:
            pass
            if verbose:
                print('No X-series DAQ card found')
        try:
            self.device = find_m_series_devices()[0]
        except BaseException:
            if verbose:
                print('No M-series DAQ card found')

        self.input_channels = get_analog_input_channels(self.device)[:Nchannel_in]
        
    def launch(self):

        self.read_task = nidaqmx.Task()
        self.sample_clk_task = nidaqmx.Task()

        # Use a counter output pulse train task as the sample clock source
        # for both the AI and AO tasks.
        self.sample_clk_task.co_channels.add_co_pulse_chan_freq('{0}/ctr0'.format(self.device.name),
                                                                freq=self.sampling_rate)
        self.sample_clk_task.timing.cfg_implicit_timing(samps_per_chan=int(self.max_time/self.dt))

        samp_clk_terminal = '/{0}/Ctr0InternalOutput'.format(self.device.name)

        self.read_task.ai_channels.add_ai_voltage_chan(
                flatten_channel_string(self.input_channels),
            max_val=10, min_val=-10)

        self.read_task.timing.cfg_samp_clk_timing(
                self.sampling_rate, source=samp_clk_terminal,
                active_edge=Edge.FALLING, samps_per_chan=int(self.max_time/self.dt))

        self.reader = AnalogMultiChannelReader(self.read_task.in_stream)

        self.read_task.register_every_n_samples_acquired_into_buffer_event(self.buffer_size, self.reading_task_callback)
        
        self.read_task.start() # Start the read task before starting the sample clock source task.
        self.sample_clk_task.start()

    def close(self):
        self.read_task.close()
        self.sample_clk_task.close()
        np.save(self.filename, self.data[:,1:])
        print('NIdaq data saved as: %s ' % self.filename)

    def reading_task_callback(self, task_idx, event_type, num_samples, callback_data=None):
        if self.running:
            buffer = np.zeros((self.Nchannel_in, num_samples), dtype=np.float64)
            self.reader.read_many_sample(buffer, num_samples, timeout=WAIT_INFINITELY)
            self.data = np.append(self.data, buffer, axis=1)
        else:
            self.close()
        return 0  # Absolutely needed for this callback to be well defined (see nidaqmx doc).

        
if __name__=='__main__':
    acq = Acquisition()
    acq.launch()
    tstart = time.time()
    while (time.time()-tstart)<3.:
        pass
    acq.running=False
    acq.close()
    print(acq.data)
    print(acq.data.shape)
    np.save('data.npy', acq.data)
