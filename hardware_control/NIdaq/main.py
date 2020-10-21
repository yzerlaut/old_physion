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
                 Nchannel_analog_in=2,
                 Nchannel_digital_in=0,
                 max_time=10,
                 buffer_time=0.5,
                 filename=None,
                 device=None,
                 outputs=None,
                 output_steps=[], # should be a set of dictionaries, output_steps=[{'channel':0, 'onset': 2.3, 'duration': 1., 'value':5}]
                 verbose=False):
        
        self.running = True
        self.dt = dt
        self.max_time = max_time
        self.sampling_rate = 1./self.dt
        self.buffer_size = int(buffer_time/self.dt)
        self.Nchannel_in = Nchannel_in
        self.filename = filename
        self.select_device()

        # preparing input channels
        # - analog:
        self.analog_input_channels = get_analog_input_channels(self.device)[:Nchannel_analog_in]
        self.analog_data = np.zeros((Nchannel_analog_in, 1), dtype=np.float64)
        # - digital:
        self.digital_input_channels = get_digital_input_channels(self.device)[:Nchannel_digital_in]
        self.digital_data = np.zeros((1, 1), dtype=np.uint32)
        
        # preparing output channels
        if outputs is not None: # used as a flag for output or not
            self.output_channels = get_analog_output_channels(self.device)[:outputs.shape[0]]
        elif len(output_steps)>0:
            # have to be elements 
            t = np.arange(int(self.max_time/self.dt))*self.dt
            outputs = np.zeros((1,len(t)))
            # add as many channels as necessary
            for step in output_steps:
                if step['channel']>outputs.shape[0]:
                    outputs =  np.append(outputs, np.zeros((1,len(t))), axis=0)
            for step in output_steps:
                cond = (t>step['onset']) & (t<=step['onset']+step['duration'])
                outputs[step['channel']][cond] = step['value']
            self.output_channels = get_analog_output_channels(self.device)[:outputs.shape[0]]
        self.outputs = outputs
                
            
    def launch(self):

        if self.outputs is not None:
            self.write_task = nidaqmx.Task()
            
        self.read_analog_task = nidaqmx.Task()
        self.sample_clk_task = nidaqmx.Task()

        # Use a counter output pulse train task as the sample clock source
        # for both the AI and AO tasks.
        self.sample_clk_task.co_channels.add_co_pulse_chan_freq('{0}/ctr0'.format(self.device.name),
                                                                freq=self.sampling_rate)
        self.sample_clk_task.timing.cfg_implicit_timing(samps_per_chan=int(self.max_time/self.dt))

        samp_clk_terminal = '/{0}/Ctr0InternalOutput'.format(self.device.name)

        if self.outputs is not None:
            self.write_task.ao_channels.add_ao_voltage_chan(
                flatten_channel_string(self.output_channels),
                max_val=10, min_val=-10)
            self.write_task.timing.cfg_samp_clk_timing(
                self.sampling_rate, source=samp_clk_terminal,
                active_edge=Edge.FALLING, samps_per_chan=int(self.max_time/self.dt))
        
        self.read_analog_task.ai_channels.add_ai_voltage_chan(
                flatten_channel_string(self.input_channels),
            max_val=10, min_val=-10)

        self.read_analog_task.timing.cfg_samp_clk_timing(
            self.sampling_rate, source=samp_clk_terminal,
            active_edge=Edge.FALLING, samps_per_chan=int(self.max_time/self.dt))
        
        self.reader = AnalogMultiChannelReader(self.read_analog_task.in_stream)
        self.read_analog_task.register_every_n_samples_acquired_into_buffer_event(self.buffer_size, self.reading_task_callback)

        if self.outputs is not None:
            self.writer = AnalogMultiChannelWriter(self.write_task.out_stream)
            self.writer.write_many_sample(self.outputs)

        self.read_analog_task.start() # Start the read task before starting the sample clock source task.
        if self.outputs is not None:
            self.write_task.start()
        self.sample_clk_task.start()
        if self.filename is not None:
            np.save(self.filename.replace('.npy', '.start.npy'),
                    np.ones(1)*time.time()) # saving the time stamp of the start !

    def close(self):
        try:
            self.read_analog_task.close()
        except AttributeError:
            pass
        try:
            self.write_task.close()
        except AttributeError:
            pass
        try:
            self.sample_clk_task.close()
        except AttributeError:
            pass
        if self.filename is not None:
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



    def select_device(self):
        success = False
        try:
            self.device = find_x_series_devices()[0]
            print('X-series card found:', self.device)
            success = True
        except BaseException:
            pass
        try:
            self.device = find_m_series_devices()[0]
            print('M-series card found:', self.device)
            success = True
        except BaseException:
            pass
        if not success:
            print('Neither M-series nor X-series NI DAQ card found')

        
if __name__=='__main__':
    acq = Acquisition(output_steps=[{'channel':0, 'onset': 0.3, 'duration': 1., 'value':5}])
    acq.launch()
    tstart = time.time()
    while (time.time()-tstart)<3.:
        pass
    acq.running=False
    acq.close()
    print(acq.data)
    print(acq.data.shape)
    np.save('data.npy', acq.data)
    # from datavyz import ge
    # ge.plot(acq.data[1,:][::10])
    # ge.show()
