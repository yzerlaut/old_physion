import nidaqmx, time
import numpy as np

from nidaqmx.utils import flatten_channel_string
from nidaqmx.constants import Edge
from nidaqmx.stream_readers import (
    AnalogSingleChannelReader, AnalogMultiChannelReader)
from nidaqmx.stream_writers import (
    AnalogSingleChannelWriter, AnalogMultiChannelWriter)

import sys, os, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from hardware_control.NIdaq.config import find_x_series_devices, find_m_series_devices, get_analog_input_channels, get_analog_output_channels

def rec_only(device, t_array, inputs):

    dt = t_array[1]-t_array[0]
    sampling_rate = 1./dt
    
    # if outputs.shape[0]>0:
    input_channels = get_analog_input_channels(device)[:inputs.shape[0]]
        
    with nidaqmx.Task() as read_task,  nidaqmx.Task() as sample_clk_task:

        # Use a counter output pulse train task as the sample clock source
        # for both the AI and AO tasks.
        sample_clk_task.co_channels.add_co_pulse_chan_freq(
            '{0}/ctr0'.format(device.name), freq=sampling_rate)
        sample_clk_task.timing.cfg_implicit_timing(
            samps_per_chan=len(t_array))

        samp_clk_terminal = '/{0}/Ctr0InternalOutput'.format(device.name)

        read_task.ai_channels.add_ai_voltage_chan(
                flatten_channel_string(input_channels),
            max_val=10, min_val=-10)
        read_task.timing.cfg_samp_clk_timing(
                sampling_rate, source=samp_clk_terminal,
                active_edge=Edge.FALLING, samps_per_chan=len(t_array))

        reader = AnalogMultiChannelReader(read_task.in_stream)

        # Start the read task before starting the sample clock source task.
        read_task.start()
        sample_clk_task.start()
            
        reader.read_many_sample(
            inputs, number_of_samples_per_channel=len(t_array),
            timeout=t_array[-1]+2*dt)

def stim_and_rec(device, t_array, inputs, outputs):

    dt = t_array[1]-t_array[0]
    sampling_rate = 1./dt
    
    # if outputs.shape[0]>0:
    output_channels = get_analog_output_channels(device)[:outputs.shape[0]]
    input_channels = get_analog_input_channels(device)[:inputs.shape[0]]
        
    with nidaqmx.Task() as write_task, nidaqmx.Task() as read_task,  nidaqmx.Task() as sample_clk_task:

        # Use a counter output pulse train task as the sample clock source
        # for both the AI and AO tasks.
        sample_clk_task.co_channels.add_co_pulse_chan_freq(
            '{0}/ctr0'.format(device.name), freq=sampling_rate)
        sample_clk_task.timing.cfg_implicit_timing(
            samps_per_chan=len(t_array))

        samp_clk_terminal = '/{0}/Ctr0InternalOutput'.format(device.name)

        write_task.ao_channels.add_ao_voltage_chan(
            flatten_channel_string(output_channels),
                                   max_val=10, min_val=-10)
        write_task.timing.cfg_samp_clk_timing(
            sampling_rate, source=samp_clk_terminal,
            active_edge=Edge.RISING, samps_per_chan=len(t_array))

        read_task.ai_channels.add_ai_voltage_chan(
                flatten_channel_string(input_channels),
            max_val=10, min_val=-10)
        read_task.timing.cfg_samp_clk_timing(
                sampling_rate, source=samp_clk_terminal,
                active_edge=Edge.FALLING, samps_per_chan=len(t_array))

        writer = AnalogMultiChannelWriter(write_task.out_stream)
        reader = AnalogMultiChannelReader(read_task.in_stream)

        writer.write_many_sample(outputs)

        # Start the read and write tasks before starting the sample clock
        # source task.
        read_task.start()
        write_task.start()
        sample_clk_task.start()
            
        reader.read_many_sample(
            inputs, number_of_samples_per_channel=len(t_array),
            timeout=t_array[-1]+2*dt)
        
if __name__=='__main__':

    device = find_m_series_devices()[0]
    print(device)
    print(get_analog_output_channels(device))

    T, dt = 10, 1e-4
    t_array = np.arange(int(T/dt))*dt
    inputs = np.zeros((2,int(T/dt)))
    outputs = np.array([5e-2*np.sin(2*np.pi*t_array),
                                    2e-2*np.sin(2*np.pi*t_array)])
    print('running rec & stim [...]')
    stim_and_rec(device, t_array, inputs, outputs)
    tstart = 1e3*time.time()
    np.save('data.npy', inputs)
    print('writing T=%.1fs of recording (at f=%.2fkHz, across N=%i channels) in : %.2f ms' % (T, 1e-3/dt,inputs.shape[0],1e3*time.time()-tstart))
    print('Running 5 rec only')
    for i in range(5):
        rec_only(device, t_array, inputs)
        tstart = 1e3*time.time()
        np.save('data.npy', inputs)
        print('writing T=%.1fs of recording (at f=%.2fkHz, across N=%i channels) in : %.2f ms' % (T, 1e-3/dt,inputs.shape[0],1e3*time.time()-tstart))
