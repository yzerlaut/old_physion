# import numpy
# import nidaqmx


# with nidaqmx.Task() as task:
#     task.ao_channels.add_ao_voltage_chan('Dev1/ao0')

#     print('1 Channel 1 Sample Write: ')
#     print(task.write(1.0))
#     task.stop()

#     print('1 Channel N Samples Write: ')
#     print(task.write([1.1, 2.2, 3.3, 4.4, 5.5], auto_start=True))
#     task.stop()

#     task.ao_channels.add_ao_voltage_chan('Dev1/ao1')

#     print('N Channel 1 Sample Write: ')
#     print(task.write([1.1, 2.2]))
#     task.stop()

#     print('N Channel N Samples Write: ')
#     print(task.write([[1.1, 2.2, 3.3], [1.1, 2.2, 4.4]],
#                      auto_start=True))
#     task.stop()        


import PyDAQmx as nidaq


t = nidaq.Task()
t.CreateAIVoltageChan("Dev1/ai0", None, nidaq.DAQmx_Val_Diff, 0, 10, nidaq.DAQmx_Val_Volts, None)
t.CfgSampClkTiming("", 1000, nidaq.DAQmx_Val_Rising, nidaq.DAQmx_Val_FiniteSamps, 5000)
t.StartTask()


import nidaqmx

pp = pprint.PrettyPrinter(indent=4)

with nidaqmx.Task() as task:
    task.ai_channels.add_ai_voltage_chan("Dev1/ai0")

    print('1 Channel 1 Sample Read Raw: ')
    data = task.read_raw()
    pp.pprint(data)

    print('1 Channel N Samples Read Raw: ')
    data = task.read_raw(number_of_samples_per_channel=8)
    pp.pprint(data)

    task.ai_channels.add_ai_voltage_chan("Dev1/ai1:3")

    print('N Channel 1 Sample Read Raw: ')
    data = task.read_raw()
    pp.pprint(data)

    print('N Channel N Samples Read Raw: ')
    data = task.read_raw(number_of_samples_per_channel=8)
    pp.pprint(data)    
