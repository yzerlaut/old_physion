import nidaqmx, collections

from nidaqmx.constants import ProductCategory, UsageTypeAI

acq_freq = 1000. # seconds

def get_analog_input_channels(device):
    return  [c.name for c in device.ai_physical_chans]

def get_analog_output_channels(device):
    return  [c.name for c in device.ao_physical_chans]

def find_x_series_devices():
    system = nidaqmx.system.System.local()

    DEVICES = []
    for device in system.devices:
        if (not device.dev_is_simulated and
                device.product_category == ProductCategory.X_SERIES_DAQ and
                len(device.ao_physical_chans) >= 2 and
                len(device.ai_physical_chans) >= 4 and
                len(device.do_lines) >= 8 and
                (len(device.di_lines) == len(device.do_lines)) and
                len(device.ci_physical_chans) >= 4):
            DEVICES.append(device)
    return DEVICES

if __name__=='__main__':
    DEVICES = find_x_series_devices()
    device = DEVICES[0]
    print(dir(device))
    print(get_analog_input_channels(device))
