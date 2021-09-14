import numpy as np
from scipy.ndimage.filters import gaussian_filter1d

def compute_position_from_binary_signals(A, B):
    '''
    Takes traces A and B and converts it to a trace that has the same number of
    points but with positions points.

    Algorithm based on the schematic of cases shown in the doc
    ---------------
    Input:
        A, B - traces to convert
   
    Output:
        Positions through time

    '''

    Delta_position = np.zeros(len(A)-1, dtype=float) # N-1 elements

    ################################
    ## positive_increment_cond #####
    ################################
    PIC = ( (A[:-1]==1) & (B[:-1]==1) & (A[1:]==0) & (B[1:]==1) ) | \
        ( (A[:-1]==0) & (B[:-1]==1) & (A[1:]==0) & (B[1:]==0) ) | \
        ( (A[:-1]==0) & (B[:-1]==0) & (A[1:]==1) & (B[1:]==0) ) | \
        ( (A[:-1]==1) & (B[:-1]==0) & (A[1:]==1) & (B[1:]==1) )
    Delta_position[PIC] = 1

    ################################
    ## negative_increment_cond #####
    ################################
    NIC = ( (A[:-1]==1) & (B[:-1]==1) & (A[1:]==1) & (B[1:]==0) ) | \
        ( (A[:-1]==1) & (B[:-1]==0) & (A[1:]==0) & (B[1:]==0) ) | \
        ( (A[:-1]==0) & (B[:-1]==0) & (A[1:]==0) & (B[1:]==1) ) | \
        ( (A[:-1]==0) & (B[:-1]==1) & (A[1:]==1) & (B[1:]==1) )
    Delta_position[NIC] = -1

    return np.cumsum(np.concatenate([[0], Delta_position]))

def compute_locomotion_speed(binary_signal, 
			     acq_freq=1e4, 
                       	     position_smoothing=10e-3, # s
			     radius_position_on_disk=1,	# cm
			     rotoencoder_value_per_rotation=1, # a.u.
                             with_raw_position=False):

    A = binary_signal%2
    B = np.round(binary_signal/2, 0)

    position = compute_position_from_binary_signals(A, B)*2.*np.pi*radius_position_on_disk/rotoencoder_value_per_rotation

    if position_smoothing>0:
        speed = np.diff(gaussian_filter1d(position, int(position_smoothing*acq_freq), mode='nearest'))
        speed[:int(2*position_smoothing*acq_freq)] = speed[int(2*position_smoothing*acq_freq)]
        speed[-int(2*position_smoothing*acq_freq):] = speed[-int(2*position_smoothing*acq_freq)]
    else:
        speed = np.diff(position)

    speed *= acq_freq

    if with_raw_position:
        return speed, position
    else:
        return speed
 	

if __name__=='__main__':

    """
    testing the code on the setup with the NIdaq
    """

    import matplotlib.pylab as plt
    import sys, os, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
    from analysis.tools import resample_signal
    
    import argparse
    # First a nice documentation 
    parser=argparse.ArgumentParser(description="""
    Perform the calibration for the conversion from roto-encoder signal to rotating speed
    """,
                                   formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-Nai', "--Nchannel_analog_rec", help="Number of analog input channels to be recorded ", type=int, default=1)
    parser.add_argument('-Ndi', "--Nchannel_digital_rec", help="Number of digital input channels to be recorded ", type=int, default=2)
    parser.add_argument('-dt', "--acq_time_step", help="Temporal sampling (in s): 1/acquisition_frequency ", type=float, default=1e-4)
    parser.add_argument('-T', "--recording_time", help="Length of recording time in (s)", type=float, default=15)
    parser.add_argument('-df', "--datafolder", type=str, default='')
    args = parser.parse_args()

    if args.datafolder!='':
        metadata = np.load(os.path.join(args.datafolder, 'metadata.npy'),
                       allow_pickle=True).item()
        NIdaq_data = np.load(os.path.join(args.datafolder, 'NIdaq.npy'), allow_pickle=True).item()
        digital_inputs = NIdaq_data['digital']
        args.acq_time_step = 1./metadata['NIdaq-acquisition-frequency']
        t_array = np.arange(len(digital_inputs[0]))*args.acq_time_step
        print('computing position [...]')
        plt.figure()
        speed = compute_locomotion_speed(digital_inputs[0],
                                         acq_freq=metadata['NIdaq-acquisition-frequency'],
                                         radius_position_on_disk=metadata['rotating-disk']['radius-position-on-disk-cm'],
                                         rotoencoder_value_per_rotation=metadata['rotating-disk']['roto-encoder-value-per-rotation'])
        t_array, speed = resample_signal(speed,
                                         original_freq=metadata['NIdaq-acquisition-frequency'],
                                         new_freq=50.,
                                         post_smoothing=2./50.,
                                         verbose=True)
        print('mean speed: %.1f cm/s' % np.mean(speed))
        plt.plot(t_array, speed)
        plt.ylabel('speed (cm/s)')
        plt.xlabel('time (s)')
        plt.show()
        
    else:
        from hardware_control.NIdaq.recording import *
        device = find_m_series_devices()[0]
        t_array = np.arange(int(args.recording_time/args.acq_time_step))*args.acq_time_step
        analog_inputs = np.zeros((args.Nchannel_analog_rec,len(t_array)))
        analog_outputs = 100*np.array([5e-2*np.sin(2*np.pi*t_array)])

        print('You have %i s to do 5 rotations of the disk [...]' % args.recording_time)
        analog_inputs, digital_inputs = stim_and_rec(device, t_array, analog_inputs, analog_outputs,
                                                     args.Nchannel_digital_rec)

        speed, position = compute_locomotion_speed(digital_inputs[0],
                                                   acq_freq=1./args.acq_time_step,
                                                   position_smoothing=100e-3,
                                                   with_raw_position=True)

        print('The roto-encoder value for a round is: ', (position[-1]-position[0])/5.,  '(N.B. evaluated over 5 rotations)')
        import matplotlib.pylab as plt
        plt.figure()
        plt.plot(t_array, position)
        plt.ylabel('travel distance (a.u.)')
        plt.xlabel('time (s)')
        plt.figure()
        plt.plot(t_array[1:], speed)
        plt.ylabel('speed (cm/s)')
        plt.xlabel('time (s)')
        plt.show()

