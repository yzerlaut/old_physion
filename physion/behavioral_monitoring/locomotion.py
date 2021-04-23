import numpy as np
from scipy.ndimage.filters import gaussian_filter1d

def compute_position_from_binary_signals(A, B,
                                         perimeter_cm=25,
                                         smoothing=10,
                                         cpr=1000):
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

    # position = np.cumsum(np.concatenate([[0], Delta_position]))
    # return position*perimeter_cm/cpr
    
    speed = gaussian_filter1d(np.concatenate([[0], Delta_position]), smoothing)
    return -speed*perimeter_cm/cpr


def compute_locomotion(binary_signal, acq_freq=1e4,
                       speed_smoothing=10e-3, # s
                       t0=0):

    A = binary_signal%2
    B = np.round(binary_signal/2, 0)

    return compute_position_from_binary_signals(A, B,
                                                smoothing=int(speed_smoothing*acq_freq))


if __name__=='__main__':


    """
    testing the code on the setup with the NIdaq
    """
    
    import sys, os, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
    from hardware_control.NIdaq.recording import *

    import argparse
    # First a nice documentation 
    parser=argparse.ArgumentParser(description="""
    Perform the calibration for the conversion from roto-encoder signal to rotating speed
    """,
                                   formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-Nai', "--Nchannel_analog_rec", help="Number of analog input channels to be recorded ", type=int, default=1)
    parser.add_argument('-Ndi', "--Nchannel_digital_rec", help="Number of digital input channels to be recorded ", type=int, default=2)
    parser.add_argument('-dt', "--acq_time_step", help="Temporal sampling (in s): 1/acquisition_frequency ", type=float, default=1e-3)
    parser.add_argument('-T', "--recording_time", help="Length of recording time in (s)", type=float, default=5)
    args = parser.parse_args()

    device = find_m_series_devices()[0]
        
    t_array = np.arange(int(args.recording_time/args.acq_time_step))*args.acq_time_step
    analog_inputs = np.zeros((args.Nchannel_analog_rec,len(t_array)))
    analog_outputs = 100*np.array([5e-2*np.sin(2*np.pi*t_array),
                                   2e-2*np.sin(2*np.pi*t_array)])
    
    print('You have %i s to do 10 rotations of the disk [...]' % args.recording_time)
    analog_inputs, digital_inputs = stim_and_rec(device, t_array, analog_inputs, analog_outputs,
                                                 args.Nchannel_digital_rec)

    print(len(digital_inputs[0]), len(t_array))
    speed = compute_locomotion(digital_inputs[0], acq_freq=1./args.acq_time_step,
                               speed_smoothing=10e-3)
    
    import matplotlib.pylab as plt
    plt.plot(t_array, speed)
    plt.show()
    
