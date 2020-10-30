import numpy as np

def compute_position_from_binary_signals(A, B,
                                         perimeter_cm=25,
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

    Delta_position = np.zeros(len(A)-1, dtype=int) # N-1 elements

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

    print(np.sum(Delta_position), len(Delta_position))
    position = np.cumsum(np.concatenate([[0], Delta_position]))

    return position*perimeter_cm/cpr


if __name__=='__main__':


    import sys, os, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
    from hardware_control.NIdaq.recording import *

    import argparse
    # First a nice documentation 
    parser=argparse.ArgumentParser(description="Record data and send signals through a NI daq card",
                                   formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-os', "--output_signal", help="npy file for an array of output signal", default='')
    parser.add_argument('-Nai', "--Nchannel_analog_rec", help="Number of analog input channels to be recorded ", type=int, default=2)
    parser.add_argument('-Ndi', "--Nchannel_digital_rec", help="Number of digital input channels to be recorded ", type=int, default=4)
    parser.add_argument('-dt', "--acq_time_step", help="Temporal sampling (in s): 1/acquisition_frequency ", type=float, default=1e-4)
    parser.add_argument('-T', "--recording_time", help="Length of recording time in (s)", type=float, default=3)
    parser.add_argument('-f', "--filename", help="filename",type=str, default='data.npy')
    parser.add_argument('-d', "--device", help="device name", type=str, default='')
    args = parser.parse_args()

    if args.device=='':
        args.device = find_m_series_devices()[0]
        
    t_array = np.arange(int(args.recording_time/args.acq_time_step))*args.acq_time_step
    analog_inputs = np.zeros((args.Nchannel_analog_rec,len(t_array)))

    analog_outputs = 100*np.array([5e-2*np.sin(2*np.pi*t_array),
                                   2e-2*np.sin(2*np.pi*t_array)])
    
    print('running rec [...]')
    analog_inputs, digital_inputs = stim_and_rec(args.device, t_array, analog_inputs, analog_outputs,
                                                 args.Nchannel_digital_rec)
    print(digital_inputs[0,:])

    A = digital_inputs[0]%2
    B = np.round(digital_inputs[0]/2, 0)
    print(B)
    
    x = compute_position_from_binary_signals(A, B)
    import matplotlib.pylab as plt
    plt.plot(t_array, x)
    plt.show()
    
