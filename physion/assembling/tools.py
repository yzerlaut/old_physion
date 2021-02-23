import os, sys, pathlib, time, datetime
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from behavioral_monitoring.locomotion import compute_position_from_binary_signals

def compute_locomotion(binary_signal, acq_freq=1e4,
                       speed_smoothing=10e-3, # s
                       t0=0):

    A = binary_signal%2
    B = np.round(binary_signal/2, 0)

    return compute_position_from_binary_signals(A, B,
                                                smoothing=int(speed_smoothing*acq_freq))


def build_subsampling_from_freq(subsampled_freq, original_freq, N, Nmin=3):
    """

    """
    if original_freq==0:
        print('  /!\ problem with original sampling freq /!\ ')
    if subsampled_freq==0:
        SUBSAMPLING = np.linspace(0, N-1, Nmin).astype(np.int)
    else:
        SUBSAMPLING = np.arange(0, N, max([int(subsampled_freq/original_freq),Nmin]))

    return SUBSAMPLING

