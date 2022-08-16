import os, sys, pathlib, pynwb, itertools, skimage
from scipy.ndimage.filters import gaussian_filter1d
import numpy as np
import matplotlib.pylab as plt
from matplotlib import colorbar, colors
from skimage import measure
from scipy.interpolate import interp1d

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from physion.analysis.analyz.analyz.processing.filters \
        import butter_highpass_filter, butter_bandpass_filter


def resample_data(array, old_time, time):
    new_array = 0*time
    for i1, i2 in zip(range(len(time)-1), range(1, len(time))):
        cond=(old_time>time[i1]) & (old_time<=time[i2])
        if len(cond)>1:
            new_array[i1] = np.mean(array[cond])
        elif len(cond)==1:
            new_array[i1] = array[cond][0]
    return new_array

def resample_img(img, Nsubsampling):
    if Nsubsampling>1:
        return measure.block_reduce(img, block_size=(Nsubsampling,
                                                     Nsubsampling), func=np.mean)
    else:
        return img


def load_single_datafile(datafile):
    io = pynwb.NWBHDF5IO(datafile, 'r')
    nwbfile = io.read()
    t, x = nwbfile.acquisition['image_timeseries'].timestamps[:],\
        nwbfile.acquisition['image_timeseries'].data[:,:,:]
    io.close()
    return t, x


def load_raw_data(datafolder, protocol,
                  run_id='sum'):
    params = np.load(os.path.join(datafolder, 'metadata.npy'), allow_pickle=True).item()

    if run_id=='sum':
        data = np.zeros((len(params['STIM']['%s-times' % protocol]), *params['imgsize']))
        n=0
        for i in range(1, 15): # no more than 15 repeats...(but some can be removed, hence the "for" loop)
            if os.path.isfile(os.path.join(datafolder, '%s-%i.nwb' % (protocol, i))):
                t, D = load_single_datafile(os.path.join(datafolder, '%s-%i.nwb' % (protocol, i)))
                data += D
                n+=1
        if n>0:
            data /= (1.0*n)
        return params, (t, data)

    elif os.path.isfile(os.path.join(datafolder, '%s-%s.nwb' % (protocol, run_id))):
        return params, load_single_datafile(os.path.join(datafolder, '%s-%s.nwb' % (protocol, run_id)))


def perform_fft_analysis(data, nrepeat,
                         high_pass_filtering=0,
                         zero_two_pi_convention=False,
                         plus_one_convention=False):

    if high_pass_filtering>0:
        spectrum = np.fft.fft(butter_highpass_filter(data, high_pass_filtering, 1., axis=0), axis=0)
    else:
        spectrum = np.fft.fft(data, axis=0)
        
    if zero_two_pi_convention:
        power, phase = np.abs(spectrum), (-np.angle(spectrum))%(2.*np.pi)
    else:
        power, phase = np.abs(spectrum), -np.angle(spectrum)
    if plus_one_convention:
        return power[nrepeat+1, :, :], phase[nrepeat+1, :, :]
    else:
        return power[nrepeat, :, :], phase[nrepeat, :, :]


def get_retinotopic_maps(datafolder, map_type,
                         altitude_Zero_shift=10,
                         azimuth_Zero_shift=60,
                         run_id='sum',
                         zero_two_pi_convention=False):

    if map_type=='altitude':
        p, (t, data1) = load_raw_data(datafolder, 'up',
                                      run_id=run_id)
        p, (t, data2) = load_raw_data(datafolder, 'down',
                                      run_id=run_id)

        if zero_two_pi_convention:
            phase_to_angle_func = interp1d(np.linspace(0, 2*np.pi, len(p['STIM']['up-angle'])),
                                           p['STIM']['up-angle'], kind='linear')
        else:
            phase_to_angle_func = interp1d(np.linspace(-np.pi, np.pi, len(p['STIM']['up-angle'])),
                                           p['STIM']['up-angle'], kind='linear')
        
    else:
        p, (t, data1) = load_raw_data(datafolder, 'left',
                                      run_id=run_id)
        p, (t, data2) = load_raw_data(datafolder, 'right',
                                      run_id=run_id)

        if zero_two_pi_convention:
            phase_to_angle_func = interp1d(np.linspace(0, 2*np.pi, len(p['STIM']['left-angle'])),
                                       p['STIM']['left-angle'], kind='linear')
        else:
            phase_to_angle_func = interp1d(np.linspace(-np.pi, np.pi, len(p['STIM']['left-angle'])),
                                       p['STIM']['left-angle'], kind='linear')

    power1, phase1 = perform_fft_analysis(data1, p['Nrepeat'],
                                          zero_two_pi_convention=zero_two_pi_convention)
    power2, phase2 = perform_fft_analysis(data2, p['Nrepeat'],
                                          zero_two_pi_convention=zero_two_pi_convention)

    retinotopy = .5*(phase2-phase1)

    if zero_two_pi_convention:
        retinotopy = np.clip(retinotopy, 0, 2*np.pi)
    else:
        retinotopy = np.clip(retinotopy, -np.pi, np.pi)
        
    return .5*(power1+power2), retinotopy

def build_trial_data(datafolder,
                     zero_two_pi_convention=False):

    altitude_power_map, altitude_delay_map = get_retinotopic_maps(datafolder, 'altitude',
                                                                  zero_two_pi_convention=zero_two_pi_convention)
    azimuth_power_map, azimuth_delay_map = get_retinotopic_maps(datafolder, 'azimuth',
                                                                zero_two_pi_convention=zero_two_pi_convention)
    
    metadata = np.load(os.path.join(datafolder, 'metadata.npy'), allow_pickle=True).item()
    vasculature_img = np.load(os.path.join(datafolder, 'vasculature.npy'),
                              allow_pickle=True)
    
    return dict(altPosMap=altitude_delay_map,
                aziPosMap=azimuth_delay_map,
                altPowerMap=altitude_power_map,
                aziPowerMap=azimuth_power_map,
                vasculatureMap=vasculature_img,
                mouseID=metadata['subject'].replace('Mouse', 'ouse'),
                dateRecorded='202'+datafolder.split('202')[1],
                comments='')
    
if __name__=='__main__':

    df = '/home/yann/DATA/2022_01_13/17-41-53/'

    altitude_power_map, altitude_phase_map = get_retinotopic_maps(df, 'altitude', zero_two_pi_convention=True)
    plt.imshow(altitude_power_map)
    plt.title('power map')
    plt.figure()
    plt.imshow(altitude_phase_map)
    plt.title('phase map')
    plt.colorbar()
    plt.show()
    
    # data = build_trial_data(df)
