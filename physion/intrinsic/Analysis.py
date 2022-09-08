import os, sys, pathlib, pynwb, itertools, skimage
from scipy.ndimage.filters import gaussian_filter1d
import numpy as np
import matplotlib.pylab as plt
from matplotlib import colorbar, colors
from skimage import measure
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter

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
        if len(img.shape)==3:
            # means movie !
            return measure.block_reduce(img, block_size=(1,
                                                         Nsubsampling,
                                                         Nsubsampling), func=np.mean)

        else:
            return measure.block_reduce(img, block_size=(Nsubsampling,
                                                     Nsubsampling), func=np.mean)
    else:
        return img


def load_single_datafile(datafile):
    """
    the image data need interpolation to get regularly spaced data for FFT
    """
    io = pynwb.NWBHDF5IO(datafile, 'r')
    nwbfile = io.read()
    t, x = nwbfile.acquisition['image_timeseries'].timestamps[:],\
        nwbfile.acquisition['image_timeseries'].data[:,:,:]
    interp_func = interp1d(t, x, axis=0, kind='nearest', fill_value='extrapolate')
    real_t = nwbfile.acquisition['angle_timeseries'].timestamps[:]
    io.close()
    return real_t, interp_func(real_t)


def load_raw_data(datafolder, protocol,
                  run_id='sum'):

    params = np.load(os.path.join(datafolder, 'metadata.npy'),
                     allow_pickle=True).item()

    if run_id=='sum':
        Data = []
        for i in range(1, 15): # no more than 15 repeats...(but some can be removed, hence the "for" loop)
            if os.path.isfile(os.path.join(datafolder, '%s-%i.nwb' % (protocol, i))):
                t, data  = load_single_datafile(os.path.join(datafolder, '%s-%i.nwb' % (protocol, i)))
                Data.append(data) 
        if len(Data)>0: 
            return params, (t, np.mean(Data, axis=0)) 
        else:
            return params, (None, None)

    elif os.path.isfile(os.path.join(datafolder, '%s-%s.nwb' % (protocol, run_id))):
        return params, load_single_datafile(os.path.join(datafolder, '%s-%s.nwb' % (protocol, run_id)))


def preprocess_data(data, Facq,
                    temporal_smoothing=0,
                    spatial_smoothing=0,
                    high_pass_filtering=0):

    pData = resample_img(data, spatial_smoothing) # pre-processed data

    if high_pass_filtering>0:
        pData = butter_highpass_filter(pData-pData.mean(axis=0), high_pass_filtering, Facq, axis=0)
    if temporal_smoothing>0:
        pData = gaussian_filter1d(pData, Facq*temporal_smoothing, axis=0)
       
    return pData

def perform_fft_analysis(data, nrepeat,
                         zero_two_pi_convention=False,
                         plus_one_convention=False):
    """
    Fourier transform
        we center the phase around 0 (by shifting by pi)
    """
    spectrum = np.fft.fft(data, axis=0)

    power = np.abs(spectrum)
    phase = -np.angle(spectrum)    #   [-pi,pi] by default

    if np.mean(np.abs(phase[nrepeat,:,:]))>np.pi/2.:
        # in [0,2.pi] and then substracted by pi
        phase = (-np.angle(spectrum))%(2.*np.pi)-np.pi

    return power[nrepeat, :, :], phase[nrepeat, :, :]


def compute_maps(data):
    # compute maps
    for l, label in enumerate(['up', 'down', 'left', 'right']):
        power, phase = perform_fft_analysis(data, label)
        data[label]['power_map'] = power
        data[label]['phase_map'] = phase

    # altitude map
    data['altitude_delay_map'] = 0.5*(data['up']['phase_map']-data['down']['phase_map'])
    data['altitude_power_map'] = 0.5*(data['up']['power_map']+data['down']['power_map'])

    # azimuthal map
    data['azimuth_delay_map'] = 0.5*(data['left']['phase_map']-data['right']['phase_map'])
    data['azimuth_power_map'] = 0.5*(data['left']['power_map']+data['right']['power_map'])


def compute_retinotopic_maps(datafolder, map_type,
                             altitude_Zero_shift=10,
                             azimuth_Zero_shift=60,
                             run_id='sum',
                             verbose=True):

    if verbose:
        print('- computing retinotopic maps [...] ')

    if map_type=='altitude':
        # load raw data
        p, (t, data1) = load_raw_data(datafolder, 'up', run_id=run_id)
        p, (t, data2) = load_raw_data(datafolder, 'down', run_id=run_id)
        phase_to_angle_func = interp1d(np.linspace(-np.pi, np.pi, len(p['STIM']['up-angle'])),
                                       p['STIM']['up-angle'], kind='linear')
        
    else:
        p, (t, data1) = load_raw_data(datafolder, 'left', run_id=run_id)
        p, (t, data2) = load_raw_data(datafolder, 'right', run_id=run_id)
        phase_to_angle_func = interp1d(np.linspace(-np.pi, np.pi, len(p['STIM']['left-angle'])),
                                       p['STIM']['left-angle'], kind='linear')

    power1, phase1 = perform_fft_analysis(data1, p['Nrepeat'],
                                          zero_two_pi_convention=zero_two_pi_convention)

    power2, phase2 = perform_fft_analysis(data2, p['Nrepeat'],
                                          zero_two_pi_convention=zero_two_pi_convention)

    retinotopy = phase_to_angle_func(.5*(phase2-phase1))
        
    if verbose:
        print('-> retinotopic map calculation over ! ')

    return {'%s-power' % map_type:.5*(power1+power2),
            '%s-retinotopy' % map_type:retinotopy}


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
