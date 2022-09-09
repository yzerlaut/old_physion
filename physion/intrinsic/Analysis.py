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
from physion.dataviz.datavyz.datavyz import graph_env
ge = graph_env('screen') # for display on screen

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
    else:
        print('"%s" file not found' % os.path.join(datafolder, '%s-%s.nwb' % (protocol, run_id)))


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

    # relative power w.r.t. luminance
    rel_power = np.abs(spectrum)[nrepeat, :, :]/data.shape[0]/data.mean(axis=0)
    # phase
    phase = (np.angle(spectrum)[nrepeat, :, :]+2*np.pi)%(2*np.pi) - np.pi # forced in [-pi,pi]

    return rel_power, phase


def compute_delay_power_maps(datafolder, direction,
                             maps={},
                             run_id='sum'):

    # load raw data
    p, (t, data) = load_raw_data(datafolder, direction, run_id=run_id)

    # phase to angle conversion
    if direction=='up':
        bounds = [p['STIM']['zmin'], p['STIM']['zmax']]
    elif direction=='right':
        bounds = [p['STIM']['xmin'], p['STIM']['xmax']]
    elif direction=='down':
        bounds = [p['STIM']['zmax'], p['STIM']['zmin']]
    else:
        bounds = [p['STIM']['xmax'], p['STIM']['xmin']]

    if 'vasculature' not in maps:
        maps['vasculature'] = np.load(os.path.join(datafolder, 'vasculature.npy'))

    # FFT and write maps
    maps['%s-power' % direction],\
           maps['%s-phase' % direction] = perform_fft_analysis(data, p['Nrepeat'])

    # keep phase to angle relathionship
    phase_to_angle_func = lambda x: bounds[0]+(x+np.pi)/(2*np.pi)*(bounds[1]-bounds[0]) # for [-pi,pi] interval !!
    maps['%s-phase_to_angle_func' % direction] = phase_to_angle_func
    maps['%s-angle' % direction] = phase_to_angle_func(maps['%s-phase' % direction])

    return maps
    
def compute_retinotopic_maps(datafolder, map_type,
                             maps={}, # we fill the dictionary passed as argument
                             altitude_Zero_shift=10,
                             azimuth_Zero_shift=60,
                             run_id='sum',
                             keep_maps=False,
                             verbose=True):
    """
    map type is either "altitude" or "azimuth"
    """

    if verbose:
        print('- computing "%s" retinotopic maps [...] ' % map_type)

    if map_type=='altitude':
        directions = ['up', 'down']
    else:
        directions = ['right', 'left']

    for direction in directions:
        if (('%s-power'%direction) not in maps) and not keep_maps:
            compute_delay_power_maps(datafolder, direction, maps=maps)

    if verbose:
        print('-> retinotopic map calculation over ! ')

    # build maps
    maps['%s-power' % map_type] = .5*(maps['%s-power' % directions[0]]+maps['%s-power' % directions[1]])
    maps['%s-double-delay' % map_type] = (maps['%s-phase' % directions[1]]+\
                                          maps['%s-phase' % directions[0]]+\
                                          2*np.pi)%(2*np.pi)-np.pi
    maps['%s-phase-diff' % map_type] = (maps['%s-phase' % directions[1]]-\
                                        maps['%s-phase' % directions[0]]+\
                                        2*np.pi)%(2*np.pi)-np.pi
    maps['%s-retinotopy' % map_type] = maps['%s-phase_to_angle_func' % directions[0]](\
                                                                        maps['%s-phase-diff' % map_type])

    return maps


def build_trial_data(maps):

    output = {'mouseID':'N/A', 'comments':'', 'dateRecorded':'2022-01-01'}
    for key1, key2 in zip(\
            ['vasculature', 'altitude-retinotopy', 'azimuth-retinotopy',\
                            'altitude-power', 'azimuth-power'],
            ['vasculature', 'altPos', 'aziPos', 'altPower', 'aziPower']):
        if key1 in maps:
            output[key2+'Map'] = maps[key1]
        else:
            output[key2+'Map'] = 0.*maps['vasculature']
            
    return output
    
# -------------------------------------------------------------- #
# ----------- PLOT FUNCTIONS ----------------------------------- #
# -------------------------------------------------------------- #

def plot_delay_power_maps(maps, direction):


    fig, AX = ge.figure(axes=(1,3), top=1.5, wspace=0.3, hspace=0.5, 
                        left=0.2, bottom=0.5, right=5, reshape_axes=False)

    ge.annotate(fig, '\n"%s" protocol' % direction, (0.5,1), ha='center', va='top',
                xycoords='figure fraction', size='small')

    AX[0][0].imshow(maps['%s-phase' % direction], cmap=plt.cm.brg, vmin=-np.pi, vmax=np.pi)
    ge.title(AX[0][0], 'phase map', size='xx-small')

    ge.bar_legend(AX[0][0], X=[-np.pi, 0, np.pi], label='phase (Rd)',
                  colormap=plt.cm.brg, continuous=True,
                  ticks=[-np.pi, 0, np.pi], ticks_labels=['-$\pi$', '0', '$\pi$'],
                  bounds=[-np.pi, np.pi],
                  colorbar_inset=dict(rect=[1.2,.1,.05,.8], facecolor=None))

    bounds = [np.min(maps['%s-power' % direction]),
              np.max(maps['%s-power' % direction])]

    AX[1][0].imshow(maps['%s-power' % direction], cmap=plt.cm.binary, vmin=bounds[0], vmax=bounds[1])

    ge.title(AX[1][0], 'power map', size='xx-small')

    ge.bar_legend(AX[1][0],
                  label=' rel. power \n ($10^{-4}$ a.u.)', colormap=plt.cm.binary,
                  bounds=bounds, ticks=bounds, ticks_labels=['%.1f'%(1e4*b) for b in bounds],
                  colorbar_inset=dict(rect=[1.2,.1,.05,.8], facecolor=None))

    bounds = [np.min(maps['%s-angle' % direction]),
              np.max(maps['%s-angle' % direction])]

    AX[2][0].imshow(maps['%s-angle' % direction], cmap=plt.cm.PRGn, vmin=bounds[0], vmax=bounds[1])
    ge.title(AX[2][0], 'angle/delay map', size='xx-small')

    ge.bar_legend(AX[2][0],
                  label='angle (deg.)\n visual field', colormap=plt.cm.PRGn,
                  bounds=bounds, ticks=bounds, ticks_labels=['%i'%b for b in bounds],
                  colorbar_inset=dict(rect=[1.2,.1,.05,.8], facecolor=None))

    for ax in ge.flat(AX):
        ax.axis('off')

    return fig


def plot_retinotopic_maps(maps, map_type='altitude'):
    
    if map_type=='altitude':
        plus, minus = 'up', 'down'
    else:
        plus, minus = 'left', 'right'
        
    fig, AX = ge.figure(axes=(2,3), left=0.3, top=1.5, wspace=0.3, hspace=0.5, right=3)
    
    ge.annotate(fig, '\n"%s" maps' % map_type, (0.5,1), ha='center', va='top', 
                xycoords='figure fraction', size='small')
    
    AX[0][0].imshow(maps['%s-phase' % plus], cmap=plt.cm.brg, vmin=-np.pi, vmax=np.pi)
    AX[0][1].imshow(maps['%s-phase' % minus], cmap=plt.cm.brg, vmin=-np.pi, vmax=np.pi)
    
    ge.annotate(AX[0][0], '$\phi$+', (1,1), ha='right', va='top', color='w')
    ge.annotate(AX[0][1], '$\phi$-', (1,1), ha='right', va='top', color='w')
    ge.title(AX[0][0], 'phase map: "%s"' % plus, size='xx-small')
    ge.title(AX[0][1], 'phase map: "%s"' % minus, size='xx-small')
    ge.bar_legend(AX[0][1], X=[-np.pi, 0, np.pi], label='phase (Rd)', 
                  colormap=plt.cm.brg, continuous=True,
                  ticks=[-np.pi, 0, np.pi], ticks_labels=['-$\pi$', '0', '$\pi$'],
                  bounds=[-np.pi, np.pi], 
                  colorbar_inset=dict(rect=[1.2,.1,.05,.8], facecolor=None))
    
    bounds = [np.min([maps['%s-power' % x].min() for x in [plus, minus]]),
              np.max([maps['%s-power' % x].max() for x in [plus, minus]])]

    AX[1][0].imshow(maps['%s-power' % plus], cmap=plt.cm.binary, vmin=bounds[0], vmax=bounds[1])
    AX[1][1].imshow(maps['%s-power' % minus], cmap=plt.cm.binary, vmin=bounds[0], vmax=bounds[1])
    
    ge.title(AX[1][0], 'power map: "%s"' % plus, size='xx-small')
    ge.title(AX[1][1], 'power map: "%s"' % minus, size='xx-small')
    
    ge.bar_legend(AX[1][1],
                  label=' rel. power \n ($10^{-4}$a.u./a.u.)', colormap=plt.cm.binary,
                  bounds=bounds, ticks=bounds, ticks_labels=['%.1f'%(1e4*b) for b in bounds],
                  colorbar_inset=dict(rect=[1.2,.1,.05,.8], facecolor=None))
    
    bounds = [np.min(maps['%s-retinotopy' % map_type]),
              np.max(maps['%s-retinotopy' % map_type])]
    
    AX[2][0].imshow(maps['%s-double-delay' % map_type], cmap=plt.cm.brg, vmin=-np.pi, vmax=np.pi)
    AX[2][1].imshow(maps['%s-retinotopy' % map_type], cmap=plt.cm.PRGn, vmin=bounds[0], vmax=bounds[1])
    ge.annotate(AX[2][0], '$\phi^{+}$-$\phi^{-}$', (1,1), ha='right', va='top', color='w', size='small')
    ge.annotate(AX[2][1], 'F[$\phi^{+}$+$\phi^{-}$]', (1,1), ha='right', va='top', color='w', size='xx-small')
    ge.title(AX[2][0], 'double delay map', size='xx-small')
    ge.title(AX[2][1], 'retinotopy map', size='xx-small')

    ge.bar_legend(AX[2][1],
                  label='angle (deg.)\n visual field', colormap=plt.cm.PRGn,
                  bounds=bounds, ticks=bounds, ticks_labels=['%i'%b for b in bounds],
                  colorbar_inset=dict(rect=[1.2,.1,.05,.8], facecolor=None))
    
    for ax in ge.flat(AX):
        ax.axis('off')
        
    return fig

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
