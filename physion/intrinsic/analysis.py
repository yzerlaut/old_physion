import os, pynwb, itertools, skimage
from scipy.ndimage.filters import gaussian_filter1d
import numpy as np
import matplotlib.pylab as plt
from matplotlib import colorbar, colors
from skimage import measure

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
        i=1
        while os.path.isfile(os.path.join(datafolder, '%s-%i.nwb' % (protocol, i))):
            t, D = load_single_datafile(os.path.join(datafolder, '%s-%i.nwb' % (protocol, i)))
            data += D
            i+=1
        data /= (1.0*(i-1))
        return params, (t, data)

    elif os.path.isfile(os.path.join(datafolder, '%s-%s.nwb' % (protocol, run_id))):
        return params, load_single_datafile(os.path.join(datafolder, '%s-%s.nwb' % (protocol, run_id)))


def perform_fft_analysis(data, nrepeat):

    spectrum = np.fft.fft(data, axis=0)
    power, phase = np.abs(spectrum), (-np.angle(spectrum))%(2.*np.pi)
    return power[nrepeat, :, :], phase[nrepeat, :, :]


def get_retinotopic_maps(datafolder, map_type,
                         run_id='sum'):

    if map_type=='altitude':
        p, (t, data1) = load_raw_data(datafolder, 'up',
                                      run_id=run_id)
        p, (t, data2) = load_raw_data(datafolder, 'down',
                                      run_id=run_id)

    else:
        p, (t, data1) = load_raw_data(datafolder, 'left',
                                      run_id=run_id)
        p, (t, data2) = load_raw_data(datafolder, 'right',
                                      run_id=run_id)

    power1, phase1 = perform_fft_analysis(data1, p['Nrepeat'])
    power2, phase2 = perform_fft_analysis(data2, p['Nrepeat'])
    
    return .5*(power1+power2), .5*(phase2-phase1)

            
"""    
def init_data(datafolder,
              spatial_subsampling=0):
    
    data = {}

    # # determining sampling time
    success = True
    for (l1, l2) in [('up', 'down'), ('left', 'right')]:
        try:
            io1 = pynwb.NWBHDF5IO(os.path.join(datafolder, '%s-1.nwb' % l1), 'r')
            io2 = pynwb.NWBHDF5IO(os.path.join(datafolder, '%s-1.nwb' % l2), 'r')
            nwbfile1, nwbfile2 = io1.read(), io2.read()
            imshape = resample_img(nwbfile1.acquisition['image_timeseries'].data[0,:,:], spatial_subsampling).shape
            t = nwbfile1.acquisition['image_timeseries'].timestamps[:]
            for l in [l1, l2]:
                data[l] = {'t':t, 'movie':np.zeros((len(t), *imshape))}
            data['dt'] = t[1]-t[0]
            data['imshape'] = imshape
            io1.close()
            io2.close()
        except BaseException as be:
            print(be)
            success = False

    if success:
        return data
    else:
        return None

def get_data(datafolder,
             spatial_subsampling=0,
             temporal_smoothing=0,
             exclude=[],
             std_exclude_factor=100.):

    data = init_data(datafolder, spatial_subsampling=spatial_subsampling)
    
    if data is not None:
        
        # average all data across recordings
        for l, label in enumerate(['up', 'down', 'left', 'right']):
            i=1
            while os.path.isfile(os.path.join(datafolder, '%s-%i.nwb' % (label, i))):
                include = True
                for e in exclude:
                    if e in '%s-%i.nwb' % (label, i):
                        include = False

                if include:
                    io = pynwb.NWBHDF5IO(os.path.join(datafolder, '%s-%i.nwb' % (label, i)), 'r')
                    nwbfile = io.read()
                    movie = nwbfile.acquisition['image_timeseries'].data[:,:,:]

                    for (m,n) in itertools.product(range(data['imshape'][0]), range(data['imshape'][1])):
                        # exclusion
                        exclude_cond = np.abs(movie[:,m,n]-np.mean(movie[:,m,n]))>std_exclude_factor*np.std(movie[:,m,n])
                        movie[exclude_cond,m,n] = np.mean(movie[~exclude_cond,m,n]) # replace outlier by mean value
                        # temporal smoothing
                        if temporal_smoothing>0:
                            movie[:,m,n] = gaussian_filter1d(movie[:,m,n], int(temporal_smoothing/data[l]['dt']))

                    data[label]['movie'][:,:,:] += np.array([resample_img(movie[i,:,:], spatial_subsampling) for i in range(movie.shape[0])])
                    data[label]['angle'] = nwbfile.acquisition['angle_timeseries'].data[:]
                    
                i+=1

            if i>1:
                data[label]['movie'] *= 1./(i-1)

        # compute the maps
        for l, label in enumerate(['up', 'down', 'left', 'right']):
            # finding the minimum
            data[label]['min-map'] = data[label]['angle'][np.argmin(data[label]['movie'], axis=0)]
            data[label]['abs-resp'] = np.min(data[label]['movie'], axis=0)-np.mean(data[label]['movie'], axis=0)

        
        return data
    else:
        return None


def show_raw(datafolder,
             exclude=[],
             Npixels=5):

    data = init_data(datafolder)

    fig0, AX0 = plt.subplots(1,4, figsize=(16,5))
    plt.subplots_adjust(right=.95, bottom=0.1, top=0.95)
    
    FIGS, AXS, COORDS = [], [], []
    for i in range(Npixels):
        fig, AX = plt.subplots(4, 1, figsize=(10,6))
        plt.subplots_adjust(right=.98, bottom=0.1, hspace=1, top=0.95)
        FIGS.append(fig)
        AXS.append(AX)
        COORDS.append([np.random.choice(np.arange(data['up']['movie'].shape[1]),1)[0],
                       np.random.choice(np.arange(data['up']['movie'].shape[2]),1)[0]])


    for l, label in enumerate(['up', 'down', 'left', 'right']):
        
        i=1
        while os.path.isfile(os.path.join(datafolder, '%s-%i.nwb' % (label, i))):
            include = True
            for e in exclude:
                if e in '%s-%i.nwb' % (label, i):
                    include = False

            if include:

                io = pynwb.NWBHDF5IO(os.path.join(datafolder, '%s-%i.nwb' % (label, i)), 'r')
                nwbfile = io.read()
                movie = nwbfile.acquisition['image_timeseries'].data[:,:,:]

                if i==1:
                    # mean Img per protocol
                    AX0[l].set_title('%s' % label, fontsize=10)
                    AX0[l].imshow(movie[0,:,:], cmap=plt.cm.binary,
                                  origin='lower')

                # time trace
                for p in range(Npixels):
                    AXS[p][l].plot(nwbfile.acquisition['image_timeseries'].timestamps[:],
                                   nwbfile.acquisition['image_timeseries'].data[:,COORDS[p][0], COORDS[p][1]], color=plt.cm.tab10(i-1))
                    AXS[p][l].set_ylabel(label)
                    if i==1:
                        AX0[l].annotate(' %i' % (p+1), (COORDS[p][1], COORDS[p][0]),
                                        xycoords='data', color='r')
                        AXS[p][0].set_title('loc. # %i' % (p+1))
                        AX0[l].plot([COORDS[p][1]], [COORDS[p][0]], 'ro')

                    if l==0:
                        AXS[p][l].annotate(' i=%i'%(i) +i*'\n', (0, .1),
                                           xycoords='figure fraction', color=plt.cm.tab10(i-1))

                io.close()
            i+=1

    return FIGS, AXS, fig0

        
def run(data,
        show=False,
        cmap=plt.cm.brg,
        spatial_subsampling=0,
        temporal_smoothing=0,
        exclude=[],
        std_exclude_factor=100.):

    fig, AX = plt.subplots(1,4, figsize=(16,5))
    plt.subplots_adjust(right=.98, left=0.02, bottom=0.2, wspace=0.1)

    for l, label in enumerate(['up', 'down', 'left', 'right']):
        AX[l].set_title('%s' % label, fontsize=10)
        im = AX[l].imshow(data[label]['min-map'], cmap=cmap,
                          vmin=np.min(data[label]['angle']),
                          vmax=np.max(data[label]['angle']))
        AX[l].axis('off')
        # then colorbar
        ax_cb = plt.axes([l*0.25+0.02, 0.16, 0.2, 0.03])
        bounds = np.linspace(np.min(data[label]['angle']), np.max(data[label]['angle']), 40)
        norm = colors.BoundaryNorm(bounds, cmap.N)
        cb = colorbar.ColorbarBase(ax_cb, cmap=cmap, norm=norm,
                                   orientation='horizontal')
        cb.set_ticks([10*int(np.min(data[label]['angle'])/10), 0, 10*int(np.max(data[label]['angle'])/10)])
        cb.set_label('angle (deg.)')

    fig2, AX2 = plt.subplots(3, 2, figsize=(9,9))
    plt.subplots_adjust(right=.8, left=0.1, bottom=0.1, wspace=0.1)

    AX2[0][0].set_title('horizontal') # left/right
    AX2[0][1].set_title('vertical') # up/down

    # retinotopy map
    AX2[0][0].set_ylabel('retinotopy map')
    AX2[0][0].imshow(data['up']['min-map']-data['down']['min-map'],
                    # vmin=np.min(data[label]['angle']),
                    # vmax=np.max(data[label]['angle']),
                    cmap=cmap)
    AX2[0][1].imshow(data['left']['min-map']-data['right']['min-map'],
                    # vmin=np.min(data[label]['angle']),
                    # vmax=np.max(data[label]['angle']),
                    cmap=cmap)

    # double delay map
    AX2[1][0].set_ylabel('double delay map')
    AX2[1][0].imshow(data['up']['min-map']+data['down']['min-map'],
                    cmap=cmap)
    AX2[1][1].imshow(data['left']['min-map']+data['right']['min-map'],
                    cmap=cmap)

    # response magnitude  map
    AX2[2][0].set_ylabel('resp. magnitude map')
    AX2[2][0].imshow(data['up']['abs-resp']+data['down']['abs-resp'],
                    cmap=cmap)
    AX2[2][1].imshow(data['left']['abs-resp']+data['right']['abs-resp'],
                    cmap=cmap)


    if show:
        plt.show()

    return fig
    
if __name__=='__main__':
    

    from physion.assembling.saving import day_folder, last_datafolder_in_dayfolder

    import argparse

    parser=argparse.ArgumentParser(description="Experiment interface",
                       formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-df', "--datafolder", type=str,default='')
    parser.add_argument('-e', "--exclude", type=str, nargs='*', default=[])
    parser.add_argument('-ss', "--spatial_subsampling", type=int, default=0)
    parser.add_argument('-ts', "--temporal_smoothing", type=int, default=0)
    parser.add_argument('-sef', "--std_exclude_factor", type=float, default=100.)
    parser.add_argument('-sr', "--show_raw", action="store_true")
    parser.add_argument('-gs', "--get_saved_npz", action="store_true")
    parser.add_argument('-v', "--verbose", action="store_true")
    args = parser.parse_args()

    if args.datafolder=='':
        args.datafolder = last_datafolder_in_dayfolder(day_folder(os.path.join(os.path.expanduser('~'), 'DATA')),
                                              with_NIdaq=False)

    if args.show_raw:
        show_raw(args.datafolder,
                 exclude=args.exclude)
    else:

        if args.get_saved_npz:
            data = np.load('/tmp/data.npz', allow_pickle=True).item()
        else:
            data = get_data(args.datafolder,
                            exclude=args.exclude,
                            spatial_subsampling=args.spatial_subsampling,
                            temporal_smoothing=args.temporal_smoothing,
                            std_exclude_factor=args.std_exclude_factor)
            np.save('/tmp/data.npz', data)
            print('data saved as "/tmp/data.npz" ')
        if data is not None:
            run(data,
                exclude=args.exclude,
                spatial_subsampling=args.spatial_subsampling,
                temporal_smoothing=args.temporal_smoothing,
                std_exclude_factor=args.std_exclude_factor)
    
    plt.show()
    
"""    
