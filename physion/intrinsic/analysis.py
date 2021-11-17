import os, pynwb, itertools, skimage
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
        


def get_data(datafolder,
             img_subsampling=1,
             std_exclude_factor=0.5):
    
    data = {}

    # # determining sampling time
    for (l1, l2) in [('up', 'down'), ('left', 'right')]:
        io1 = pynwb.NWBHDF5IO(os.path.join(datafolder, '%s-1.nwb' % l1), 'r')
        io2 = pynwb.NWBHDF5IO(os.path.join(datafolder, '%s-1.nwb' % l2), 'r')
        nwbfile1, nwbfile2 = io1.read(), io2.read()
        imshape = resample_img(nwbfile1.acquisition['image_timeseries'].data[0,:,:], img_subsampling).shape
        t = nwbfile1.acquisition['image_timeseries'].timestamps[:]
        for l in [l1, l2]:
            data[l] = {'t':t, 'movie':np.zeros((len(t), *imshape))}
        io1.close()
        io2.close()

    # average all data across recordings
    for l, label in enumerate(['up', 'down', 'left', 'right']):
        i=1
        while os.path.isfile(os.path.join(datafolder, '%s-%i.nwb' % (label, i))):
            io = pynwb.NWBHDF5IO(os.path.join(datafolder, '%s-%i.nwb' % (label, i)), 'r')
            nwbfile = io.read()
            movie = nwbfile.acquisition['image_timeseries'].data[:,:,:]
            exclude_cond = np.abs(movie-np.mean(movie))>std_exclude_factor*np.std(movie)
            movie[exclude_cond] = np.mean(movie)
            data[label]['movie'][:,:,:] += movie

            if i==1:
                data[label]['angle'] = nwbfile.acquisition['angle_timeseries'].data[:]
            i+=1
        
        if i>1:
            data[label]['movie'] /= (i-1)

    # compute the maps
    for l, label in enumerate(['up', 'down', 'left', 'right']):
        data[label]['map'] = data[label]['angle'][np.argmin(data[label]['movie'], axis=0)]

    return data


def run(datafolder,
        
        show=False, cmap=plt.cm.brg):

    fig2, AX2 = plt.subplots(4, 1, figsize=(10,6))
    plt.subplots_adjust(right=.98, bottom=0.2, hspace=0.5)
    
    fig, AX = plt.subplots(1,4, figsize=(16,5))
    plt.subplots_adjust(right=.98, left=0.02, bottom=0.2, wspace=0.1)

    data = get_data(datafolder)

    for l, label in enumerate(['up', 'down', 'left', 'right']):
        AX[l].set_title('%s' % label, fontsize=10)
        im = AX[l].imshow(data[label]['map'], cmap=cmap,
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

        # time trace
        AX2[l].set_ylabel('%s' % label, fontsize=10)
        # AX2[l].plot(data[label]['t'], data[label]['movie'][:,int(data[label]['movie'].shape[1]/2), int(data[label]['movie'].shape[2]/2)])
        AX2[l].plot(data[label]['t'], data[label]['movie'][:,120:140, 120:140].mean(axis=(1,2)))
        
    if show:
        plt.show()
        
    return fig

if __name__=='__main__':
    

    from physion.assembling.saving import day_folder, last_datafolder_in_dayfolder
    
    datafolder = last_datafolder_in_dayfolder(day_folder(os.path.join(os.path.expanduser('~'), 'DATA')),
                                              with_NIdaq=False)

    run(datafolder, show=True)

