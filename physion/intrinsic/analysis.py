import os, pynwb, itertools, skimage
import numpy as np
import matplotlib.pylab as plt

def resample_data(array, old_time, time):
    new_array = 0*time
    for i1, i2 in zip(range(len(time)-1), range(1, len(time))):
        cond=(old_time>time[i1]) & (old_time<=time[i2])
        if len(cond)>1:
            new_array[i1] = np.mean(array[cond])
        elif len(cond)==1:
            new_array[i1] = array[cond][0]
    return new_array


def get_data(datafolder):
    
    data = {}

    # # determining sampling time
    for (l1, l2) in [('up', 'down'), ('left', 'right')]:
        io1 = pynwb.NWBHDF5IO(os.path.join(datafolder, '%s-1.nwb' % l1), 'r')
        io2 = pynwb.NWBHDF5IO(os.path.join(datafolder, '%s-2.nwb' % l1), 'r')
        nwbfile1, nwbfile2 = io1.read(), io2.read()
        imshape = nwbfile1.acquisition['image_timeseries'].data[0,:,:].shape
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
            data[label]['movie'][:,:,:] += nwbfile.acquisition['image_timeseries'].data[:,:,:]

            if i==1:
                data[label]['angle'] = nwbfile.acquisition['angle_timeseries'].data[:]
            i+=1
        
        if i>1:
            data[label]['movie'] /= (i-1)

    # compute the maps
    for l, label in enumerate(['up', 'down', 'left', 'right']):
        data[label]['map'] = np.argmax(data[label]['movie'], axis=0)
    
    return data


def run(datafolder,
        show=False):

    fig, AX = plt.subplots(1,4, figsize=(15,5))
    plt.subplots_adjust(right=.99, left=0.01, bottom=0.01)

    data = get_data(datafolder)

    for l, label in enumerate(['up', 'down', 'left', 'right']):
        AX[l].set_title(label, fontsize=8)
        AX[l].imshow(data[label]['map'], cmap=plt.cm.hsv)
        AX[l].axis('off')
        
    if show:
        plt.show()
        
    return fig

if __name__=='__main__':
    

    from physion.assembling.saving import day_folder, last_datafolder_in_dayfolder
    
    datafolder = last_datafolder_in_dayfolder(day_folder(os.path.join(os.path.expanduser('~'), 'DATA')),
                                              with_NIdaq=False)

    run(datafolder, show=True)

