import numpy as np
import os

def realign_from_photodiode(signal, metadata,
                            debug=False, verbose=True):

    if verbose:
        print('---> Realigning data with respect to photodiode signal [...] ')

    success = True

    
    # extract parameters
    dt = 1./metadata['NIdaq-acquisition-frequency']
    t = np.arange(len(signal))*dt
    
    tlim, tnew = [0, t[-1]], 0

    t0 = metadata['time_start'][0]
    length = metadata['presentation-duration']+metadata['presentation-interstim-period']
    npulses = int(metadata['presentation-duration'])
    metadata['time_start_realigned'] = []
    Nepisodes = np.sum(metadata['time_start']<tlim[1])
    for i in range(Nepisodes):
        cond = (t>=t0-.3) & (t<=t0+length)
        try:
            tnew, integral, threshold = find_onset_time(t[cond]-t0,
                                                        signal[cond], npulses)
            if debug and ((i<3) or (i>Nepisodes-3)):
                fig, ax = plt.subplots()

                ax.plot(t[cond], normalize_signal(signal[cond])[0], label='photodiode-signal')
                norm_integral, norm, xmin = normalize_signal(integral)
                ax.plot(t[cond], norm_integral, label='integral')
                ax.plot(t[cond], integral*0+norm*(threshold-xmin), label='threshold')
                ax.plot([t0,t0,], [0,1], 'k:', label='onset')
                plt.xlabel('time (s)')
                plt.ylabel('norm. signals')
                ax.legend(frameon=False)
                plt.show()
        except Exception:
            success = False # one exception is enough to make it fail
        t0+=tnew
        metadata['time_start_realigned'].append(t0)
        t0+=length

    if verbose:
        if success:
            print('[ok]          --> succesfully realigned')
        else:
            print('[X]          --> realignement failed')
    if success:
        metadata['time_start_realigned'] = np.array(metadata['time_start_realigned'])
        metadata['time_stop_realigned'] = metadata['time_start_realigned']+\
            metadata['presentation-duration']
    else:
        metadata['time_start_realigned'] = np.array([])
        metadata['time_stop_realigned'] = np.array([])
    return success, metadata


def find_onset_time(t, photodiode_signal, npulses,
                    time_for_threshold=10e-3):
    """
    the threshold of integral increase corresponds to spending X-ms at half the maximum
    """
    H, bins = np.histogram(photodiode_signal, bins=100)
    baseline = bins[np.argmax(H)]

    integral = np.cumsum(photodiode_signal-baseline)*(t[1]-t[0])

    threshold = time_for_threshold*np.max(photodiode_signal)
    t0 = t[np.argwhere(integral>threshold)[0][0]]
    return t0-time_for_threshold, integral, threshold

def normalize_signal(x):
    # just to plot above
    norm, xmin = 1./(np.max(x)-np.min(x)), np.min(x)
    return norm*(x-xmin), norm, xmin

if __name__=='__main__':

    import matplotlib.pylab as plt
    
    fn = '/media/yann/Yann/2020_11_10/16-59-49/'
    
    data = np.load(os.path.join(fn, 'NIdaq.npy'), allow_pickle=True).item()['analog'][0]
    metadata = np.load(os.path.join(fn, 'metadata.npy'), allow_pickle=True).item()
    VisualStim = np.load(os.path.join(fn, 'visual-stim.npy'), allow_pickle=True).item()
    for key in ['time_start', 'time_stop']:
        metadata[key] = VisualStim[key]

    realign_from_photodiode(data, metadata, debug=True)
    
    # print(data)
    # H, bins = np.histogram(data[:10000], bins=50)
    # baseline = bins[np.argmax(H)]
    # plt.figure()
    # plt.hist(data[:10000], bins=50)
    # plt.figure()
    # plt.plot(np.cumsum(data[:10000]-baseline))
    # plt.figure()
    # plt.plot(data[:10000])
    # plt.plot(data[:10000]*0+baseline)
    # # plt.plot(data['NIdaq'][0][:10000])
    # plt.show()
    # else:
    #     dataset = Dataset(fn,
    #                       compressed_version=False,
    #                       modalities=['Face', 'Pupil'])

    #     # print(dataset.Pupil.t)
    #     print(len(dataset.Pupil.t), len(dataset.Pupil.iframes), len(dataset.Pupil.index_frame_map))
    #     # frame = dataset.Pupil.grab_frame(30, verbose=True)
        
    #     # from datavyz import ges
    #     # ges.image(frame)
    #     # ges.show()
        
    #     # import json
    #     # DFFN = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'master', 'data-folder.json') # DATA-FOLDER-FILENAME
    #     # with open(DFFN, 'r') as fp:
    #     #     df = json.load(fp)['folder']
    #     # data = get_multimodal_dataset(last_datafile(df))
    #     # transform_into_realigned_episodes(data, debug=True)
        
    #     # transform_into_realigned_episodes(data)
    #     # print(len(data['time_start_realigned']), len(data['NIdaq_realigned']))

    #     # print('max blank time of FaceCamera: %.0f ms' % (1e3*np.max(np.diff(data['FaceCamera-times']))))
    #     # import matplotlib.pylab as plt
    #     # plt.hist(1e3*np.diff(data['FaceCamera-times']))
    #     # plt.show()
