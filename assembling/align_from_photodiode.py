    def realign_from_photodiode(self, debug=False, verbose=True):

        if verbose:
            print('---> Realigning data with respect to photodiode signal [...] ')

        if debug:
            from datavyz import ges as ge

        success = True
        
        # extract parameters
        dt = 1./self.metadata['NIdaq-acquisition-frequency']
        tlim, tnew = [0, self.Screen.photodiode.t[-1]], 0

        t0 = self.metadata['time_start'][0]
        length = self.metadata['presentation-duration']+self.metadata['presentation-interstim-period']
        npulses = int(self.metadata['presentation-duration'])
        self.metadata['time_start_realigned'] = []
        Nepisodes = np.sum(self.metadata['time_start']<tlim[1])
        for i in range(Nepisodes):
            cond = (self.Screen.photodiode.t>=t0-.3) & (self.Screen.photodiode.t<=t0+length)
            try:
                tnew, integral, threshold = find_onset_time(self.Screen.photodiode.t[cond]-t0,
                                                            self.Screen.photodiode.val[cond], npulses)
                if debug and ((i<3) or (i>Nepisodes-3)):
                    ge.plot(self.Screen.photodiode.t[cond], self.Screen.photodiode.val[cond])
                    ge.plot(self.Screen.photodiode.t[cond], Y=[integral, integral*0+threshold])
                    ge.show()
            except Exception:
                success = False # one exception is enough to make it fail
            t0+=tnew
            self.metadata['time_start_realigned'].append(t0)
            t0+=length

        if verbose:
            if success:
                print('[ok]          --> succesfully realigned')
            else:
                print('[X]          --> realignement failed')
        if success:
            self.metadata['time_start_realigned'] = np.array(self.metadata['time_start_realigned'])
            self.metadata['time_stop_realigned'] = self.metadata['time_start_realigned']+\
                self.metadata['presentation-duration']
        else:
            self.metadata['time_start_realigned'] = np.array([])
            self.metadata['time_stop_realigned'] = np.array([])
        return success


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


if __name__=='__main__':

    fn = '/home/yann/DATA/2020_10_07/16-02-19/'
    
    if sys.argv[-1]=='photodiode':

        data = np.load(os.path.join(fn, 'NIdaq.npy'))
        import matplotlib.pylab as plt
        H, bins = np.histogram(data[0,:10000], bins=50)
        baseline = bins[np.argmax(H)]
        plt.figure()
        plt.hist(data[0,:10000], bins=50)
        plt.figure()
        plt.plot(np.cumsum(data[0,:][:10000]-baseline))
        plt.figure()
        plt.plot(data[0,:][:10000])
        plt.plot(data[0,:][:10000]*0+baseline)
        # plt.plot(data['NIdaq'][0][:10000])
        plt.show()
    else:
        dataset = Dataset(fn,
                          compressed_version=False,
                          modalities=['Face', 'Pupil'])

        # print(dataset.Pupil.t)
        print(len(dataset.Pupil.t), len(dataset.Pupil.iframes), len(dataset.Pupil.index_frame_map))
        # frame = dataset.Pupil.grab_frame(30, verbose=True)
        
        # from datavyz import ges
        # ges.image(frame)
        # ges.show()
        
        # import json
        # DFFN = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'master', 'data-folder.json') # DATA-FOLDER-FILENAME
        # with open(DFFN, 'r') as fp:
        #     df = json.load(fp)['folder']
        # data = get_multimodal_dataset(last_datafile(df))
        # transform_into_realigned_episodes(data, debug=True)
        
        # transform_into_realigned_episodes(data)
        # print(len(data['time_start_realigned']), len(data['NIdaq_realigned']))

        # print('max blank time of FaceCamera: %.0f ms' % (1e3*np.max(np.diff(data['FaceCamera-times']))))
        # import matplotlib.pylab as plt
        # plt.hist(1e3*np.diff(data['FaceCamera-times']))
        # plt.show()
