import numpy as np


def from_time_to_array_index(self, t):
    if self.stimulus_params['static'] and (t<=self.t0):
         return 0
    elif self.stimulus_params['static']:
        return 1
    else:
        return int(np.round(t*self.SCREEN['refresh_rate'],0))

def sparse_noise(SCREEN, duration,
                 noise_refresh_rate,
                 SN_square_size=4.,
                 SN_sparseness=0.1,
                 SN_noise_mean_refresh_time=0.3,
                 SN_noise_rdm_jitter_refresh_time=0.3):

                                   
    Nx = np.floor(SCREEN['width']/SN_square_size)+1
    Ny = np.floor(SCREEN['height']/SN_square_size)+1

    Ntot_square = Nx*Ny

    # an estimate of the number of shifts needed
    nshift = int(duration/SN_noise_mean_refresh_time)+10
    events = np.cumsum(np.abs(SN_noise_mean_refresh_time+\
                              np.random.randn(nshift)*SN_noise_rdm_jitter_refresh_time))
    events = np.concatenate([[0], events[events<duration], [duration]]) # restrict to stim
    
    x, y = SCREEN['x_2d'], SCREEN['y_2d']
    for t1, t2 in zip(events[:-1], events[1:]):

        Loc = np.random.choice(np.arange(Ntot_square), int(SN_sparseness*Ntot_square), replace=False)
        Val = np.random.choice([0, 1], int(SN_sparseness*Ntot_square))

        Z = 0.5+0.*x

        for r, v in zip(Loc, Val):
            x0, y0 = (r % Nx)*SN_square_size, int(r / Nx)*SN_square_size
            cond = (x>=x0) & (x<x0+SN_square_size) & (y>=y0) & (y<y0+SN_square_size)
            Z[cond] = v

        it1 = int(np.round(t1*SCREEN['refresh_rate'],0))
        it2 = int(np.round(t2*SCREEN['refresh_rate'],0))

        self.full_array[it1:it2,:,:] = Z

    t = np.arange(int(duration*noise_refresh_rate))/noise_refresh_rate
    
    STIM = {'t':t, 'dt':1/noise_refresh_rate,
            'x-vem':np.zeros(len(t)),
            'y-vem':np.zeros(len(t)),
            'array':np.zeros(len(t), )


def dense_noise(SCREEN,
                DN_square_size=4.,
                DN_noise_mean_refresh_time=0.5,
                DN_noise_rdm_jitter_refresh_time=0.2):

    Nx = np.floor(self.SCREEN['width']/DN_square_size)+1
    Ny = np.floor(self.SCREEN['height']/DN_square_size)+1

    Ntot_square = Nx*Ny
    nshift = int((self.tstop-self.t0)/DN_noise_mean_refresh_time)+10
    events = np.cumsum(np.abs(DN_noise_mean_refresh_time+\
                              np.random.randn(nshift)*DN_noise_rdm_jitter_refresh_time))
    events = np.concatenate([[self.t0], self.t0+events[events<self.tstop], [self.tstop]]) # restrict to stim

    x, y = self.SCREEN['x_2d'], self.SCREEN['y_2d']
    for t1, t2 in zip(events[:-1], events[1:]):

        Loc = np.arange(int(Ntot_square))
        Val = np.random.choice([0, 1], int(Ntot_square))

        Z = 0.5+0.*x

        for r, v in zip(Loc, Val):
            x0, y0 = (r % Nx)*DN_square_size, int(r / Nx)*DN_square_size
            cond = (x>=x0) & (x<x0+DN_square_size) & (y>=y0) & (y<y0+DN_square_size)
            Z[cond] = v

        it1 = self.from_time_to_array_index(t1)
        it2 = self.from_time_to_array_index(t2)

        self.full_array[it1:it2,:,:] = Z


