import numpy as np

def from_time_to_array_index(self, t):
    if self.stimulus_params['static'] and (t<=self.t0):
         return 0
    elif self.stimulus_params['static']:
        return 1
    else:
        return int(np.round(t*self.SCREEN['refresh_rate'],0))

def build_sparse_noise(duration,
                       monitor,
                       square_size=4.,
                       sparseness=0.1,
                       noise_mean_refresh_time=0.3,
                       noise_rdm_jitter_refresh_time=0.15,
                       seed=0):

    print('building sparse noise array [...]')
    pix = monitor.getSizePix()
    width_deg = 2*np.arctan(monitor.getWidth()/2./monitor.getDistance())*180./np.pi
    height_deg = 2*np.arctan(monitor.getWidth()*pix[1]/pix[0]/2./monitor.getDistance())*180./np.pi

    Nx = np.floor(width_deg/square_size)+1
    Ny = np.floor(height_deg/square_size)+1

    Ntot_square = Nx*Ny

    # an estimate of the number of shifts needed
    nshift = int(duration/noise_mean_refresh_time)+10
    events = np.cumsum(np.abs(noise_mean_refresh_time+\
                              np.random.randn(nshift)*noise_rdm_jitter_refresh_time))
    events = np.concatenate([[0], events[events<duration], [duration]]) # restrict to stim
    
    x, y = np.meshgrid(np.linspace(0, width_deg, int(pix[0])),
                       np.linspace(0, height_deg, int(pix[1])), indexing='ij')

    array = np.zeros((len(events)-1, int(pix[0]), int(pix[1])))
    
    for i in range(len(events)-1):

        Loc = np.random.choice(np.arange(Ntot_square), int(sparseness*Ntot_square), replace=False)
        Val = np.random.choice([-1, 1], int(sparseness*Ntot_square)) # either white or black

        for r, v in zip(Loc, Val):
            x0, y0 = (r % Nx)*square_size, int(r / Nx)*square_size
            cond = (x>=x0) & (x<x0+square_size) & (y>=y0) & (y<y0+square_size)
            array[i,:,:][cond] = v

    print('[ok] sparse noise array initialized !')
    STIM = {'t':events,
            'array':array}
    return STIM

def build_dense_noise(duration,
                      monitor,
                      square_size=4.,
                      noise_mean_refresh_time=0.3,
                      noise_rdm_jitter_refresh_time=0.15,
                      seed=0):


    print('building dense noise array [...]')
    pix = monitor.getSizePix()
    width_deg = 2*np.arctan(monitor.getWidth()/2./monitor.getDistance())*180./np.pi
    height_deg = width_deg*pix[1]/pix[0]

    Nx = np.floor(width_deg/square_size)+1
    Ny = np.floor(height_deg/square_size)+1

    Ntot_square = Nx*Ny

    # an estimate of the number of shifts needed
    nshift = int(duration/noise_mean_refresh_time)+10
    events = np.cumsum(np.abs(noise_mean_refresh_time+\
                              np.random.randn(nshift)*noise_rdm_jitter_refresh_time))
    events = np.concatenate([[0], events[events<duration], [duration]]) # restrict to stim
    
    x, y = np.meshgrid(np.linspace(0, width_deg, int(pix[0])),
                       np.linspace(0, height_deg, int(pix[1])), indexing='ij')

    array = np.zeros((len(events)-1, int(pix[0]), int(pix[1])))
    
    for i in range(len(events)-1):

        Loc = np.arange(int(Ntot_square))
        Val = np.random.choice([-1, 1], int(Ntot_square)) # either black or white

        for r, v in zip(Loc, Val):
            x0, y0 = (r % Nx)*square_size, int(r / Nx)*square_size
            cond = (x>=x0) & (x<x0+square_size) & (y>=y0) & (y<y0+square_size)
            array[i,:,:][cond] = v

    STIM = {'t':events,
            'array':array}
    print('[ok] dense noise array initialized !')
    return STIM

if __name__=='__main__':

    
    stim = build_dense_noise(5)

    from datavyz import ge
    ge.image(stim['array'][0,:,:])
    ge.show()
    
