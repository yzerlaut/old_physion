import numpy as np

def from_time_to_array_index(self, t):
    if self.stimulus_params['static'] and (t<=self.t0):
         return 0
    elif self.stimulus_params['static']:
        return 1
    else:
        return int(np.round(t*self.SCREEN['refresh_rate'],0))

def sparse_noise(duration,
                 SCREEN = [800, int(800*9/16)],
                 screen_angular_width=20,
                 screen_angular_height=20*9./16.,
                 square_size=4.,
                 sparseness=0.1,
                 noise_mean_refresh_time=0.3,
                 noise_rdm_jitter_refresh_time=0.15,
                 seed=0):

                                   
    Nx = np.floor(screen_angular_width/square_size)+1
    Ny = np.floor(screen_angular_height/square_size)+1

    Ntot_square = Nx*Ny

    # an estimate of the number of shifts needed
    nshift = int(duration/noise_mean_refresh_time)+10
    events = np.cumsum(np.abs(noise_mean_refresh_time+\
                              np.random.randn(nshift)*noise_rdm_jitter_refresh_time))
    events = np.concatenate([[0], events[events<duration], [duration]]) # restrict to stim
    
    x, y = np.meshgrid(np.arange(SCREEN[0]), np.arange(SCREEN[1]), indexing='ij')
    x, y = x.flatten(), y.flatten()
    
    array = np.zeros((len(events)-1, *SCREEN))
    
    for i in range(len(events)-1):

        Loc = np.random.choice(np.arange(Ntot_square), int(sparseness*Ntot_square), replace=False)
        Val = np.random.choice([-1, 1], int(sparseness*Ntot_square)) # either white or black

        Z = 0*x
        for r, v in zip(Loc, Val):
            x0, y0 = (r % Nx)*square_size, int(r / Nx)*square_size
            cond = (x>=x0) & (x<x0+square_size) & (y>=y0) & (y<y0+square_size)
            Z[cond] = v
            
        array[i,:,:] = Z.reshape(*SCREEN)

    STIM = {'t':events,
            'array':array}
    return STIM

def dense_noise(duration,
                 SCREEN = [800, int(800*9/16)],
                 screen_angular_width=20,
                 screen_angular_height=20*9./16.,
                 square_size=4.,
                 noise_mean_refresh_time=0.3,
                 noise_rdm_jitter_refresh_time=0.15,
                 seed=0):

                                   
    Nx = np.floor(screen_angular_width/square_size)+1
    Ny = np.floor(screen_angular_height/square_size)+1

    Ntot_square = Nx*Ny

    # an estimate of the number of shifts needed
    nshift = int(duration/noise_mean_refresh_time)+10
    events = np.cumsum(np.abs(noise_mean_refresh_time+\
                              np.random.randn(nshift)*noise_rdm_jitter_refresh_time))
    events = np.concatenate([[0], events[events<duration], [duration]]) # restrict to stim
    
    x, y = np.meshgrid(np.arange(SCREEN[0]), np.arange(SCREEN[1]), indexing='ij')
    x, y = x.flatten(), y.flatten()

    array = np.zeros((len(events)-1, *SCREEN))
    
    for i in range(len(events)-1):

        Loc = np.arange(int(Ntot_square))
        Val = np.random.choice([-1, 1], int(Ntot_square)) # either black or white

        Z = 0.*x

        for r, v in zip(Loc, Val):
            x0, y0 = (r % Nx)*square_size, int(r / Nx)*square_size
            cond = (x>=x0) & (x<x0+square_size) & (y>=y0) & (y<y0+square_size)
            Z[cond] = v

        array[i,:,:] = Z.reshape(*SCREEN)

        
    STIM = {'t':events,
            'array':array}
    return STIM

if __name__=='__main__':

    
    stim = sparse_noise(5)

    
