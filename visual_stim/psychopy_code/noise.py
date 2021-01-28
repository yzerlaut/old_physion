import numpy as np

def from_time_to_array_index(self, t):
    if self.stimulus_params['static'] and (t<=self.t0):
         return 0
    elif self.stimulus_params['static']:
        return 1
    else:
        return int(np.round(t*self.SCREEN['refresh_rate'],0))

class sparse_noise_generator:

    def __init__(self,
                 duration=100.,
                 screen = {'name':'Lilliput',
                           'resolution':[1280, 768],
                           'width':16, # in cm
                           'distance_from_eye':15},
                           square_size=4.,
                           sparseness=0.1,
                           noise_mean_refresh_time=0.3,
                           noise_rdm_jitter_refresh_time=0.15,
                           seed=0):

        self.seed = seed
        np.random.seed(int(self.seed)) # setting seed
    
        self.pix = screen['resolution'] # monitor.getSizePix()
        self.square_size = square_size
        self.sparseness = sparseness
                           
        width_deg = 2*np.arctan(screen['width']/2./screen['distance_from_eye'])*180./np.pi
        height_deg = 2*np.arctan(screen['width']*self.pix[1]/self.pix[0]/2./screen['distance_from_eye'])*180./np.pi

        self.Nx = np.floor(width_deg/self.square_size)+1
        self.Ny = np.floor(height_deg/self.square_size)+1

        self.Ntot_square = self.Nx*self.Ny

        # an estimate of the number of shifts needed
        nshift = int(duration/noise_mean_refresh_time)+10

        # generating the events
        events = np.cumsum(np.abs(noise_mean_refresh_time+\
                                  np.random.randn(nshift)*noise_rdm_jitter_refresh_time))
        self.events = np.concatenate([[0], events[events<duration], [duration]]) # restrict to stim
        self.durations = np.diff(self.events)
        
        self.x, self.y = np.meshgrid(np.linspace(0, width_deg, int(self.pix[0])),
                        np.linspace(0, height_deg, int(self.pix[1])), indexing='ij')

    def get_frame(self, index):

        np.random.seed(int(self.seed+index))
        
        array = np.zeros((int(self.pix[0]), int(self.pix[1])))
        Loc = np.random.choice(np.arange(self.Ntot_square), int(self.sparseness*self.Ntot_square), replace=False)
        Val = np.random.choice([-1, 1], int(self.sparseness*self.Ntot_square)) # either white or black

        for r, v in zip(Loc, Val):
            x0, y0 = (r % self.Nx)*self.square_size, int(r / self.Nx)*self.square_size
            cond = (self.x>=x0) & (self.x<x0+self.square_size) & (self.y>=y0) & (self.y<y0+self.square_size)
            array[:,:][cond] = v
            
        return array
    


def build_dense_noise(duration,
                      monitor,
                      square_size=4.,
                      noise_mean_refresh_time=0.3,
                      noise_rdm_jitter_refresh_time=0.15,
                      seed=0):


    print('building dense noise array [...]')
    pix = monitor.getSizePix()
    width_deg = 2*np.arctan(screen['width']/2./screen['distance_from_eye'])*180./np.pi
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

    # stim = build_dense_noise(5)
    stim = sparse_noise_generator(duration=30.)
    import time

    from datavyz import ge
    for i in range(5):
        time.sleep(stim.events[i])
        ge.image(stim.get_frame(i))
        ge.show()
    
