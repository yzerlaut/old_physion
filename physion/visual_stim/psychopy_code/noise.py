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
                 square_size=10.,
                 sparseness=0.05,
                 bg_color=0.,
                 contrast=1.,
                 noise_mean_refresh_time=0.3,
                 noise_rdm_jitter_refresh_time=0.15,
                 seed=0):

        self.seed = int(seed)
    
        self.pix = screen['resolution'] # monitor.getSizePix()
        self.square_size = square_size
        self.sparseness = sparseness
        self.bg_color = bg_color
        self.contrast=contrast
        
        width_deg = 2*np.arctan(screen['width']/2./screen['distance_from_eye'])*180./np.pi
        height_deg = 2*np.arctan(screen['width']*self.pix[1]/self.pix[0]/2./screen['distance_from_eye'])*180./np.pi

        self.Nx = np.floor(width_deg/self.square_size)+1
        self.Ny = np.floor(height_deg/self.square_size)+1

        self.Ntot_square = self.Nx*self.Ny

        # an estimate of the number of shifts needed
        nshift = int(duration/noise_mean_refresh_time)+50

        # generating the events
        events = np.cumsum(np.abs(noise_mean_refresh_time+\
                                  np.random.randn(nshift)*noise_rdm_jitter_refresh_time))
        self.events = np.concatenate([[0], events[events<duration], [duration]]) # restrict to stim
        self.durations = np.diff(self.events)
        
        self.x, self.y = np.meshgrid(np.linspace(0, width_deg, int(self.pix[0])),
                                     np.linspace(0, height_deg, int(self.pix[1])), indexing='ij')

    def get_frame(self, index):

        np.random.seed(int(1000*self.seed+index)) # here, in case something else messes up with numpy seed
        
        array = np.ones((int(self.pix[0]), int(self.pix[1])))*self.bg_color
        Loc = np.random.choice(np.arange(self.Ntot_square), int(self.sparseness*self.Ntot_square), replace=False)
        Val = np.random.choice([-1, 1], int(self.sparseness*self.Ntot_square)) # either white or black
        grid_shift = np.random.uniform()*self.square_size # so that the square do not always fall on the same grid
        
        for r, v in zip(Loc, Val):
            x0, y0 = (r % self.Nx)*self.square_size, int(r / self.Nx)*self.square_size
            cond = (self.x>=(x0+grid_shift)) & (self.x<(x0+grid_shift)+self.square_size) &\
                (self.y>=(y0+grid_shift)) & (self.y<(y0+grid_shift)+self.square_size)
            array[:,:][cond] = self.contrast*v
            
        return array


class dense_noise_generator:

    def __init__(self,
                 duration=100.,
                 screen = {'name':'Lilliput',
                           'resolution':[1280, 768],
                           'width':16, # in cm
                           'distance_from_eye':15},
                 square_size=10.,
                 contrast=1.,
                 noise_mean_refresh_time=0.3,
                 noise_rdm_jitter_refresh_time=0.15,
                 seed=0):

        self.seed = int(seed)
    
        self.pix = screen['resolution'] # monitor.getSizePix()
        self.square_size = square_size
        self.contrast=contrast
        
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

        np.random.seed(int(1000*self.seed+index)) # here, in case something else messes up with numpy seed

        Loc = np.arange(int(self.Ntot_square))
        Val = np.random.choice([-1, 1], int(self.Ntot_square)) # either black or white
        
        array = np.zeros((int(self.pix[0]), int(self.pix[1])))
        
        for r, v in zip(Loc, Val):
            x0, y0 = (r % self.Nx)*self.square_size, int(r / self.Nx)*self.square_size
            cond = (self.x>=x0) & (self.x<x0+self.square_size) & (self.y>=y0) & (self.y<y0+self.square_size)
            array[:,:][cond] = self.contrast*v
            
        return array
    

        
if __name__=='__main__':

    import sys
    sys.path.append('./visual_stim/')
    from screens import SCREENS
    # stim = build_dense_noise(5)
    stim = dense_noise_generator(duration=30.,
                                  screen=SCREENS['Dell-P2018H'],
                                  square_size=5.,
                                  seed=1.0)
    import time

    from datavyz import ge
    for i in range(5):
        # time.sleep(stim.events[i])
        im = stim.get_frame(i)
        print(im.shape)
        ge.image(im)
        ge.show()
    
