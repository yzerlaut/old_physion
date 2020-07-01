import imageio
import numpy as np

def export_to_mp4():
    
    tstart = time.time()
    imageio.mimwrite('test.mp4', np.array(imgs))
    if verbose:
        print('Saving time: %.1f ms ' % (1e3*(time.time()-tstart)))
    
