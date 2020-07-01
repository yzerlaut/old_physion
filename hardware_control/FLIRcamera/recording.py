"""

"""
import simple_pyspin, time
import numpy as np
import imageio

def rec_and_save(duration,
                 root_folder='/tmp/',
                 t0 = None,
                 # acq_freq=30, # it 
                 verbose=True):
    """
    'duration' in seconds
    """
    imgs = []
    cam = simple_pyspin.Camera()
    cam.init()
    cam.start()
    if t0 is None:
        t0 = time.time()
    t = time.time()-t0
    times = []
    while t<duration:
        imgs.append(cam.get_array())
        t=time.time()-t0
        times.append(t)
    cam.stop()
    if verbose:
        print('Effective sampling frequency: %.1f Hz ' % (1./np.mean(np.diff(times))))

    # tstart = time.time()
    # imageio.mimwrite('test.mp4', np.array(imgs))
    # np.save('test.npy', np.array(imgs))
    # if verbose:
    #     print('Saving time: %.1f ms ' % (1e3*(time.time()-tstart)))
    


if __name__=='__main__':

    rec_and_save(2, verbose=True)
    
    

