import imageio, os, sys, time
import numpy as np

def export_to_mp4(folder, name, verbose=True):
    
    tstart = time.time()

    imgs = []
    # times = np.load(os.path.join(folder, 'camera-times.npy'))

    i=1
    while os.path.isfile(os.path.join(folder, 'camera-imgs', '%i.npy' % i)):
        imgs.append(np.load(os.path.join(folder, 'camera-imgs', '%i.npy' % i)))
        i+=1
        
    # for i in range(1, len(times)+1):
    #     imgs.append(np.load(os.path.join(folder, 'camera-imgs', '%i.npy' % i)))
    
    imageio.mimwrite(name, np.array(imgs))
    if verbose:
        print('Saving time: %.1f ms ' % (1e3*(time.time()-tstart)))
        
    
if len(sys.argv)==1:
    print("""
    should be used as :
       python export.py test1 video.mp4 # to generate "video.mp4" from the images of the folder named "test1"
    """)
else:
    export_to_mp4(sys.argv[1], sys.argv[2])

    
