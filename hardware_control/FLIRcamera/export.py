import imageio, os, sys, time
import numpy as np

def export_to_mp4(folder, name, n_frame=20000, verbose=True):
    
    tstart = time.time()

    imgs = []
    # times = np.load(os.path.join(folder, 'camera-times.npy'))

    i, n= 1, 0
    while os.path.isfile(os.path.join(folder, 'FaceCamera-imgs', '%i.npy' % i)) and n<n_frame:
        imgs.append(np.load(os.path.join(folder, 'FaceCamera-imgs', '%i.npy' % i)))
        i+=1
        n+=1
        
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

    
