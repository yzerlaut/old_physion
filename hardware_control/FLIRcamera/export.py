import imageio, os, sys, time
import numpy as np

def export_to_video(folder, name, n_frame=20000, i0=0, verbose=True):
    
    tstart = time.time()

    imgs = []
    times = np.load(os.path.join(folder, 'FaceCamera-times.npy'))
    t0 = np.load(os.path.join(folder, 'NIdaq.start.npy'))[0]
    
    i, n= i0, 0
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
    export_to_video(sys.argv[1], sys.argv[2], n_frame=1000)

    
