import numpy as np
import skvideo.io
import imageio
import os, pathlib
import matplotlib.pylab as plt
from scipy.ndimage import gaussian_filter

def compress_FaceCamera(datafolder,
                        smoothing=0, # either 2, or (1, 4, 5)
                        extension='.npz', # '.avi', or '.mp4', ...
                        tool='numpy',
                        Nframe_per_file=500,
                        verbose=False):


    # create directory if not existing
    directory = os.path.join(datafolder, 'FaceCamera-compressed')
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    
    # ------------------------------------------------------------
    # building custom compression functions depending on the tool
    
    if tool=='numpy':
        
        extension='.npz' # forcing the extension in that case
        def compress_func(X, filename):
            np.savez_compressed(filename+extension, X)
            
    elif tool=='skvideo':

        if extension not in ['.avi', '.mp4']:
            print('forcing "mp4" extension')
            extension = '.mp4' 
        def compress_func(X, filename):
            writer = skvideo.io.FFmpegWriter(filename+extension)
            for i in range(X.shape[0]):
                writer.writeFrame(X[i,:,:])
            writer.close()
            
    elif tool=='imageio':

        if extension not in ['.avi', '.mp4']:
            print('forcing "mp4" extension')
            extension = '.mp4' 
        def compress_func(X, filename):
            imageio.mimwrite(filename+extension, X)

    # ------------------------------------------------
    # Now looping over frames to build the compression

    X, i0 = [], 0
    for i, fn in enumerate(os.listdir(os.path.join(datafolder, 'FaceCamera-imgs'))):
        x = np.load(os.path.join(folder, 'FaceCamera-imgs', fn))
        if smoothing!=0:
            x = gaussian_filter(x, smoothing)
        X.append(x)

        if ((i+1)%Nframe_per_file)==0:
            # we save
            filename = 'imgs-%i-%i' % (i0, i)
            compress_func(np.array(X), os.path.join(directory, filename))
            # and we reinit
            X, i0 = [], i+1
            if verbose:
                print('wrote: ', filename)
    # saving the last frames
    filename = 'imgs-%i-%i' % (i0, i)
    compress_func(np.array(X), os.path.join(directory, filename))
    if verbose:
        print('wrote: ', filename)

                
def load_compressedFaceCamera(datafolder,
                              extension='.npz'):

    directory = os.path.join(datafolder, 'FaceCamera-compressed')
    
    # videodata = skvideo.io.vread(fn)
    # print(np.max(X), np.max(videodata))#[:,:].mean(axis=-1))

    
    X = np.empty(0)
    for i, fn in enumerate(os.listdir(directory)):
        x = skvideo.io.vread(fn)
        X = np.concatenate([X, x])

    return X
    

if __name__=='__main__':
    
    folder='/media/user/DATA/18-25-27/'

    import time

    tstart = time.time()

    extension, tool = '.npz', 'numpy'
    # compress_FaceCamera(folder, extension=extension, tool=tool, smoothing=10, verbose=True)
    # print('extension:', extension,  'tool:' , tool)
    os.system('du -sh %s ' % os.path.join(folder, 'FaceCamera-compressed', 'imgs-0-499'+extension))
    print(time.time()-tstart, 'seconds')

    extension, tool = '.avi', 'skvideo'
    compress_FaceCamera(folder, extension=extension, tool=tool, smoothing=10, verbose=True)
    print('extension:', extension,  'tool:' , tool)
    os.system('du -sh %s ' % os.path.join(folder, 'FaceCamera-compressed', 'imgs-0-499'+extension))
    print(time.time()-tstart, 'seconds')
    
    extension, tool = '.avi', 'imageio'
    compress_FaceCamera(folder, extension=extension, tool=tool, smoothing=10, verbose=True)
    print('extension:', extension,  'tool:' , tool)
    os.system('du -sh %s ' % os.path.join(folder, 'FaceCamera-compressed', 'imgs-0-499'+extension))
    print(time.time()-tstart, 'seconds')
    
    extension, tool = '.mp4', 'skvideo'
    compress_FaceCamera(folder, extension=extension, tool=tool, smoothing=10, verbose=True)
    print('extension:', extension,  'tool:' , tool)
    os.system('du -sh %s ' % os.path.join(folder, 'FaceCamera-compressed', 'imgs-0-499'+extension))
    print(time.time()-tstart, 'seconds')
    
    extension, tool = '.mp4', 'imageio'
    compress_FaceCamera(folder, extension=extension, tool=tool, smoothing=10, verbose=True)
    print('extension:', extension,  'tool:' , tool)
    os.system('du -sh %s ' % os.path.join(folder, 'FaceCamera-compressed', 'imgs-0-499'+extension))
    print(time.time()-tstart, 'seconds')
        
    

    
# # outputdata = np.random.random(size=(5, 480, 680, 3)) * 255
# # outputdata = outputdata.astype(np.uint8)

# 
# cfn = os.path.join(folder, 'FaceCamera-compressed')

# X = []
# print(len(os.listdir(os.path.join(folder, 'FaceCamera-imgs'))))
# for fn in os.listdir(os.path.join(folder, 'FaceCamera-imgs'))[:49]:
#     x = np.load(os.path.join(folder, 'FaceCamera-imgs', fn))
#     # x = gaussian_filter(x, 5)
#     X.append(x)

# Y = gaussian_filter(np.array(X), (1, 3, 3))
# # Y = np.array(X)
# print(Y.dtype)

# # f = gzip.GzipFile(os.path.join(folder, "my_array.npy.gz"), "w")
# # np.save(file=f, arr=np.array(X))
# # f.close()

# np.savez_compressed(cfn+'-1', X)
# np.savez_compressed(cfn+'-2', Y)


# # import shutil
# # shutil.make_archive(os.path.join(folder, 'archive'), 'tar', os.path.join(folder, 'FaceCamera-imgs'))


# # plt.imshow(X[-1])
# # plt.show()
    
# writer1 = skvideo.io.FFmpegWriter(cfn+'-1.avi')
# writer2 = skvideo.io.FFmpegWriter(cfn+'-2.avi')
# for i in range(Y.shape[0]):
#     writer1.writeFrame(X[i])
#     writer2.writeFrame(Y[i, :, :])
# writer1.close()
# writer2.close()

# # videodata = skvideo.io.vread(fn)
# # print(np.max(X), np.max(videodata))#[:,:].mean(axis=-1))

# # imageio.mimwrite(cfn+'-1.avi', np.array(X))
# # imageio.mimwrite(cfn+'-2.avi', np.array(Y))

