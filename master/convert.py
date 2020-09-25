import numpy as np
import skvideo.io
import os
import matplotlib.pylab as plt

# outputdata = np.random.random(size=(5, 480, 680, 3)) * 255
# outputdata = outputdata.astype(np.uint8)

folder='/media/user/DATA/18-25-27/'

# X = []
# print(len(os.listdir(os.path.join(folder, 'FaceCamera-imgs'))))
# for fn in os.listdir(os.path.join(folder, 'FaceCamera-imgs'))[:490]:
#     X.append(np.load(os.path.join(folder, 'FaceCamera-imgs', fn)))

fn = 'video.avi'

import shutil
shutil.make_archive('archive', 'zip', os.path.join(folder, 'FaceCamera-imgs'))


# plt.imshow(X[-1])
# plt.show()
    
# X = np.array(X)
# writer = skvideo.io.FFmpegWriter(fn)
# for i in range(X.shape[0]):
#     writer.writeFrame(X[i, :, :])
# writer.close()



# videodata = skvideo.io.vread(fn)
# print(np.max(X), np.max(videodata))#[:,:].mean(axis=-1))



# imageio.mimwrite('video.mp4', np.array(X), fps = [

