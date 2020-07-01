"""
The camera needs to be configured in the SpinView software
"""
import simple_pyspin, time, os
import numpy as np
from pathlib import Path

class CameraAcquisition:

    def __init__(self, folder='./'):
        self.times, self.frame_index = [], 0
        self.folder = folder
        self.imgs_folder = os.path.join(self.folder, 'camera-imgs')
        Path(self.imgs_folder).mkdir(parents=True, exist_ok=True)
        self.cam = simple_pyspin.Camera()
        self.cam.init()
        self.batch_index = 0
        self.running=True
        
    def rec(self, duration, t0=None):
        if t0 is None:
            self.t0 = time.time()
        else:
            self.t0 = t0
        self.cam.start()
        self.t = time.time()-self.t0
        while self.running and (self.t<duration):
            self.frame_index +=1
            np.save(os.path.join(self.imgs_folder, '%i.npy' % self.frame_index), np.array(self.cam.get_array()))
            self.t=time.time()-self.t0
            self.times.append(self.t)
            
        if self.t>=duration:
            self.stop()
        elif not self.running:
            print('acquisition stopped !')
            self.stop()
            
    def save_times(self, verbose=True):
        print('Camera data saved as: ', os.path.join(self.folder, 'camera-times.npy'))
        np.save(os.path.join(self.folder, 'camera-times.npy'), np.array(self.times))
        if verbose:
            print('Effective sampling frequency: %.1f Hz ' % (1./np.mean(np.diff(self.times))))

    def stop(self):
        self.cam.stop()
        self.save_times()

        

if __name__=='__main__':

    import threading
    # def launch_rec(duration):
    #     camera.rec(duration)
    camera = CameraAcquisition()
    camera_thread = threading.Thread(target=camera.rec, args=(3,))
    camera_thread.start()
    time.sleep(1)
    camera.running = False
    # camera.stop()


    
    

