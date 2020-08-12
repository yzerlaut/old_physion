"""
The camera needs to be configured in the SpinView software
"""
import simple_pyspin, time, os
import numpy as np
from pathlib import Path

class stop_func: # dummy version of the multiprocessing.Event class
    def __init__(self):
        self.stop = False
    def set(self):
        self.stop = True
    def is_set(self):
        return self.stop
    
class CameraAcquisition:

    def __init__(self,
                 folder='./',
                 stop_flag=stop_func,
                 frame_rate=20):
        
        self.times, self.frame_index = [], 0
        self.folder = folder
        self.imgs_folder = os.path.join(self.folder, 'camera-imgs')
        Path(self.imgs_folder).mkdir(parents=True, exist_ok=True)
        self.init_camera(frame_rate=frame_rate)
        self.batch_index = 0
        self.stop_flag=stop_flag

    def init_camera(self,
                    frame_rate=20):
        
        self.cam = simple_pyspin.Camera()
        self.cam.init()
        # Set the area of interest (AOI) to the middle half
        self.cam.Width = self.cam.SensorWidth // 2
        self.cam.Height = self.cam.SensorHeight // 2
        self.cam.OffsetX = self.cam.SensorWidth // 4
        self.cam.OffsetY = self.cam.SensorHeight // 4

        # # If this is a color camera, get the image in RGB format.
        # if 'Bayer' in self.cam.PixelFormat:
        #     self.cam.PixelFormat = "RGB8"

        # To change the frame rate, we need to enable manual control
        self.cam.AcquisitionFrameRateAuto = 'Off'
        # self.cam.AcquisitionFrameRateEnabled = True # seemingly not available here
        self.cam.AcquisitionFrameRate = frame_rate

        # To control the exposure settings, we need to turn off auto
        self.cam.GainAuto = 'Off'
        # Set the gain to 20 dB or the maximum of the camera.
        gain = min(20, self.cam.get_info('Gain')['max'])
        print("Setting gain to %.1f dB" % gain)
        self.cam.Gain = gain
        self.cam.ExposureAuto = 'Off'
        self.cam.ExposureTime =0.2*1e6/frame_rate # microseconds, 20% of interframe interval

        
    def rec(self, duration, stop_flag, t0=None):
        if t0 is None:
            self.t0 = time.time()
        else:
            self.t0 = t0
        self.cam.start()
        self.t = time.time()-self.t0
        while (not stop_flag.is_set()) and (self.t<duration):
            self.frame_index +=1
            np.save(os.path.join(self.imgs_folder, '%i.npy' % self.frame_index), np.array(self.cam.get_array()))
            self.t=time.time()-self.t0
            self.times.append(self.t)
            
        if self.t>=duration:
            print('camera acquisition finished !')
            self.stop()
        elif stop_flag.is_set():
            print('camera acquisition stopped !   (at t=%.2fs)' % self.t)
            self.stop()
            
    def save_times(self, verbose=True):
        print('Camera data saved as: ', os.path.join(self.folder, 'camera-times.npy'))
        np.save(os.path.join(self.folder, 'camera-times.npy'), np.array(self.times))
        if verbose:
            print('Effective sampling frequency: %.1f Hz ' % (1./np.mean(np.diff(self.times))))

    def stop(self):
        self.cam.stop()
        self.save_times()

        
def camera_init_and_rec(duration, stop_flag):
    camera = CameraAcquisition()
    camera.rec(duration, stop_flag)
    
if __name__=='__main__':

    T = 5 # seconds

    import multiprocessing
    def launch_rec(duration):
        camera.rec(duration)
        
    # camera = CameraAcquisition()
    # stop = stop_func()
    # camera.rec(T, stop)
    # stop.set()
    
    stop_event = multiprocessing.Event()
    camera_process = multiprocessing.Process(target=camera_init_and_rec, args=(T,stop_event))
    camera_process.start()
    print(stop_event.is_set())
    time.sleep(T/2)
    stop_event.set()
    print(stop_event.is_set())
    # camera.stop()


    
    

