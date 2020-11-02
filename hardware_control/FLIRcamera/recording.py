"""

"""
import simple_pyspin, time, sys, os
from skimage.io import imsave
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from assembling.saving import last_datafolder_in_dayfolder, day_folder

desktop_png = os.path.join(os.path.expanduser("~/Desktop"), 'FaceCamera.png')

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
                 root_folder=None,
                 imgs_folder=None,
                 settings={'frame_rate':20.}):
        
        self.times, self.frame_index = [], 0
        if root_folder is not None:
            self.root_folder = root_folder
            self.folder = last_datafolder_in_dayfolder(day_folder(self.root_folder))
        else:
            self.root_folder = './'
        if imgs_folder is None:
            self.imgs_folder = os.path.join(self.folder, 'FaceCamera-imgs')
            Path(self.imgs_folder).mkdir(parents=True, exist_ok=True)
        else:
            self.imgs_folder = imgs_folder
        self.init_camera(settings)
        self.batch_index = 0
        self.running=False
        
    def init_camera(self, settings):
        
        self.cam = simple_pyspin.Camera()
        self.cam.init()

        ###
        ## -- SETTINGS through the FlyCap software, easier....
        
        # # Set the area of interest (AOI) to the middle half
        # self.cam.Width = self.cam.SensorWidth // 2
        # self.cam.Height = self.cam.SensorHeight // 2
        # self.cam.OffsetX = self.cam.SensorWidth // 4
        # self.cam.OffsetY = self.cam.SensorHeight // 4

        # # To change the frame rate, we need to enable manual control
        self.cam.AcquisitionFrameRateAuto = 'Off'
        # # self.cam.AcquisitionFrameRateEnabled = True # seemingly not available here
        self.cam.AcquisitionFrameRate = settings['frame_rate']

        # # To control the exposure settings, we need to turn off auto
        # self.cam.GainAuto = 'Off'
        # # Set the gain to 20 dB or the maximum of the camera.
        # max_gain = self.cam.get_info('Gain')['max']
        # if (settings['gain']==0) or (settings['gain']>max_gain):
        #     self.cam.Gain = max_gain
        #     print("Setting FaceCamera gain to %.1f dB" % max_gain)
        # else:
        #     self.cam.Gain = settings['gain']

        # self.cam.ExposureAuto = 'Off'
        # self.cam.ExposureTime =settings['exposure_time'] # microseconds, ~20% of interframe interval


    def save_sample_on_desktop(self):
        ### SAVING A SAMPLE ON THE DESKTOP
        print('saving a sample image as:', desktop_png)
        imsave(desktop_png, np.array(self.cam.get_array()))

    def rec(self, duration, stop_flag, camready_flag, t0=None):
        if t0 is None:
            t0=time.time()
        self.running=True
        self.frame_index = 0 # need to reset the index here
        self.cam.start()
        self.t = time.time()
        camready_flag.set()
        
        while (not stop_flag.is_set()) and ((self.t-t0)<duration):
            self.frame_index +=1
            np.save(os.path.join(self.imgs_folder, '%i.npy' % self.frame_index), np.array(self.cam.get_array()))
            self.t=time.time()
            self.times.append(self.t)
            
        if (self.t-t0)>=duration:
            print('camera acquisition finished !')
            self.stop()
        elif stop_flag.is_set():
            print('camera acquisition stopped !   (at t=%.2fs)' % self.t)
            self.stop()


    def reinit_rec(self):
        self.running = True
        self.folder = last_datafolder_in_dayfolder(day_folder(self.root_folder))
        self.imgs_folder = os.path.join(self.folder, 'FaceCamera-imgs')
        Path(self.imgs_folder).mkdir(parents=True, exist_ok=True)
        self.times = []

        
    def rec_and_check(self, run_flag, quit_flag):
        
        self.cam.start()
        self.t = time.time()

        self.save_sample_on_desktop()
        
        while not quit_flag.is_set():
            
            image = self.cam.get_array()
            
            if not self.running and run_flag.is_set() : # not running and need to start  !
                self.reinit_rec()
                self.save_sample_on_desktop()
            elif self.running and not run_flag.is_set(): # running and we need to stop
                self.running=False
                self.save_times()
                self.save_sample_on_desktop()

            # after the update
            if self.running:
                np.save(os.path.join(self.imgs_folder, '%i.npy' % self.frame_index), image)
                self.frame_index +=1
                self.t=time.time()
                self.times.append(self.t)

        self.save_sample_on_desktop()
        # self.save_times()

        
    def save_times(self, verbose=True):
        print('[ok] Camera data saved as: ', os.path.join(self.folder, 'FaceCamera-times.npy'))
        np.save(os.path.join(self.folder, 'FaceCamera-times.npy'), np.array(self.times))
        if verbose:
            print('FaceCamera -- effective sampling frequency: %.1f Hz ' % (1./np.mean(np.diff(self.times))))

    def stop(self):
        self.running=False
        self.cam.stop()
        self.save_times()

        
def camera_init_and_rec(duration, stop_flag, camready_flag, settings={'frame_rate':20.}):
    camera = CameraAcquisition(folder=folder, settings=settings)
    camera.rec(duration, stop_flag, camready_flag)

def launch_FaceCamera(run_flag, quit_flag, root_folder, settings={'frame_rate':20.}):
    camera = CameraAcquisition(root_folder=root_folder, settings=settings)
    camera.rec_and_check(run_flag, quit_flag)
    
if __name__=='__main__':

    T = 2 # seconds

    import multiprocessing
        
    # camera = CameraAcquisition()
    # stop = stop_func()
    # camera.rec(T, stop)
    # stop.set()
    
    # stop_event = multiprocessing.Event()
    # camera_process = multiprocessing.Process(target=camera_init_and_rec, args=(T,stop_event, './'))
    # camera_process.start()

    run = multiprocessing.Event()
    quit_event = multiprocessing.Event()
    camera_process = multiprocessing.Process(target=launch_FaceCamera, args=(run, quit_event, './'))
    run.clear()
    camera_process.start()
    time.sleep(3)
    run.set()
    time.sleep(10)
    run.clear()
    time.sleep(3)
    quit_event.set()
    
    # print(stop_event.is_set())
    # time.sleep(T/2)
    # stop_event.set()
    # print(stop_event.is_set())
    # time.sleep(T/4)
    # stop_event.clear()
    # print(stop_event.is_set())
    # time.sleep(T/4)
    # stop_event.set()
    # print(stop_event.is_set())
    # camera.stop()
    times = np.load('FaceCamera-times.npy')
    print('max blank time of FaceCamera: %.0f ms' % (1e3*np.max(np.diff(times))))
    import matplotlib.pylab as plt
    plt.plot(times, 0*times, '|')
    plt.show()

    
    

