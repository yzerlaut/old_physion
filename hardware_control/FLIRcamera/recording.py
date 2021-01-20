"""

"""
import simple_pyspin, time, sys, os, datetime
from skimage.io import imsave
import numpy as np
import pynwb
from hdmf.data_utils import DataChunkIterator

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from assembling.saving import last_datafolder_in_dayfolder, day_folder

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class stop_func: # dummy version of the multiprocessing.Event class
    def __init__(self):
        self.stop = False
    def set(self):
        self.stop = True
    def is_set(self):
        return self.stop
    
class CameraAcquisition:

    def __init__(self, stop_flag,
                 settings={'frame_rate':20.}):

        try:
            self.stop = stop_flag
            print(self.stop.is_set())
            self.cam = simple_pyspin.Camera()
            self.cam.init()
            self.init_camera_settings(settings)
            self.cam.start()
            self.img = self.cam.get_array()
            self.success_init = True
        except BaseException as be:
            print(be)
            print('\n /!\ the camera could not be initialized /!\  ')
            self.success_init = False

        self.times = []
        
    def init_camera_settings(self, settings):
        
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


    def frame_generator(self, max_frame=100):
        """
        ...
        """
        # while (not stop_flag.is_set()):
        i=0
        self.times = []
        while self.stop.is_set() and i<max_frame:
            i+=1
            self.times.append(time.time())
            yield self.cam.get_array()
            if i>10:
                self.stop.clear()
        return        
        
        
    def rec(self, filename):

        self.nwbfile = pynwb.NWBFile(identifier=filename,
                                     session_description='FaceCamera Acquisition',
                                     session_start_time=datetime.datetime.now(),
                                     source_script=str(Path(__file__).resolve()),
                                     source_script_file_name=str(Path(__file__).resolve()),
                                     file_create_date=datetime.datetime.today())

        print(self.stop.is_set())
        data = DataChunkIterator(data=self.frame_generator(),
                                 maxshape=(None,self.img.shape[0], self.img.shape[1]),
                                 dtype=np.dtype(np.uint8))

        images = pynwb.image.ImageSeries(name='FaceCamera',
                                         data=data,
                                         unit='NA',
                                         starting_time=time.time(),
                                         rate=self.cam.AcquisitionFrameRate)
        
        self.nwbfile.add_acquisition(images)
        print(self.times)
        io = pynwb.NWBHDF5IO(filename, 'w')
        io.write(self.nwbfile)
        io.close()

        # stopping the camera
        self.cam.stop()
        
        # adding times
        io = pynwb.NWBHDF5IO(filename, 'r+')
        self.nwbfile = io.read()
        ts = pynwb.TimeSeries(name='frame-timestamps',
                              data=np.array(self.times),
                              unit='second',
                              rate=self.cam.AcquisitionFrameRate)
        self.nwbfile.add_acquisition(ts)
        print(np.unique(self.nwbfile.acquisition['FaceCamera'].data))
        # write the modified NWB file
        io.write(self.nwbfile)
        io.close()

        
def launch_FaceCamera(filename, stop_flag,
                      settings={'frame_rate':20.}):
    camera = CameraAcquisition(stop_flag, settings=settings)
    if camera is not None:
        camera.stop.set()
        print(camera.stop.is_set())
        camera.rec(filename)
    else:
        print(' /!\ The camera process could NOT be launched /!\ ')

if __name__=='__main__':

    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    Trun = 2 # seconds

    import multiprocessing
        
    fn = os.path.join(os.path.expanduser('~'), 'DATA', 'frames.nwb')
    stop = multiprocessing.Event()

    settings = {'frame_rate':20.}
    camera_process = multiprocessing.Process(target=launch_FaceCamera,
                                             args=(fn, stop, settings))
    camera_process.start()
    time.sleep(Trun)
    stop.clear()
