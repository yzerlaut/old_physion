import sys

import RPi.GPIO as GPIO

sys.path.append(str(Path(__file__).resolve().parents[1]))
from zaber_stage import ZaberStageControl


class StageController:

    def __init__(self,
                 stage_controller_args={}):

        stage_controller = ZaberStageControl(**stage_controller_args)

        GPIO.setwarnings(False) # Ignore warning for now
        GPIO.setmode(GPIO.BCM) # Use physical pin numbering
        GPIO.setup(11, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        GPIO.setup(12, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        GPIO.add_event_detect(11, GPIO.RISING)
        GPIO.add_event_callback(11, stage_controller.stage_move)
        GPIO.add_event_detect(12, GPIO.RISING)
        GPIO.add_event_callback(12, stage_controller.start_stop)


if __name__=='__main__':

    print('*** Starting whisker stimulation *** \n')
    print('[1] Setting up the controller')

    time.sleep(2)
    
    pi = StageController()
