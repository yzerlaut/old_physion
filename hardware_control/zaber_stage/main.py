from zaber_motion import Library
from zaber_motion.ascii import Connection
from zaber_motion import Units
import time
import RPi.GPIO as GPIO
import os

### SETTINGS ###

# COM port
if os.name == 'nt':
    com_port = 'COM4'
elif os.uname()[4][:3] == 'arm':
    com_port = '/dev/ttyUSB0'
# Offset from the whiskers
offset = 10


def set_start_position():
    print('Using the manual controls, set the stimulator in the whisker field')
    time.sleep(1)
    input("Press Enter when ready...")
    position = axis.get_position(Units.LENGTH_MILLIMETRES)
    position_start = position - offset
    axis.move_absolute(position_start, Units.LENGTH_MILLIMETRES)
    return position, position_start

def start_stop():
    if status == 'Standby':
        print('\n* Experiment Started')
        status = 'Recording'
        time.sleep(2)
    elif status == 'Recording':
        print('\n* Experiment Stopped')
        status = 'Standby'
        axis.home()
        time.sleep(2)

#def rand_move():


def stage_move():
    if status == 'Recording':
        current_position = axis.get_position()
        if current_position == position_start:
            axis.move_absolute(position_whisker, Units.LENGTH_MILLIMETRES)
        elif current_position == position_whisker:
            axis.move_absolute(position_start, Units.LENGTH_MILLIMETRES)
   
def setup_rpi():
    GPIO.setwarnings(False) # Ignore warning for now
    GPIO.setmode(GPIO.BCM) # Use physical pin numbering
    GPIO.setup(11, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    GPIO.setup(12, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    GPIO.add_event_detect(11, GPIO.RISING)
    GPIO.add_event_callback(11, stage_move)
    GPIO.add_event_detect(12, GPIO.RISING)
    GPIO.add_event_callback(12, start_stop)

def startup():
    print('*** Starting whisker stimulation *** \n')
    print('[1] Setting up the controller')
    setup_rpi()
    time.sleep(2)
    print('[2] Setting up x-axis')
    Library.enable_device_db_store()
    connection = Connection.open_serial_port(com_port)
    # Setup and home x-axis
    device_list = connection.detect_devices()
    print("    -Found {} devices".format(len(device_list)))
    device = device_list[0]
    axis = device.get_axis(1)
    print("    -Homing the device")
    time.sleep(1)
    axis.home()
    time.sleep(1)
    print('[3] Setting up treadmill interface')
    # Setup rotary encoder
    time.sleep(2)
    print('[4] Setting up c-axis')
    # Set stepper motor to base position
    time.sleep(2)
    return connection, device, axis

def experiment():
    position_whisker, position_start = set_start_position()
    print('\n*** READY ***')
    print('* Waiting for start signal...')

def exit():
    print('* Closing connection....')
    connection.close()
   
# Run function

connection, device, axis = startup()
status = 'Standby'

while True:
    experiment()

exit()

#################################
#
# while True: # Run forever
#     if GPIO.input(11) == GPIO.HIGH:
#         if status == 'Standy':
#             print('\n* Experiment Started')
#             status = 'Recording'
#             time.sleep(2)
#     elif GPIO.input(11) == GPIO.LOW:
#         if status == 'Recording':
#             print('\n* Experiment Stopped')
#             status = 'Standby'
#             axis.home()
#             time.sleep(2)
#            
# axis.move_absolute(position_whisker, Units.LENGTH_MILLIMETRES)
# time.sleep(2)
# axis.move_absolute(position_start, Units.LENGTH_MILLIMETRES)
# time.sleep(5)
# axis.move_absolute(position_whisker, Units.LENGTH_MILLIMETRES)
# time.sleep(3)
