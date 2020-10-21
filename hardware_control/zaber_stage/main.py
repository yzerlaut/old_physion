import os, time

from zaber_motion import Library
from zaber_motion.ascii import Connection
from zaber_motion import Units


class ZaberStageControl:
    """
    
    """
    def __init__(self,
                 comp_port=None,
                 position_start = 0,
                 offset=10):

        if comp_port is None:
            # COM port
            if os.name == 'nt':
                com_port = 'COM4'
            elif os.uname()[4][:3] == 'arm':
                com_port = '/dev/ttyUSB0'
            else:
                raise Exception('', 'Need to specify port')

        self.offset = offset
        self.position = 0
        self.position_start = 0 - self.offset

        self.status = 'Standby'


        Library.enable_device_db_store()
        self.connection = Connection.open_serial_port(com_port)

        self.init_device()

        
    def init_device(self):
        # Setup and home x-axis
        device_list = connection.detect_devices()
        print("    -Found {} devices".format(len(device_list)))
        self.device = device_list[0]
        self.axis = device.get_axis(1)
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
        

    def set_start_position(self):
        self.position = axis.get_position(Units.LENGTH_MILLIMETRES)
        self.position_start = self.position - self.offset
        axis.move_absolute(self.position_start, Units.LENGTH_MILLIMETRES)


    def start_stop(self):
        if self.status == 'Standby':
            print('\n* Experiment Started')
            self.status = 'Recording'
            time.sleep(2)
        elif status == 'Recording':
            print('\n* Experiment Stopped')
            self.status = 'Standby'
            axis.home()
            time.sleep(2)

    def stage_move(self):
        if self.status == 'Recording':
            current_position = axis.get_position()
            if current_position == position_start:
                axis.move_absolute(position_whisker, Units.LENGTH_MILLIMETRES)
            elif current_position == position_whisker:
                axis.move_absolute(position_start, Units.LENGTH_MILLIMETRES)
        else:
            print('bla')

            
   
def startup():
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


if __name__=='__main__':

    control = ZaberStageControl()
    

    print('Using the manual controls, set the stimulator in the whisker field')
    time.sleep(1)
    input("Press Enter when ready...")
    control.set_start_position()


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
