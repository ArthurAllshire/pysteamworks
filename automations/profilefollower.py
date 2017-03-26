from components.chassis import Chassis
from components.bno055 import BNO055
from networktables import NetworkTable

import math

class ProfileFollower:

    # linear motion feedforward/back gains
    #
    # # position P controller
    # kP = 6
    # # velocity and acceleration feedforward
    # kV = 1
    # kI = 0.3
    # kD = 1
    # position P controller
    kP = 5
    # velocity and acceleration feedforward
    kV = 1
    kI = 0.1
    kD = 0.7
    kA = 0.0

    # heading motion feedforward/back gains
    # heading feedback
    kPh = 4
    # angular velocity feedforward
    kVh = 1
    # kIh = 0.5
    # kDh = 40
    kIh = 0.1
    kDh = 20
    # # heading motion feedforward/back gains
    # # heading feedback
    # kPh = 4.5
    # # angular velocity feedforward
    # kVh = 1.2

    chassis = Chassis

    bno055 = BNO055
    sd = NetworkTable

    def __init__(self):
        # queue of segments to be exectued
        # in the form [linear, angular]
        self.queue = [[],[]]
        self.executing = False

    def modify_queue(self, heading, linear=None, overwrite=False):
        """Modify the motion profiling queue, or overwrite it.
        """
        if type(heading) == int or type(heading) == float:
            heading = [(heading, 0.0, 0.0)] * len(linear)
        if not linear:
           linear = [[0.0, 0.0, 0.0]] * len(heading)
        if overwrite:
            self.queue = [linear, heading]
        else:
            self.queue = [self.queue[0]+linear, self.queue[1]+heading]

    def execute_queue(self):
        """Start executing the queue that is in the motion profile buffer.
        Also resets encoder positions in the chassis, and disables driver
        input to the chassis in teleop.
        """
        # ensure that there is a queue to execute from
        if len(self.queue[0]):
            self.executing = True
            self.chassis.input_enabled = False
            self.chassis.set_enc_pos()
            self.position_error_i = 0
            self.heading_error_i = 0
            self.last_position_error = 0
            self.last_heading_error = 0

    def stop(self):
        """Stop executing the motion profile.  Also clears the MP queue, and
        re-enables chassis driver input.
        """
        self.executing = False
        self.queue = [[], []]
        self.chassis.input_enabled = True
        self.heading_error_i = 0
        self.position_error_i = 0

    def execute(self):
        if self.executing:
            # get the next linear and angular segments from the front of the
            # queue
            linear_seg = self.queue[0].pop(0)
            heading_seg = self.queue[1].pop(0)

            [left_pos, right_pos] = self.chassis.get_wheel_distances()

            pos = (left_pos + right_pos) / 2 # average the two wheel distances
            pos_error = linear_seg[0] - pos

            self.d_pos_error = (pos_error - self.last_position_error)
            self.position_error_i += pos_error

            linear_output = (self.kP * pos_error + self.kV * linear_seg[1]
                + self.kA * linear_seg[2] + self.kI*self.position_error_i + self.kD*self.d_pos_error)

            self.sd.putNumber("distance_error_mp", pos_error)
            self.sd.putNumber("linear_output_mp", linear_output)

            # generate the linear output to the chassis (m/s)
            heading = self.bno055.getRawHeading() - self.bno055.offset
            heading_error = heading_seg[0] - heading
            # wrap heading error, stops jumping by tau from the gyro
            heading_error = math.atan2(math.sin(heading_error),
                    math.cos(heading_error))

            self.heading_error_i += heading_error
            d_heading_error = (heading_error - self.last_heading_error)

            # generate the rotational output to the chassis
            heading_output = (
                self.kPh * heading_error + self.kVh * heading_seg[1] + self.heading_error_i*self.kIh + d_heading_error*self.kDh)

            self.last_heading_error = heading_error
            self.last_position_error = pos_error

            self.sd.putNumber("heading_error_mp", heading_error)

            if self.queue[0]:
                self.chassis.set_velocity(linear_output, heading_output)
            else:
                self.chassis.set_velocity(linear_seg[1], heading_seg[1])
        if not self.queue[0]:
            self.executing = False
