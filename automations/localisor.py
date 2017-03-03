import numpy as np
import math
import time

from collections import deque

import magicbot

from components.vision import Vision
from components.range_finder import RangeFinder
from components.chassis import Chassis
from components.bno055 import BNO055
from utilities.kalman import Kalman

class Localisor:

    absolute_positioning_variance_x = 0.01
    absolute_positioning_variance_y = 0.01

    heading_variance_per_timestep = (
        0.03*50/135)

    odometry_variance_per_timestep = 0.5e-6

    initial_pos_variance = 0.01

    hist_len = 50

    center_peg_x = 2.93
    side_peg_x = 2.54 + 0.75
    side_peg_y = 1.75 - 1.3

    vision = Vision
    range_finder = RangeFinder
    chassis = Chassis
    bno055 = BNO055

    def __init__(self):
        pass

    def setup(self):
        self.reset()

    def reset(self):
        # initial state
        x_hat = np.array([0, 0, self.bno055.getHeading()]).reshape(3, -1)

        P = np.zeros(shape=(3, 3))
        P[0][0] = Localisor.initial_pos_variance
        P[0][1] = Localisor.initial_pos_variance
        Q = np.zeros(shape=(3, 3))
        Q[0][0] = Localisor.odometry_variance_per_timestep
        Q[1][1] = Localisor.odometry_variance_per_timestep
        # the predict state for odometry works off heading, so convert
        # odditional variance to radians and bultiply by 2 for each wheel
        Q[2][2] = (2*Localisor.odometry_variance_per_timestep)/Chassis.wheelbase_width
        self.history = deque(
                iterable=
                [np.array([self.chassis.get_raw_wheel_distances()[0],
                    self.chassis.get_raw_wheel_distances()[1],
                    self.bno055.getHeading(),
                    self.range_finder.getDistance()]).reshape(-1, 1)],
                maxlen=self.hist_len)
        self.last_vision_time = self.vision.time
        R = np.zeros(shape=(3, 3))
        R[0][0] = Localisor.absolute_positioning_variance_x
        R[1][1] = Localisor.absolute_positioning_variance_y
        R[2][2] = Localisor.heading_variance_per_timestep
        self.filter = Kalman(x_hat, P, Q, R)

    def predict(self, history):
        F = np.identity(3)
        B = np.identity(3)
        # difference in the whees position form last timestep to this one
        deltas = history[-1]-history[-2]
        # print("Deltas %s" % (str(deltas)))

        # difference in heading from last timestep to this one as measured by the odometry
        delta_theta = (-deltas[0]+deltas[1])/Chassis.wheelbase_width

        # current heading - best estimate of heading according to kalman filter
        # for last timestep + heading change
        current_heading = self.filter.x_hat[2][0] + delta_theta

        # arclength of the path of the robot from last timestep to this one
        center_movement = (deltas[0]+deltas[1])/2
        # print(center_movement)

        # average heading of the robot over the path since the last timestep
        average_heading = current_heading-(delta_theta/2)

        delta_x = center_movement*math.cos(current_heading)
        delta_y = center_movement*math.sin(current_heading)
        # print("d_x %s, d_y %s" % (delta_x, delta_y))

        u = np.array([
            delta_x, delta_y, delta_theta]).reshape(-1, 1)
        self.filter.predict(F, u, B)

    def gyro_update(self, history):

        z = np.zeros(shape=(3, 1))
        z[2] = [history[-1][2]]
        # print(z)
        H = np.zeros(shape=(3, 3))
        H[2][2] = 1
        self.filter.update(z, H)

    def absolute_update(self, history):
        measured_heading = history[-1][2]
        measured_range = history[-1][3]
        vision_angle = self.vision.derive_vision_angle()

        # peg x and y
        p_x = 0
        p_y = 0

        # tower side angle
        theta_airship = 0

        if -math.pi/3 < self.filter.measured_heading < math.pi/3:
            # calculate offsets based off of center tower
            p_x = Localisor.center_peg_x
            theta_airship = 0
        elif self.filter.measured_heading > math.pi/3:
            # calculate offsets based off of right tower
            p_x = Localisor.side_peg_x
            p_y = -Localisor.side_peg_y
            theta_airship = math.pi/3
        else:
            # calculate offsets based off of left tower
            p_x = Localisor.side_peg_x
            p_y = -Localisor.side_peg_y
            theta_airship = -math.pi/3

        # the angle made between the line perpendicular to the airship and
        # the angle towards the peg
        theta_peg = vision_angle - (theta_airship-measured_heading)

        # straight line range from the robot to the nearest point on the airship side
        range_airship = measured_range*math.sin((math.pi/2)-(theta_airship-measured_heading))

        # straight line distance to the base of the peg
        range_peg = range_airship/math.sin((math.pi/2)-theta_peg)

        theta_peg_x_axis = math.pi - measured_heading - vision_angle

        theta_peg_y_axis = math.pi/2 - theta_peg_x_axis

        delta_x = range_peg/math.sin(theta_peg_y_axis)
        delta_y = range_peg/math.sin(theta_peg_x_axis)

        # field centered x and y
        field_x = delta_x + p_x
        field_y = delta_y + p_y

        z = np.array([field_x, field_y, measured_heading]).reshape(-1, 1)
        H = np.identity(3)
        self.filter.update(z, H)

    def get_x(self):
        return self.filter.x_hat[0][0]

    def get_y(self):
        return self.filter.x_hat[1][0]

    def get_theta(self):
        return self.filter.x_hat[2][0]

    def execute(self):
        self.history.append(
                np.array([self.chassis.get_raw_wheel_distances()[0],
                    self.chassis.get_raw_wheel_distances()[1],
                    self.bno055.getHeading(),
                    self.range_finder.getDistance()]).reshape(-1, 1))
        if len(self.history) < 2:
            return
        self.predict(self.history)
        self.gyro_update(self.history)
        if False:#self.last_vision_time != self.vision.time and self.vision.num_targets > 1:
            timesteps_since_vision = int((time.monotonic() - self.vision.time())*1/50)
            self.filter.roll_back(timesteps_since_vision)
            self.absolute_update(self, history[:-timesteps_since_vision])
            # perform update, then re predict our position forward
            for i in range(timesteps_since_vision-1, -1, -1):
                self.predict(self.history[:i])
                self.gyro_update(self.history[:i])
            self.last_vision_time = self.vision.time
