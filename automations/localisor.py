import numpy as np
import math
import time

from collections import deque
from itertools import islice

import magicbot

from components.vision import Vision
from components.range_finder import RangeFinder
from components.chassis import Chassis
from components.bno055 import BNO055
from utilities.kalman import Kalman

class Localisor:

    # absolute_positioning_variance_x = 0.01
    # absolute_positioning_variance_y = 0.01
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

    center_to_front_bumper = 0.49
    sensors_to_front_bumper = 0.36

    sensor_center_offset = center_to_front_bumper - sensors_to_front_bumper

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

        # difference in heading from last timestep to this one as measured by the odometry
        delta_theta = (-deltas[0]+deltas[1])/Chassis.wheelbase_width

        # current heading - best estimate of heading according to kalman filter
        # for last timestep + heading change
        current_heading = self.filter.x_hat[2][0] + delta_theta

        # arclength of the path of the robot from last timestep to this one
        center_movement = (deltas[0]+deltas[1])/2

        # average heading of the robot over the path since the last timestep
        average_heading = current_heading-(delta_theta/2)

        delta_x = center_movement*math.cos(current_heading)
        delta_y = center_movement*math.sin(current_heading)

        u = np.array([
            delta_x, delta_y, delta_theta]).reshape(-1, 1)
        self.filter.predict(F, u, B)

    def gyro_update(self, history):

        z = np.zeros(shape=(3, 1))
        z[2] = [history[-1][2]]
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

        if -math.pi/3 < measured_heading < math.pi/3:
            # calculate offsets based off of center tower
            p_x = Localisor.center_peg_x
            theta_airship = 0
        elif measured_heading > math.pi/3:
            # calculate offsets based off of right tower
            p_x = Localisor.side_peg_x
            p_y = -Localisor.side_peg_y
            theta_airship = math.pi/3
        else:
            # calculate offsets based off of left tower
            p_x = Localisor.side_peg_x
            p_y = -Localisor.side_peg_y
            theta_airship = -math.pi/3

        # various angles that crop up from the maths
        theta_peg_from_y_axis = math.pi/2 - measured_heading - vision_angle
        theta_peg_from_x_axis = math.pi/2 - theta_peg_from_y_axis
        theta_alpha = math.pi - theta_peg_from_x_axis
        delta_d = measured_range * math.sin(math.pi - theta_alpha - vision_angle) / math.sin(theta_alpha)

        # calculate the x and y components of the sensor offset for our current heading
        sensor_x_offset = self.sensor_center_offset*math.sin(math.pi/2 - measured_heading)
        sensor_y_offset = self.sensor_center_offset*math.sin(measured_heading)

        delta_x = -(delta_d * math.sin(theta_peg_from_y_axis)) - sensor_x_offset
        delta_y = -(delta_d * math.sin(theta_peg_from_x_axis)) - sensor_y_offset

        # field based x and y
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
        if self.last_vision_time != self.vision.time and self.vision.num_targets > 1:
            timesteps_since_vision = int((time.monotonic() - self.vision.time)*1/50)
            if timesteps_since_vision > 0:
                self.filter.roll_back(timesteps_since_vision)
                self.absolute_update(deque(islice(self.history, 0, -timesteps_since_vision)))
            else:
                self.absolute_update(self.history)
            # perform update, then re predict our position forward
            if timesteps_since_vision > 0:
                for i in range(timesteps_since_vision-1, -1, -1):
                    self.predict(deque(islice(self.history, 0, i)))
                    self.gyro_update(deque(islice(self.history, 0, i)))
                self.last_vision_time = self.vision.time
