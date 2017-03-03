from unittest.mock import MagicMock
import math
from magicbot.magic_tunable import setup_tunables
from automations.localisor import Localisor
from ctre import CANTalon
import numpy as np

class StepController(object):
    '''
        Robot test controller
    '''

    def __init__(self, control, on_step):
        '''constructor'''
        self.control = control
        self.step = 0
        self._on_step = on_step

    def __call__(self, tm):
        '''Called when a new robot DS packet is received'''
        self.step += 1
        return self._on_step(tm, self.step)

def test_localisation_predict(control, hal_data):

    l = Localisor()
    l.vision = MagicMock()
    l.range_finder = MagicMock()
    l.chassis = MagicMock()
    l.bno055 = MagicMock()


    l.bno055.getHeading = MagicMock(return_value=0.0)
    l.chassis.get_raw_wheel_distances = MagicMock(return_value=[0.0, 0.0])
    pos_increment_per_sec_si = 1
    pos_increment_per_sec_enc = pos_increment_per_sec_si*l.chassis.counts_per_meter

    l.reset()

    def _on_step(tm, step):
        enc_dist = step*pos_increment_per_sec_si/50.0
        l.chassis.get_raw_wheel_distances = MagicMock(return_value=[enc_dist, enc_dist])
        l.execute()
        print("Step %s, Hist len %s" % (step, len(l.history)))
        if 2 < step <= 50:
            print("x_hat: %s" % (str(l.filter.x_hat)))
            print("chassis_wheel_dist %s" % (str(l.chassis.get_raw_wheel_distances())))
            assert math.isclose(l.filter.x_hat[0][0], pos_increment_per_sec_si * (step)/50, rel_tol = 0.01)
            # l.chassis.get_raw_wheel_distances = MagicMock(return_value=[step*pos_increment_per_sec_enc/50,
            #         -step*pos_increment_per_sec_enc/50])
            print(l.history[-1])
        elif step > 50:
            return False
        return True

    c = StepController(control, _on_step)
    control.run_test(c)
    assert c.step == 51

def test_angle_predict(control, hal_data):

    l = Localisor()
    l.vision = MagicMock()
    l.range_finder = MagicMock()
    l.chassis = MagicMock()
    l.bno055 = MagicMock()

    l.chassis.get_raw_wheel_distances = MagicMock(return_value=[0.0, 0.0])
    pos_increment_per_sec_si = 1
    pos_increment_per_sec_enc = pos_increment_per_sec_si*l.chassis.counts_per_meter

    l.reset()

    l.bno055.getHeading = MagicMock(return_value=math.pi/4)

    l.filter.x_hat = np.array([0, 0, math.pi/4]).reshape(-1, 1)

    def _on_step(tm, step):
        print("x_hat: %s" % (str(l.filter.x_hat)))
        enc_dist = step*pos_increment_per_sec_si/50.0
        l.chassis.get_raw_wheel_distances = MagicMock(return_value=[enc_dist, enc_dist])
        l.execute()
        print("Step %s, Hist len %s" % (step, len(l.history)))
        if 2 < step <= 50:
            print("chassis_wheel_dist %s" % (str(l.chassis.get_raw_wheel_distances())))
            assert math.isclose(l.filter.x_hat[0][0], (1/math.sqrt(2)) * pos_increment_per_sec_si * step/50, rel_tol = 0.01)
            # l.chassis.get_raw_wheel_distances = MagicMock(return_value=[step*pos_increment_per_sec_enc/50,
            #         -step*pos_increment_per_sec_enc/50])
            print(l.history[-1])
        elif step > 50:
            return False
        return True

    c = StepController(control, _on_step)
    control.run_test(c)
    assert c.step == 51
