import pymoskito as pm
import numpy as np
import math
from scipy.linalg import norm

from extended_kalman_filter import ExtendedKalmanFilterObserver
from quadcopter.model import QuadcopterObserverModel
from quadcopter.util import *


class QuadcopterEKF(ExtendedKalmanFilterObserver):
    @staticmethod
    def add_public_settings():
        QuadcopterEKF.public_settings['trajectory'] = 'sim_1_stationary'
        QuadcopterEKF.public_settings['q norm correction'] = True

    def __init__(self, settings):
        np.random.seed(10)

        super().__init__(settings, QuadcopterObserverModel())

        # self.gyro_noise = self._settings['gyro noise']
        # self.accelerometer_noise = self._settings['accelerometer noise']
        self.trajectory = self._settings['trajectory']
        self.norm_correction = self._settings['q norm correction']
        self.ts, self.ref_pos, self.ref_angles, self.meas_gyro, self.meas_accelerometer = get_trajectory(self.trajectory)

    def _observe(self, t, system_input, system_output):
        pi = np.pi
        index = time_to_index(t, self.ts)
        ref_pos = self.ref_pos[index]
        ref_orientation = self.ref_angles[index]
        meas_gyro = self.meas_gyro[index]
        meas_accelerometer = self.meas_accelerometer[index]

        system_measurement = np.concatenate((meas_accelerometer, np.array([ref_orientation[2]])))

        observer_out = super()._observe(t, system_input=meas_gyro, system_output=system_measurement)
        obs_q = observer_out[:4]

        # Normalize quaterion
        if self.norm_correction:
            self.filter_algorithm.x = obs_q / norm(obs_q)
            obs_q = self.filter_algorithm.x

        obs_orientation = q_to_euler(obs_q)
        obs_err = euler_difference(ref_orientation, obs_orientation)

        output = np.concatenate((ref_orientation, obs_orientation, obs_err, meas_gyro, meas_accelerometer, [norm(obs_q)]))
        return output


QuadcopterEKF.add_public_settings()
pm.register_simulation_module(pm.Observer, QuadcopterEKF)
