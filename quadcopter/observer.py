import pymoskito as pm
import numpy as np
import math
from scipy.linalg import norm

from extended_kalman_filter import ExtendedKalmanFilterObserver
from moving_horizon_estimation import MovingHorizonEstimator, Array
from quadcopter.model import QuadcopterObserverModel
from quadcopter.util import *


class QuadcopterEKF(ExtendedKalmanFilterObserver):
    @staticmethod
    def add_public_settings():
        QuadcopterEKF.public_settings['trajectory'] = 'sim_1_stationary'
        QuadcopterEKF.public_settings['q norm correction'] = True

    def __init__(self, settings):
        super().__init__(settings, QuadcopterObserverModel())

        self.trajectory = self._settings['trajectory']
        self.norm_correction = self._settings['q norm correction']
        self.ts, self.meas_accelerometer, self.meas_gyro, self.ref_ts, self.ref_pos, self.ref_angles = get_trajectory(self.trajectory)

    def _observe(self, t, system_input, system_output):
        imu_index = time_to_index(t, self.ts)
        ref_index = time_to_index(t, self.ref_ts)
        ref_pos = self.ref_pos[ref_index]
        ref_orientation = self.ref_angles[ref_index]
        meas_gyro = self.meas_gyro[imu_index]
        meas_accelerometer = self.meas_accelerometer[imu_index]

        system_measurement = np.concatenate((meas_accelerometer, np.array([ref_orientation[2]])))

        observer_out = super()._observe(t, system_input=meas_gyro, system_output=system_measurement)
        obs_q = observer_out[:4]

        # Normalize quaterion
        if self.norm_correction:
            self.filter_algorithm.x[:4] = obs_q / norm(obs_q)
            obs_q = self.filter_algorithm.x[:4]

        obs_orientation = q_to_euler(obs_q)
        obs_err = euler_difference(ref_orientation, obs_orientation)

        output = np.concatenate((ref_pos, rad_to_deg(ref_orientation), rad_to_deg(obs_orientation), rad_to_deg(obs_err), observer_out[4:7], meas_gyro, meas_accelerometer, [norm(obs_q)]))

        return output


class QuadcopterMHE(MovingHorizonEstimator):
    @staticmethod
    def add_public_settings():
        QuadcopterMHE.public_settings['trajectory'] = 'sim_1_stationary'

    def __init__(self, settings):
        super().__init__(settings, QuadcopterObserverModel())

        self.trajectory = self._settings['trajectory']
        self.ts, self.meas_accelerometer, self.meas_gyro, self.ref_ts, self.ref_pos, self.ref_angles = get_trajectory(self.trajectory)

    def _observe(self, t, system_input: Array, system_output: Array) -> Array:
        imu_index = time_to_index(t, self.ts)
        ref_index = time_to_index(t, self.ref_ts)
        ref_pos = self.ref_pos[ref_index]
        ref_orientation = self.ref_angles[ref_index]
        meas_gyro = self.meas_gyro[imu_index]
        meas_accelerometer = self.meas_accelerometer[imu_index]

        system_measurement = np.concatenate((meas_accelerometer, np.array([ref_orientation[2]])))

        observer_out = super()._observe(t, system_input=meas_gyro, system_output=system_measurement)
        obs_q = observer_out[:4]

        obs_orientation = q_to_euler(obs_q)
        obs_err = euler_difference(ref_orientation, obs_orientation)

        output = np.concatenate((ref_pos, rad_to_deg(ref_orientation), rad_to_deg(obs_orientation), rad_to_deg(obs_err), observer_out[4:7], meas_gyro, meas_accelerometer, [norm(obs_q)]))
        return output


QuadcopterEKF.add_public_settings()
pm.register_simulation_module(pm.Observer, QuadcopterEKF)
QuadcopterMHE.add_public_settings()
pm.register_simulation_module(pm.Observer, QuadcopterMHE)