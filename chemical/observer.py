import numpy as np
import pymoskito as pm
from extended_kalman_filter import ExtendedKalmanFilterObserver
from moving_horizon_estimation import MovingHorizonEstimator
from chemical.model import ChemicalObserverModel


class ChemicalEKF(ExtendedKalmanFilterObserver):
    @staticmethod
    def add_public_settings():
        ChemicalEKF.public_settings['zero clipping'] = False

    def __init__(self, settings):
        np.random.seed(0)
        super().__init__(settings, ChemicalObserverModel())
        self.zero_clipping = self._settings['zero clipping']

    def _observe(self, time, system_input, system_output):
        obs_result = super()._observe(time, system_input, system_output)
        if self.zero_clipping:
            self.filter_algorithm.x = np.fmax(self.filter_algorithm.x, np.zeros(2, dtype=float))

        return self.filter_algorithm.x


class ChemicalMHE(MovingHorizonEstimator):
    def __init__(self, settings):
        np.random.seed(0)
        super().__init__(settings, ChemicalObserverModel())


ChemicalEKF.add_public_settings()
pm.register_simulation_module(pm.Observer, ChemicalEKF)
pm.register_simulation_module(pm.Observer, ChemicalMHE)