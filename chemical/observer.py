import numpy as np
import pymoskito as pm
from extended_kalman_filter import ExtendedKalmanFilterObserver
from moving_horizon_estimation import MovingHorizonEstimator
from chemical.model import ChemicalObserverModel


class ChemicalEKF(ExtendedKalmanFilterObserver):
    def __init__(self, settings):
        np.random.seed(0)
        super().__init__(settings, ChemicalObserverModel())


class ChemicalMHE(MovingHorizonEstimator):
    def __init__(self, settings):
        np.random.seed(0)
        super().__init__(settings, ChemicalObserverModel())


pm.register_simulation_module(pm.Observer, ChemicalEKF)
pm.register_simulation_module(pm.Observer, ChemicalMHE)