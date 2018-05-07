from collections import OrderedDict
from typing import *
import numpy as np

import pymoskito as pm

Array = np.ndarray


class ObserverModel:
    """
    Base class for system models to be used in observers like the EKF and MHE.
    """

    def __init__(self, state_dim: int, input_dim: int, output_dim: int,
                 f_jacobian: Callable[[Array, Array, float], Array], h_jacobian: Callable[[Array], Array],
                 state_ineq_constraints: List[Callable[[Array], Union[Array, float]]] = None,
                 state_eq_constraints: List[Callable[[Array], Union[Array, float]]] = None):
        self.state_dim = state_dim
        """Size of the state vector x"""
        self.input_dim = input_dim
        """Size of the input vector u"""
        self.output_dim = output_dim
        """Size of the output vector y"""

        self.f_jacobian = f_jacobian
        """Callable (array of states, array of inputs, step width) that returns the discrete state function jacobian with respect to x"""
        self.h_jacobian = h_jacobian
        """Callable (array of states) that returns the output function jacobian with respect to x"""

        self.state_ineq_constraints: List[Callable[[Array], float]] = []
        """List of callables to evaluate inequality constraints (constraint satisfied when > 0)"""

        if state_ineq_constraints is not None:
            self.state_ineq_constraints = state_ineq_constraints

        self.state_eq_constraints: List[Callable[[Array], float]] = []
        """List of callables to evaluate equality constraints (constraint satisfied when = 0)"""

        if state_eq_constraints is not None:
            self.state_eq_constraints = state_eq_constraints

    def discrete_state_func(self, x: Array, u: Array, w: Array, h: float) -> Array:
        """
        State transition function f(x, u, w) of discrete time model
        :param x: State vector at current time
        :param u: System input at current time
        :param w: Process noise vector at current time
        :param h: Observer step width
        :return State vector at next time step
        """
        raise NotImplementedError

    def output_func(self, x: Array) -> Array:
        """
        System output function h(x)
        :param x: State vector at current time
        :return System output vector y
        """
        raise NotImplementedError


class ExtendedKalmanFilter:
    """
    EKF algorithm that can be used either in a standalone EKF observer or embedded in other observers (used for MHE)
    """

    def __init__(self, observer_model: ObserverModel, Q, R, x0, P0):
        self.observer_model = observer_model
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0

    def step(self, u, z, h):
        x_prev = self.x
        P_prev = self.P

        if u is not None:
            # we are at k >= 1
            # Predict
            f_jacobian = self.observer_model.f_jacobian(x_prev, u, h)
            x_priori = self.observer_model.discrete_state_func(x_prev, u, np.zeros(self.observer_model.state_dim), h)
            P_priori = f_jacobian @ P_prev @ f_jacobian.T + self.Q
        else:
            # we are at k = 0 and can therefore only update our initial state based on the current measurement
            x_priori = x_prev
            P_priori = P_prev

        h_jacobian = self.observer_model.h_jacobian(x_prev)

        # Update
        K = P_priori @ h_jacobian.T @ np.linalg.inv(h_jacobian @ P_priori @ h_jacobian.T + self.R)
        self.x = x_priori + K @ (z - self.observer_model.output_func(x_priori))
        self.P = (np.identity(self.observer_model.state_dim) - K @ h_jacobian) @ P_priori

        return self.x, self.P


class ExtendedKalmanFilterObserver(pm.Observer):
    """
    Extended Kalman Filter with analytical jacobians, based on a forward Euler discretization of the continuous time
    nonlinear system model
    """

    public_settings = OrderedDict([
        ("initial state", [0.0, 0.0, 0.0, 0.0]),
        ("sqrt Qii", [0.01, 0.01, 0.01, 0.01]),
        ("sqrt Rii", [0.001]),
        ("tick divider", 10)
    ])

    def __init__(self, settings, observer_model):
        settings.update(output_dim=observer_model.state_dim)
        super().__init__(settings)

        Q_diagonal = np.array(self._settings['sqrt Qii'], dtype=np.float32) ** 2
        Q = np.diag(Q_diagonal)
        R_diagonal = np.array(self._settings['sqrt Rii'], dtype=np.float32) ** 2
        R = np.diag(R_diagonal)

        P0 = np.identity(observer_model.state_dim, dtype=np.float32)
        x0 = np.array(self._settings["initial state"], dtype=np.float32)

        self.filter_algorithm = ExtendedKalmanFilter(observer_model, Q, R, x0, P0)

    def _observe(self, time, system_input, system_output):
        """
        Observer output function that returns the state estimate as well as the estimated variances for each state
        """
        h = self.step_width
        x, P = self.filter_algorithm.step(system_input, system_output, h)

        return np.concatenate((x, P.diagonal()))
