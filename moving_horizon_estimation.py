from functools import partial
from typing import *
import pymoskito as pm
import numpy as np
from scipy.optimize import minimize
import scipy
from collections import OrderedDict
from extended_kalman_filter import ObserverModel, ExtendedKalmanFilter

Array = np.ndarray


class MovingHorizonEstimator(pm.Observer):
    public_settings = OrderedDict([
        ("initial state", [0, 0, 0, 0]),
        ("sqrt Qii", [0.01, 0.01, 0.01, 0.01]),
        ("sqrt Rii", [0.001]),
        ("N", 1),
        ("max iter", 15),
        ("constraints", False),
        ("quad", True)
    ])

    def __init__(self, settings, observer_model):
        self.observer_model: ObserverModel = observer_model

        settings.update(output_dim=self.observer_model.state_dim)
        settings['tick divider'] = 10
        super().__init__(settings)

        Q_diagonal = np.array(self._settings['sqrt Qii'], dtype=np.float32) ** 2
        self.Q = np.diag(Q_diagonal)
        self.Q_inv: Array = np.linalg.inv(self.Q)
        self.Q_inv_sqrt: Array = np.sqrt(self.Q_inv)
        R_diagonal = np.array(self._settings['sqrt Rii'], dtype=np.float32) ** 2
        self.R = np.diag(R_diagonal)
        self.R_inv: Array = np.linalg.inv(self.R)
        self.R_inv_sqrt: Array = np.sqrt(self.R_inv)

        self.N: int = self._settings['N']
        self.max_iter: int = self.settings['max iter']
        self.use_constraints = self._settings['constraints']
        self.last_us = np.empty((0, self.observer_model.input_dim), dtype=np.float32)
        """Sequence of N last system input vectors (grows for the first few simulation steps)"""
        self.last_ys = np.empty((0, self.observer_model.output_dim), dtype=np.float32)
        """Sequence of N last system measurement vectors (grows for the first few simulation steps)"""
        self.last_xs = np.array([self._settings["initial state"]], dtype=np.float32)
        """Sequence of N last best state estimates (grows for the first few simulation steps)"""

        self.use_quadratic = self._settings['quad']

        self.k = 0

        self.ekf = ExtendedKalmanFilter(self.observer_model, self.Q, self.R, self.last_xs[0], np.identity(self.observer_model.state_dim))

    def reshape_opt_vector(self, chis_and_omegas: Array) -> Tuple[Array, Array, int]:
        """
        Reshapes the 1D optimization vector into two sequences of state estimates and process noise
        :param chis_and_omegas: 1D array containing in sequence: components of N+1 state estimate vectors, components
                                of N process noise vectors
        :return: Sequence of state estimates, sequence of process noise, horizon length
        """

        dim = self.observer_model.state_dim
        N: int = ((chis_and_omegas.size // dim) - 1) // 2
        chis: Array = chis_and_omegas[:dim * (N + 1)].reshape(N + 1, dim)
        omegas: Array = chis_and_omegas[dim * (N + 1):].reshape(N, dim)
        return chis, omegas, N

    def calc_states(self, x0: Array, us: Array, ws: Array, N: int):
        """
        Takes an initial state, a sequence of inputs and disturbance vectors to calculate N additional state vectors
        :param x0: Initial state vector as 1D-Array
        :param us: Sequence of N input vectors from 0 to N - 1
        :param ws: Sequence of N disturbance vectors from 0 to N - 1
        :param N: How many additional states to compute
        """
        xs = np.empty((N + 1, self.observer_model.state_dim), dtype=np.float32)  # the initial value and N calculated values
        xs[0] = x0

        for i in range(N):
            xs[i + 1] = self.observer_model.discrete_state_func(xs[i], us[i], ws[i], self.step_width)

        return xs

    def cost_functional_quadratic(self, x0: Array, P_inv: Array, ys: Array, N: int, chis_and_omegas: Array) -> float:
        """
        Calculates the MHE cost for a finite horizon from a sequence of estimated states, process noise vectors and
        measurements. Arrival cost is quadratic, weighted with an estimate covariance matrix.

        :param x0: Best estimate for initial state vector
        :param P_inv: Inverse of state estimate covariance matrix
        :param ys: Sequence of N output measurement vectors
        :param N: Horizon size
        :param chis_and_omegas: 1D array containing in sequence: components of N+1 state estimate vectors, components
                                of N process noise vectors
        :return: Total cost for this horizon
        """
        chis, omegas, _ = self.reshape_opt_vector(chis_and_omegas)
        # Am Anfang ist x0 bekannt, daher P_inv -> inf
        cost: float = (x0 - chis[0]) @ P_inv @ (x0 - chis[0])

        for i in range(N):
            cost += omegas[i] @ self.Q_inv @ omegas[i]
            v = self.observer_model.output_func(chis[i + 1]) - ys[i]
            cost += v @ self.R_inv @ v

        return cost

    def cost_functional_abs(self, x0: Array, P_inv_sqrt: Array, ys: Array, N: int, chis_and_omegas: Array) -> float:
        """
        Calculates the MHE cost for a finite horizon from a sequence of estimated states, process noise vectors and
        measurements. Arrival cost is linear absolute, weighted with an estimate covariance matrix.

        :param x0: Best estimate for initial state vector
        :param P_inv_sqrt: Inverse of state estimate covariance matrix
        :param ys: Sequence of N output measurement vectors
        :param N: Horizon size
        :param chis_and_omegas: 1D array containing in sequence: components of N+1 state estimate vectors, components
                                of N process noise vectors
        :return: Total cost for this horizon
        """
        chis, omegas, _ = self.reshape_opt_vector(chis_and_omegas)
        # Am Anfang ist x0 bekannt, daher P_inv -> inf
        cost: float = np.linalg.norm(P_inv_sqrt @ (x0 - chis[0]), 1)

        for i in range(N):
            cost += np.linalg.norm(self.Q_inv_sqrt @ omegas[i], 1)
            v = self.observer_model.output_func(chis[i + 1]) - ys[i]
            cost += np.linalg.norm(self.R_inv_sqrt @ v, 1)

        return cost

    def cost_jacobian_quadratic(self, x0: Array, P_inv: Array, ys: Array, N: int, chis_and_omegas: Array) -> Array:
        """
        Analytically computes the jacobian of the cost function with respect to all elements of the optimization vector
        :param x0: Initial best state estimate
        :param P_inv: Inverse of state estimate covariance matrix
        :param ys: Sequence of N measurements
        :param N: Horizon length
        :param chis_and_omegas: 1D array containing in sequence: components of N+1 state estimate vectors, components
                                of N process noise vectors
        :return: Jacobian of cost function
        """
        dim = self.observer_model.state_dim
        chis, omegas, _ = self.reshape_opt_vector(chis_and_omegas)

        cost_jacobian = np.empty(chis_and_omegas.size)
        cost_jacobian[:dim] = (chis[0] - x0) @ (P_inv + P_inv.T)

        for i in range(N):
            cost_jacobian[(i+1)*dim:(i+2)*dim] = 2 * (self.observer_model.output_func(chis[i + 1]) - ys[i]) @ self.R_inv @ self.observer_model.h_jacobian(chis[i+1])
            cost_jacobian[(i+1+N)*dim:(i+2+N)*dim] = 2 * omegas[i] @ self.Q_inv

        return cost_jacobian

    def cost_jacobian_abs(self, x0: Array, P_inv_sqrt: Array, ys: Array, N: int, chis_and_omegas: Array) -> Array:
        """
        Analytically computes the jacobian of the cost function with respect to all elements of the optimization vector
        :param x0: Initial best state estimate
        :param P_inv_sqrt: Inverse of state estimate covariance matrix
        :param ys: Sequence of N measurements
        :param N: Horizon length
        :param chis_and_omegas: 1D array containing in sequence: components of N+1 state estimate vectors, components
                                of N process noise vectors
        :return: Jacobian of cost function
        """
        dim = self.observer_model.state_dim
        chis, omegas, _ = self.reshape_opt_vector(chis_and_omegas)

        cost_jacobian = np.empty(chis_and_omegas.size)
        cost_jacobian[:dim] = np.abs(P_inv_sqrt) @ np.sign(chis[0] - x0)

        for i in range(N):
            cost_jacobian[(i+1)*dim:(i+2)*dim] = np.abs(self.R_inv_sqrt) @ np.sign(self.observer_model.output_func(chis[i + 1]) - ys[i]) @ self.observer_model.h_jacobian(chis[i+1])
            cost_jacobian[(i+1+N)*dim:(i+2+N)*dim] = np.abs(self.Q_inv_sqrt) @ np.sign(omegas[i])

        return cost_jacobian

    def discrete_state_func_constraint(self, u: Array, x_index: int, chis_and_omegas: Array) -> Array:
        """
        Equality constraint function that makes sure the discrete state function is being honored
        :param u: System input vector at x_index
        :param x_index: Time index in chis_and_omegas of state that should be checked
        :param chis_and_omegas: 1D array containing in sequence: components of N+1 state estimate vectors, components
                                of N process noise vectors
        :return: Constraint function value
        """

        chis, omegas, _ = self.reshape_opt_vector(chis_and_omegas)

        return self.observer_model.discrete_state_func(chis[x_index], u, omegas[x_index], self.step_width) - chis[x_index+1]

    def discrete_state_func_constraint_jac(self, u: Array, x_index: int, chis_and_omegas: Array) -> Array:
        """
        Analytical jacobian of discrete state function constraint with respect to all elements of the optimization vector
        :param u: System input vector at x_index
        :param x_index: Time index in chis_and_omegas of state that should be checked
        :param chis_and_omegas: 1D array containing in sequence: components of N+1 state estimate vectors, components
                                of N process noise vectors
        :return: Jacobian of constraint function
        """

        dim = self.observer_model.state_dim
        chis, _, N = self.reshape_opt_vector(chis_and_omegas)

        jac = np.zeros((dim, (2*N+1)*dim))

        jac[:, x_index*dim:(x_index+1)*dim] = self.observer_model.f_jacobian(chis[x_index], u, self.step_width)
        jac[:, (x_index+1)*dim:(x_index+2)*dim] = -np.eye(dim)
        jac[:, (N+1+x_index)*dim:(N+1+x_index+1)*dim] = np.eye(dim)

        return jac

    def state_ineq_constraint_wrapper(self, x_index: int, constraint_index: int, chis_and_omegas: Array) -> Union[float, Array]:
        """
        Evaluates a specific state inequality constraint with the vector that is being optimized in the format that
        scipy.optimize.minimize uses.
        :param x_index: Index of the state vector for which the constraint should be checked
        :param constraint_index: Constraint index in the list supplied by the observer model
        :param chis_and_omegas: 1D array containing in sequence: components of N+1 state estimate vectors, components
                                of N process noise vectors
        :return: Constraint function value (scalar or vector)
        """

        chis, _, _ = self.reshape_opt_vector(chis_and_omegas)

        return self.observer_model.state_ineq_constraints[constraint_index](chis[x_index])

    def state_eq_constraint_wrapper(self, x_index: int, constraint_index: int, chis_and_omegas: Array) -> Union[float, Array]:
        """
        Evaluates a specific state equality constraint with the vector that is being optimized in the format that
        scipy.optimize.minimize uses.
        :param x_index: Index of the state vector for which the constraint should be checked
        :param constraint_index: Constraint index in the list supplied by the observer model
        :param chis_and_omegas: 1D array containing in sequence: components of N+1 state estimate vectors, components
                                of N process noise vectors
        :return: Constraint function value (scalar or vector)
        """

        chis, _, _ = self.reshape_opt_vector(chis_and_omegas)

        return self.observer_model.state_eq_constraints[constraint_index](chis[x_index])

    def _observe(self, time, system_input: Array, system_output: Array) -> Array:
        """
        Observer output function that returns the state estimate as well as the estimated variances for each state
        """
        y = system_output  # y[k]
        u = system_input  # u[k-1]

        state_dim = self.observer_model.state_dim
        if self.k == 0:
            # Wir können nur den Initialwert ausgeben
            # y[0] schreiben wir nicht mit, da wir es sowieso nicht für Messupdates verwenden
            self.k += 1
            return np.concatenate((self.last_xs[0], np.zeros(state_dim)))

        if self.k > self.N:
            # Wir haben keinen FIE mehr sondern müssen jetzt die arrival cost approximieren
            # Es ist wichtig, dass wir das vor dem update des u und y Speichers machen, da sonst das erste Element, das
            # wir fürs EKF Update brauchen verloren geht
            _, P = self.ekf.step(self.last_us[0], self.last_ys[0], self.step_width)  # P[k-N]
            P_inv = np.linalg.inv(P)
        else:
            # Wichtung der Arrival Cost so, dass möglichst exakt der Initialzustand genommen wird
            P_inv = np.identity(state_dim) * 1e6

        current_cache_size = self.last_us.shape[0]

        if current_cache_size < self.N:
            self.last_us = np.resize(self.last_us, (current_cache_size + 1, self.observer_model.input_dim))
            self.last_ys = np.resize(self.last_ys, (current_cache_size + 1, self.observer_model.output_dim))
        else:
            self.last_us[:-1] = self.last_us[1:]
            self.last_ys[:-1] = self.last_ys[1:]

        self.last_us[-1] = u  # u[k-N .. k-1]
        self.last_ys[-1] = y  # y[k-N+1 .. k]

        N = self.last_us.shape[0]
        x0 = self.last_xs[0]

        constraints = []
        if self.use_constraints:
            # we need to add constraints to every state vector contained in the current optimization horizon
            for time_index in range(N + 1):
                # define equality constraints
                for constraint_index in range(len(self.observer_model.state_eq_constraints)):
                    constraint_lambda = partial(self.state_eq_constraint_wrapper, time_index, constraint_index)
                    new_constraint = {'type': 'eq', 'fun': constraint_lambda}
                    constraints.append(new_constraint)

                # define inequality constraints
                for constraint_index in range(len(self.observer_model.state_ineq_constraints)):
                    constraint_lambda = partial(self.state_ineq_constraint_wrapper, time_index, constraint_index)
                    new_constraint = {'type': 'ineq', 'fun': constraint_lambda}
                    constraints.append(new_constraint)

        # add constraints to make the state sequence adhere to the model's state transition function
        for i in range(N):
            state_constraint_lambda = partial(self.discrete_state_func_constraint, self.last_us[i], i)
            state_constraint_jac = partial(self.discrete_state_func_constraint_jac, self.last_us[i], i)
            new_constraint = {'type': 'eq', 'fun': state_constraint_lambda, 'jac': state_constraint_jac}
            constraints.append(new_constraint)

        # pre-fill constant values in cost and cost jacobian calculations
        # the only input of these partials that remains is the optimization vector
        if self.use_quadratic:
            cost_lambda = partial(self.cost_functional_quadratic, x0, P_inv, self.last_ys, N)
            cost_jac_lambda = partial(self.cost_jacobian_quadratic, x0, P_inv, self.last_ys, N)
        else:
            P_inv_sqrt = np.sqrt(np.abs(P_inv))
            cost_lambda = partial(self.cost_functional_abs, x0, P_inv_sqrt, self.last_ys, N)
            cost_jac_lambda = partial(self.cost_jacobian_abs, x0, P_inv_sqrt, self.last_ys, N)

        new_state_init_guess = self.observer_model.discrete_state_func(self.last_xs[-1], u, np.zeros(self.observer_model.state_dim), self.step_width)
        opt_result: scipy.optimize.OptimizeResult = minimize(cost_lambda, np.concatenate((self.last_xs.reshape((self.last_xs.size,)),
                                                                                          new_state_init_guess, np.zeros(state_dim * N))), jac=cost_jac_lambda, constraints=constraints, method='SLSQP', options={'maxiter': self.max_iter})
        estimated_state = opt_result.x[N * state_dim:(N+1)*state_dim]

        if self.last_xs.shape[0] < self.N:
            self.last_xs = np.resize(self.last_xs, (self.last_xs.shape[0] + 1, state_dim))
        else:
            self.last_xs[:-1] = self.last_xs[1:]

        self.last_xs[-1] = estimated_state # x[k-N + 1 .. k]

        self.k += 1

        return np.concatenate((estimated_state, self.ekf.P.diagonal()))
