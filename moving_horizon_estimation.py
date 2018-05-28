from functools import partial
from typing import *
import pymoskito as pm
import numpy as np
from scipy.optimize import minimize
import scipy
from collections import OrderedDict
from extended_kalman_filter import ObserverModel, ExtendedKalmanFilter
from easytime import Timer

import cvxopt as co

Array = np.ndarray


class MovingHorizonEstimator(pm.Observer):
    public_settings = OrderedDict([
        ("initial state", [0, 0, 0, 0]),
        ("P0ii", [0.01, 0.01, 0.01, 0.1]),
        ("Qii", [1e-5, 1e-5, 1e-5, 1e-5]),
        ("Rii", [1e-6]),
        ("N", 2),
        ("max iter", 100),
        ("constraints", True),
        ("quad", True),
        ("rti", True),
        ("tick divider", 5)
    ])

    def __init__(self, settings, observer_model):
        self.observer_model: ObserverModel = observer_model

        settings.update(output_dim=self.observer_model.state_dim)
        super().__init__(settings)

        Q_diagonal = np.array(self._settings['Qii'], dtype=np.float64)
        self.Q = np.diag(Q_diagonal)
        self.Q_inv: Array = np.linalg.inv(self.Q)
        self.Q_inv_sqrt: Array = np.sqrt(self.Q_inv)
        R_diagonal = np.array(self._settings['Rii'], dtype=np.float64)
        self.R = np.diag(R_diagonal)
        self.R_inv: Array = np.linalg.inv(self.R)
        self.R_inv_sqrt: Array = np.sqrt(self.R_inv)

        P0_diagonal = np.array(self._settings['P0ii'], dtype=np.float64)
        P0 = np.diag(P0_diagonal)
        self.P_rti = np.sqrt(np.linalg.inv(P0))
        """Weight matrix for arrival cost in RTI scheme"""
        self.xL_rti = self._settings['initial state']
        """State at left horizon edge for arrival cost in RTI scheme"""

        self.N: int = self._settings['N']
        self.max_iter: int = self.settings['max iter']
        self.use_constraints = self._settings['constraints']
        self.last_us = np.empty((0, self.observer_model.input_dim), dtype=np.float64)
        """Sequence of N last system input vectors (grows for the first few simulation steps)"""
        self.last_ys = np.empty((0, self.observer_model.output_dim), dtype=np.float64)
        """Sequence of N+1 last system measurement vectors (grows for the first few simulation steps)"""
        self.last_xs = np.array([self._settings["initial state"]], dtype=np.float64)
        """Sequence of N+1 state vectors from the last observation"""
        self.last_ws = np.empty((0, self.observer_model.state_dim), dtype=np.float64)
        """Sequence of N process noise vectors from the last observation"""
        self.last_estimates = np.empty((0, self.observer_model.state_dim), dtype=np.float64)
        """Sequence of N+1 moving horizon estimate results (the last values on their horizons)"""
        self.xL_ekf = self._settings['initial state']
        """A priori state estimate at T-N for arrival cost using EKF filtering"""

        self.use_quadratic = self._settings['quad']
        self.rti = self._settings['rti']

        self.k = 0

        self.ekf = ExtendedKalmanFilter(self.observer_model, self.Q, self.R, self.last_xs[0], P0)

        self.timer = Timer()

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
        xs = np.empty((N + 1, self.observer_model.state_dim), dtype=np.float64)  # the initial value and N calculated values
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
        cost: float = (chis[0] - x0) @ P_inv @ (chis[0] - x0)

        for i in range(N+1):
            v = self.observer_model.output_func(chis[i]) - ys[i]
            cost += v @ self.R_inv @ v

            if i < N:
                cost += omegas[i] @ self.Q_inv @ omegas[i]

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
        cost_jacobian[:dim] = (chis[0] - x0) @ (P_inv + P_inv.T) + 2 * (self.observer_model.output_func(chis[0]) - ys[0]) @ self.R_inv @ self.observer_model.h_jacobian(chis[0])

        for i in range(N):
            cost_jacobian[(i+1)*dim:(i+2)*dim] = 2 * (self.observer_model.output_func(chis[i + 1]) - ys[i+1]) @ self.R_inv @ self.observer_model.h_jacobian(chis[i+1])
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

        return self.observer_model.state_ineq_constraints[constraint_index][0](chis[x_index])

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

        return self.observer_model.state_eq_constraints[constraint_index][0](chis[x_index])

    def state_eq_constraint_jac(self, x_index: int, constraint_index: int, chis_and_omegas: Array) -> Array:
        chis, _, N = self.reshape_opt_vector(chis_and_omegas)
        state_dim = self.observer_model.state_dim

        constraint = self.observer_model.state_eq_constraints[constraint_index]
        jac = np.zeros((constraint[1], (N*2+1) * state_dim), dtype=np.float64)
        constraint_jac = constraint[2](chis[x_index])

        jac[:, x_index*state_dim:(x_index+1)*state_dim] = constraint_jac

        return jac

    def state_ineq_constraint_jac(self, x_index: int, constraint_index: int, chis_and_omegas: Array) -> Array:
        chis, _, N = self.reshape_opt_vector(chis_and_omegas)
        state_dim = self.observer_model.state_dim

        constraint = self.observer_model.state_ineq_constraints[constraint_index]
        jac = np.zeros((constraint[1], (N*2+1) * state_dim), dtype=np.float64)
        constraint_jac = constraint[2](chis[x_index])

        jac[:, x_index*state_dim:(x_index+1)*state_dim] = constraint_jac

        return jac

    def _observe(self, time, system_input: Array, system_output: Array) -> Array:
        """
        Observer output function that returns the state estimate as well as the estimated variances for each state
        """

        timer = self.timer

        if self.k % 100 == 0:
            timer.print()

        timer.tic("Total observation")

        y = system_output  # y[k]
        u = system_input  # u[k-1]

        state_dim = self.observer_model.state_dim
        output_dim = self.observer_model.output_dim

        timer.tic("Cache updates")
        current_cache_size = self.last_us.shape[0]

        if self.k > 0:
            if current_cache_size < self.N:
                self.last_us = np.resize(self.last_us, (current_cache_size + 1, self.observer_model.input_dim))
            else:
                self.last_us[:-1] = self.last_us[1:]

            self.last_us[-1] = u  # u[k-N .. k-1]

        if self.k <= self.N:
            self.last_ys = np.resize(self.last_ys, (self.last_ys.shape[0] + 1, output_dim))
        else:
            self.last_ys[:-1] = self.last_ys[1:]

        self.last_ys[-1] = y  # y[k-N .. k]
        timer.toc()

        N = self.last_us.shape[0]
        # Initialize optimization vector by shifting
        timer.tic("Shifting")
        if self.k > 0:
            new_state_init_guess = self.observer_model.discrete_state_func(self.last_xs[-1], u,
                                                                           np.zeros(state_dim),
                                                                           self.step_width)

            if self.k > self.N:
                xs = np.concatenate((self.last_xs[1:], np.array([new_state_init_guess])))
                ws = np.concatenate((self.last_ws[1:], np.zeros((1, state_dim))))
            else:
                xs = np.concatenate((self.last_xs, np.array([new_state_init_guess])))
                ws = np.concatenate((self.last_ws, np.zeros((1, state_dim))))
        else:
            xs = self.last_xs
            ws = self.last_ws

        nx = xs.shape[0]
        timer.toc()

        if self.rti:
            # Calculating the residual vector
            timer.tic("Residual vector")
            residual = np.empty(nx * (state_dim + output_dim), dtype=np.float64)
            residual[:state_dim] = self.P_rti @ (xs[0] - self.xL_rti)
            residual[state_dim:(state_dim+output_dim)] = self.R_inv_sqrt @ (self.last_ys[0] - self.observer_model.output_func(xs[0]))

            for j in range(1, nx):
                residual[j*(state_dim+output_dim):j*(state_dim+output_dim)+state_dim] = self.Q_inv_sqrt @ (xs[j] - self.observer_model.discrete_state_func(xs[j-1], self.last_us[j-1], np.zeros(state_dim), self.step_width))
                residual[j*(state_dim+output_dim)+state_dim:(j+1)*(state_dim+output_dim)] = self.R_inv_sqrt @ (self.last_ys[j] - self.observer_model.output_func(xs[j]))

            timer.toc()
            # Calculating the residual jacobian
            timer.tic("Residual jacobians")
            residual_jacobian = co.spmatrix(0, [], [], (residual.size, state_dim*nx), 'd')
            for i_time in range(0, nx):
                if i_time == 0:
                    residual_jacobian[:state_dim, :state_dim] = self.P_rti
                else:
                    residual_jacobian[(state_dim + output_dim) * i_time:(state_dim + output_dim) * i_time + state_dim,
                                      state_dim * i_time:state_dim * (i_time + 1)] = self.Q_inv_sqrt

                second_block = co.matrix(- self.R_inv_sqrt @ self.observer_model.h_jacobian(xs[i_time]))
                residual_jacobian[(state_dim + output_dim) * i_time + state_dim:(state_dim + output_dim) * i_time + state_dim + output_dim, state_dim * i_time:state_dim * (i_time + 1)] = second_block

                if i_time != nx - 1:
                    third_block = co.matrix(- self.Q_inv_sqrt @ self.observer_model.f_jacobian(xs[i_time], self.last_us[i_time], self.step_width))
                    residual_jacobian[(state_dim + output_dim) * i_time + state_dim + output_dim:(state_dim + output_dim) * i_time + state_dim + output_dim + state_dim, state_dim * i_time:state_dim * (i_time + 1)] = third_block

            P_QP = residual_jacobian.T * residual_jacobian
            q_QP = residual_jacobian.T * co.matrix(residual)

            timer.toc()
            # Calculating the constraint values
            timer.tic("Constraint values")
            # TODO: Pre-allocate properly
            all_eq_constraints = np.empty(0)

            for i_time in range(0, nx):
                for i_constraint in range(0, len(self.observer_model.state_eq_constraints)):
                    new_constraint_values = self.observer_model.state_eq_constraints[i_constraint][0](xs[i_time])
                    all_eq_constraints = np.concatenate((all_eq_constraints, new_constraint_values))

            all_ineq_constraints = np.empty(0)
            for i_time in range(0, nx):
                for i_constraint in range(0, len(self.observer_model.state_ineq_constraints)):
                    new_constraint_values = self.observer_model.state_ineq_constraints[i_constraint][0](xs[i_time])
                    all_ineq_constraints = np.concatenate((all_ineq_constraints, new_constraint_values))

            b_QP = co.matrix(-all_eq_constraints)
            h_QP = co.matrix(all_ineq_constraints)

            timer.toc()
            # Calculating the constraint jacobians
            timer.tic("Constraint jacobians")
            total_eq_constraint_dim = sum([c[1] for c in self.observer_model.state_eq_constraints])
            total_eq_constraint_jac = co.spmatrix(0, [], [], (total_eq_constraint_dim * nx, state_dim * nx), 'd')

            for i_time in range(0, nx):
                in_time_offset = 0
                for i_constraint in range(0, len(self.observer_model.state_eq_constraints)):
                    cur_constraint_dim = self.observer_model.state_eq_constraints[i_constraint][1]
                    new_constraint_jac = self.observer_model.state_eq_constraints[i_constraint][2](xs[i_time])
                    total_eq_constraint_jac[total_eq_constraint_dim*i_time+in_time_offset:total_eq_constraint_dim*i_time+in_time_offset+cur_constraint_dim,
                                            state_dim*i_time:state_dim*(i_time+1)] = new_constraint_jac
                    in_time_offset += cur_constraint_dim

            total_ineq_constraint_dim = sum([c[1] for c in self.observer_model.state_ineq_constraints])
            total_ineq_constraint_jac = co.spmatrix(0, [], [], (total_ineq_constraint_dim * nx, state_dim * nx), 'd')

            for i_time in range(0, nx):
                in_time_offset = 0
                for i_constraint in range(0, len(self.observer_model.state_ineq_constraints)):
                    cur_constraint_dim = self.observer_model.state_ineq_constraints[i_constraint][1]
                    new_constraint_jac = self.observer_model.state_ineq_constraints[i_constraint][2](xs[i_time])
                    total_ineq_constraint_jac[total_ineq_constraint_dim * i_time + in_time_offset:total_ineq_constraint_dim * i_time + in_time_offset + cur_constraint_dim,
                                              state_dim * i_time:state_dim * (i_time + 1)] = new_constraint_jac
                    in_time_offset += cur_constraint_dim

            A_QP = total_eq_constraint_jac
            G_QP = - total_ineq_constraint_jac
            timer.toc()

            # Solve QP
            timer.tic("QP subproblem")
            co.solvers.options['show_progress'] = False
            opt_result = co.solvers.qp(P_QP, q_QP, G_QP, h_QP, A_QP, b_QP)

            # Calculate updated state estimates in horizon
            deltax = np.array(opt_result['x']).reshape((nx, state_dim))
            timer.toc()

            self.last_xs = xs + deltax
            estimated_state = self.last_xs[-1]

            # Update arrival cost parameters
            if self.k >= self.N:
                timer.tic("arrival cost")
                timer.tic("arrival cost matrices")
                # We only start updating the arrival cost once we're actually moving the estimation horizon
                xL = self.last_xs[0]
                F = self.observer_model.f_jacobian(xL, self.last_us[0], self.step_width)
                H = self.observer_model.h_jacobian(xL)

                Phi = np.zeros((2*state_dim+output_dim, 2*state_dim))
                Phi[:state_dim,:state_dim] = self.P_rti
                Phi[state_dim:state_dim+output_dim,:state_dim] = -self.R_inv_sqrt @ H
                Phi[state_dim+output_dim:, :state_dim] = - self.Q_inv_sqrt @ F
                Phi[state_dim+output_dim:, state_dim:] = self.Q_inv_sqrt

                minus_b = np.empty(2*state_dim+output_dim)
                minus_b[:state_dim] = - self.P_rti @ self.xL_rti
                minus_b[state_dim:state_dim+output_dim] = self.R_inv_sqrt @ (self.last_ys[0] - self.observer_model.output_func(xL) + H @ xL)
                minus_b[state_dim+output_dim:] = - self.Q_inv_sqrt @ (self.observer_model.discrete_state_func(xL, self.last_us[0], np.zeros(state_dim), self.step_width) - F @ xL)

                timer.toc()

                timer.tic("QR decomposition")
                Q_decomp, R_decomp = scipy.linalg.qr(Phi)
                timer.toc()

                timer.tic("final arrival cost params")
                rho = Q_decomp.T @ minus_b
                rho2 = rho[state_dim:2*state_dim]
                R2 = R_decomp[state_dim:2*state_dim, state_dim:]

                self.P_rti = R2
                self.xL_rti = scipy.linalg.solve(R2, -rho2)
                timer.toc()
                timer.toc()

            # Output
            self.k += 1
            timer.toc()

            P_covariance = np.linalg.inv(self.P_rti @ self.P_rti)
            return np.concatenate((estimated_state, P_covariance.diagonal()))
        else:
            constraints = []
            timer.tic("Constraint jacobians")
            if self.use_constraints:
                # we need to add constraints to every state vector contained in the current optimization horizon
                for time_index in range(N + 1):
                    # define equality constraints
                    for constraint_index in range(len(self.observer_model.state_eq_constraints)):
                        constraint_lambda = partial(self.state_eq_constraint_wrapper, time_index, constraint_index)
                        constraint_jac = partial(self.state_eq_constraint_jac, time_index, constraint_index)
                        new_constraint = {'type': 'eq', 'fun': constraint_lambda, 'jac': constraint_jac}
                        constraints.append(new_constraint)

                    # define inequality constraints
                    for constraint_index in range(len(self.observer_model.state_ineq_constraints)):
                        constraint_lambda = partial(self.state_ineq_constraint_wrapper, time_index, constraint_index)
                        constraint_jac = partial(self.state_ineq_constraint_jac, time_index, constraint_index)
                        new_constraint = {'type': 'ineq', 'fun': constraint_lambda, 'jac': constraint_jac}
                        constraints.append(new_constraint)
            timer.toc()

            timer.tic("State function constraint jacobians")
            # add constraints to make the state sequence adhere to the model's state transition function
            for i in range(N):
                state_constraint_lambda = partial(self.discrete_state_func_constraint, self.last_us[i], i)
                state_constraint_jac = partial(self.discrete_state_func_constraint_jac, self.last_us[i], i)
                new_constraint = {'type': 'eq', 'fun': state_constraint_lambda, 'jac': state_constraint_jac}
                constraints.append(new_constraint)
            timer.toc()

            timer.tic("Inverting EKF covariance")
            P_inv = np.linalg.inv(self.ekf.P)
            timer.toc()
            x0_tilde = self.xL_ekf

            # pre-fill constant values in cost and cost jacobian calculations
            # the only input of these partials that remains is the optimization vector
            if self.use_quadratic:
                cost_lambda = partial(self.cost_functional_quadratic, x0_tilde, P_inv, self.last_ys, N)
                cost_jac_lambda = partial(self.cost_jacobian_quadratic, x0_tilde, P_inv, self.last_ys, N)
            else:
                P_inv_sqrt = np.sqrt(np.abs(P_inv))
                cost_lambda = partial(self.cost_functional_abs, x0_tilde, P_inv_sqrt, self.last_ys, N)
                cost_jac_lambda = partial(self.cost_jacobian_abs, x0_tilde, P_inv_sqrt, self.last_ys, N)

            timer.tic("Scipy minimize")
            opt_result: scipy.optimize.OptimizeResult = minimize(cost_lambda, np.concatenate((xs.reshape((xs.size,)), ws.reshape((ws.size,)))),
                                                                 jac=cost_jac_lambda, constraints=constraints, method='SLSQP', options={'maxiter': self.max_iter})
            timer.toc()
            opt_xs = opt_result.x[:(N+1)*state_dim].reshape((N+1, state_dim))
            opt_ws = opt_result.x[(N+1)*state_dim:].reshape((N, state_dim))
            estimated_state = opt_xs[-1]

            self.last_xs = opt_xs  # x[k-N .. k]
            self.last_ws = opt_ws

            if self.k <= self.N:
                self.last_estimates = np.resize(self.last_estimates, (self.last_estimates.shape[0] + 1, state_dim))
            else:
                self.last_estimates[:-1] = self.last_estimates[1:]

            self.last_estimates[-1] = estimated_state

            if self.k >= self.N:
                # Wir haben keinen FIE mehr sondern müssen jetzt die arrival cost für den nächsten Schritt approximieren
                timer.tic("EKF Update")
                self.ekf.step(self.last_us[0], self.last_ys[1], self.step_width, x_prev=self.last_estimates[0], timer=timer)  # P[k-N+1]
                self.xL_ekf = self.observer_model.discrete_state_func(self.last_estimates[0], self.last_us[0], np.zeros(state_dim), self.step_width)
                timer.toc()

            self.k += 1

            timer.toc()
            return np.concatenate((estimated_state, self.ekf.P.diagonal()))
