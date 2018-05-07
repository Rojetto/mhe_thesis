﻿# -*- coding: utf-8 -*-
from collections import OrderedDict

import numpy as np
import sympy as sp
from scipy.integrate import ode
from sympy import sin, cos

import pymoskito as pm
from pymoskito.tools import swap_cols, swap_rows

from extended_kalman_filter import ObserverModel, ExtendedKalmanFilterObserver
from moving_horizon_estimation import MovingHorizonEstimator
from . import settings as st
from .linearization import linearise_system

Array = np.ndarray


class BallBeamObserverModel(ObserverModel):
    """
    System model of ball beam system to be used by observers
    """

    def __init__(self):
        self.M = st.M
        self.R = st.R
        self.J = st.J
        self.Jb = st.Jb
        self.G = st.G
        self.B = self.M / (self.Jb / self.R ** 2 + self.M)

        f, h = self.derive_jacobians()

        super().__init__(4, 1, 1, f, h, [self.beam_edges_constraint, self.beam_angle_constraint])

    def derive_jacobians(self):
        x1, x2, x3, x4, tau, h, B, G, M, J, Jb = sp.symbols("x_1 x_2 x_3 x_4 tau h B G M J J_b")

        x = sp.Matrix([x1, x2, x3, x4])
        f_cont = sp.Matrix([x2,
                            B * (x1 * x4 ** 2 - G * sp.sin(x3)),
                            x4,
                            (tau - 2 * M * x1 * x2 * x4 - M * G * x1 * sp.cos(x3)) / (M * x1 ** 2 + J + Jb)])
        f = x + h * f_cont
        F = f.jacobian(x).subs({B: self.B, G: self.G, M: self.M, J: self.J, Jb: self.Jb})
        scalar_lambda = sp.lambdify((x1, x2, x3, x4, tau, h), F)

        f_jacobian = lambda x, u, h: scalar_lambda(x[0], x[1], x[2], x[3], u[0], h)
        h_jacobian = lambda x: np.array([[1, 0, 0, 0]], dtype=np.float32)

        return f_jacobian, h_jacobian

    def discrete_state_func(self, x, u, w, h):
        x1, x2, x3, x4 = (x[0], x[1], x[2], x[3])
        x1_next = x1 + h * x2
        x2_next = x2 + h * self.B * (x1 * x4 ** 2 - self.G * np.sin(x3))
        x3_next = x3 + h * x4
        x4_next = x4 + h * (u[0] - 2 * self.M * x1 * x2 * x4 - self.M * self.G * x1 * np.cos(x3)) / (
                self.M * x1 ** 2 + self.J + self.Jb)

        return np.array([x1_next, x2_next, x3_next, x4_next] + w, dtype=np.float32)

    def output_func(self, x):
        return np.array([x[0]], dtype=np.float32)

    def beam_edges_constraint(self, x: Array) -> Array:
        return np.array([x[0] + 4,   # left edge
                         4 - x[0]])  # right edge

    def beam_angle_constraint(self, x: Array) -> Array:
        return np.array([x[2] + np.pi/2,   # min angle
                         np.pi/2 - x[2]])  # max angle


class BallBeamEKF(ExtendedKalmanFilterObserver):
    def __init__(self, settings):
        super().__init__(settings, BallBeamObserverModel())


class BallBeamMHE(MovingHorizonEstimator):
    def __init__(self, settings):
        super().__init__(settings, BallBeamObserverModel())


class LuenbergerObserver(pm.Observer):
    """
    Luenberger Observer that uses a simple euler forward integration
    """
    public_settings = OrderedDict([
        ("initial state", [0, 0, 0, 0]),
        ("poles", [-20, -20, -20, -20]),
        ("steady state", [0, 0, 0, 0]),
        ("steady tau", 0),
        ("tick divider", 1),
    ])

    def __init__(self, settings):
        settings.update(output_dim=4)
        super().__init__(settings)

        self.a_mat, self.b_mat, self.c_mat = linearise_system(
            self._settings["steady state"],
            self._settings["steady tau"])
        self.L = pm.place_siso(self.a_mat.T,
                               self.c_mat.T,
                               self._settings["poles"]).T
        self.output = np.array(self._settings["initial state"], dtype=float)

    def state_func(self, t, q, args):
        x_o = q
        u = args[0]
        y = args[1]

        dx_o = ((self.a_mat - self.L @ self.c_mat) @ x_o
                + self.b_mat @ u
                + self.L @ y)
        return dx_o

    def _observe(self, time, system_input, system_output):
        # np.copyto(self.output, self.next_output)
        if system_input is not None:
            dy = self.state_func(time, self.output,
                                 (system_input, system_output))
            self.output += self.step_width * dy
        return self.output


class LuenbergerObserverReduced(pm.Observer):
    """
    Reduced luenberger observer that uses a simple euler forward integration
    
    The observers ode solver has been reduced to EULER forward which enables 
    a better performance.
    The output is calculated in a reduced and transformed system.
    """
    public_settings = OrderedDict([
        ("initial state", [0, 0, 0, 0]),
        ("poles", [-3, -3, -3]),
        ("steady state", [0, 0, 0, 0]),
        ("steady tau", 0),
        ("tick divider", 1),
    ])

    def __init__(self, settings):
        settings.update(output_dim=4)
        super().__init__(settings)

        self.a_mat, self.b_mat, self.c_mat = linearise_system(
            self._settings["steady state"],
            self._settings["steady tau"])

        self.n = self.a_mat.shape[0]
        self.r = self.c_mat.shape[0]

        if self.c_mat.shape != (1, 4):
            raise Exception("LuenbergerObserverReduced only usable for SISO")

        # which cols and rows must be sorted
        self.switch = 0
        for i in range(self.c_mat.shape[1]):
            if self.c_mat[0, i] == 1:
                self.switch = i
                break

        # sort A, B, C for measured and unmeasured states
        self.a_mat = swap_cols(self.a_mat, 0, self.switch)
        self.a_mat = swap_rows(self.a_mat, 0, self.switch)
        self.C = swap_cols(self.c_mat, 0, self.switch)
        self.B = swap_rows(self.b_mat, 0, self.switch)

        # get reduced system
        self.a_mat_11 = self.a_mat[0:self.r, 0:self.r]
        self.a_mat_12 = self.a_mat[0:self.r, self.r:self.n]
        self.a_mat_21 = self.a_mat[self.r:self.n, 0:self.r]
        self.a_mat_22 = self.a_mat[self.r:self.n, self.r:self.n]

        self.b_1 = self.b_mat[0:self.r, :]
        self.b_2 = self.b_mat[self.r:self.n, :]

        self.L = pm.place_siso(self.a_mat_22.T,
                               self.a_mat_12.T,
                               self._settings["poles"]).T
        self.output = np.array(self._settings["initial state"], dtype=float)

    def _observe(self, time, system_input, system_output):
        if system_input is None:
            return self.output

        # transform ((un-)measured) and collect necessary from observer_output
        x_o = self.output
        x_o = swap_rows(x_o, 0, self.switch)
        x_o = x_o[self.r:self.n]

        u = system_input
        y = system_output

        # transform system and eliminate ydot
        x_o_t = x_o - self.L @ y
        dy = ((self.a_mat_22 - self.L @ self.a_mat_12) @ x_o_t
              + (self.a_mat_21 - self.L @ self.a_mat_11
                 + (self.a_mat_22 - self.L @ self.a_mat_12) @ self.L) @ y
              + (self.b_2 - self.L @ self.b_1) @ u
              )

        # EULER integration
        x_o_t += self.step_width * dy

        # transform system back to original observer coordinates
        x_o = x_o_t + self.L @ y

        out = np.hstack((y, x_o))

        # change state order back to the original order
        self.output = swap_rows(out, 0, self.switch)

        return self.output


class HighGainObserver(pm.Observer):
    """
    High Gain Observer for nonlinear systems
    """
    public_settings = OrderedDict([
        ("initial state", [0, 0, 0, 0]),
        ("poles", [-10, -10, -10, -10]),
        ("tick divider", 1),
    ])

    def __init__(self, settings):
        settings.update(output_dim=4)
        super().__init__(settings)

        params = sp.symbols('x1, x2, x3, x4, tau')
        x1, x2, x3, x4, tau = params
        x = [x1, x2, x3, x4]
        h = sp.Matrix([[x1]])
        f = sp.Matrix([[x2],
                       [st.B * x1 * x4 ** 2 - st.B * st.G * sin(x3)],
                       [x4],
                       [(tau - 2 * st.M * x1 * x2 * x4
                         - st.M * st.G * x1 * cos(x3)) / (
                                st.M * x1 ** 2 + st.J + st.Jb)]])

        q = sp.Matrix(pm.lie_derivatives(h, f, x, len(x) - 1))
        dq = q.jacobian(x)

        if dq.rank() != len(x):
            raise Exception("System might not be observable")

        # gets p = [p0, p1, ... pn-1]
        p = pm.char_coefficients(self._settings["poles"])

        k = p[::-1]
        l = dq.inv() @ k

        mat2array = [{'ImmutableMatrix': np.array}, 'numpy']
        self.h_func = sp.lambdify((x1, x2, x3, x4, tau), h, modules=mat2array)
        self.l_func = sp.lambdify((x1, x2, x3, x4, tau), l, modules=mat2array)
        self.f_func = sp.lambdify((x1, x2, x3, x4, tau), f, modules=mat2array)

        self.output = np.array(self._settings["initial state"], dtype=float)

    def _observe(self, time, system_input, system_output):
        if system_input is None:
            return self.output

        y = system_output
        u = system_input

        l_np = self.l_func(*self.output, u)
        f_np = self.f_func(*self.output, u)
        h_x_o = self.h_func(self.output[0], 0, 0, 0, 0)

        dx_o = f_np + l_np @ (y - h_x_o)

        # EULER integration
        self.output += np.squeeze(self.step_width * dx_o, axis=1)

        return self.output


class LuenbergerObserverInt(pm.Observer):
    """
    Luenberger Observer that uses solver to integrate
    """
    public_settings = OrderedDict([
        ("initial state", [0, 0, 0, 0]),
        ("poles", [-3, -3, -3, -3]),
        ("steady state", [0, 0, 0, 0]),
        ("steady tau", 0),
        ("step width", .001),
        ("tick divider", 1),
        ("Method", "adams"),
        ("step size", 1e-2),
        ("rTol", 1e-6),
        ("aTol", 1e-9),
        ("start time", 0),
        ("end time", 5)
    ])

    def __init__(self, settings):
        settings.update(output_dim=4)
        super().__init__(settings)

        self.a_mat, self.b_mat, self.c_mat = linearise_system(
            self._settings["steady state"],
            self._settings["steady tau"])
        self.L = pm.place_siso(self.a_mat.T,
                               self.c_mat.T,
                               self._settings["poles"]).T
        self.step_width = self._settings["tick divider"] * self._settings["step width"]
        self.output = np.array(self._settings["initial state"], dtype=float)

        self.solver = ode(self.linear_state_function)
        self.solver.set_integrator("dopri5",
                                   method=self._settings["Method"],
                                   rtol=self._settings["rTol"],
                                   atol=self._settings["aTol"])
        self.solver.set_initial_value(self._settings["initial state"])
        self.nextObserver_output = self.output

    def linear_state_function(self, t, q, args):
        x1_o, x2_o, x3_o, x4_o = q
        u = args[0]
        y = args[1]

        x_o = q
        dx_o = (self.a_mat - self.L @ self.c_mat) @ x_o + self.b_mat @ u + self.L @ y
        return dx_o

    def _observe(self, time, system_input, system_output):
        if system_input is not None:
            self.solver.set_f_params((system_input, system_output))
            self.output = self.solver.integrate(time)

        return self.output


pm.register_simulation_module(pm.Observer, LuenbergerObserver)
pm.register_simulation_module(pm.Observer, LuenbergerObserverInt)
pm.register_simulation_module(pm.Observer, LuenbergerObserverReduced)
pm.register_simulation_module(pm.Observer, HighGainObserver)
pm.register_simulation_module(pm.Observer, BallBeamEKF)
pm.register_simulation_module(pm.Observer, BallBeamMHE)
