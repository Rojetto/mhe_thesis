from collections import OrderedDict

import numpy as np
import pymoskito as pm
import sympy as sp

from extended_kalman_filter import ObserverModel, Array


class Dummy(pm.Model):
    public_settings = OrderedDict([('initial state', [1])])

    def __init__(self, settings):
        settings.update(state_count=1)
        settings.update(input_count=1)
        super().__init__(settings)

    def state_function(self, t, x, args):
        return np.zeros(1)

    def calc_output(self, input_vector):
        return np.zeros(1)


class QuadcopterObserverModel(ObserverModel):
    def __init__(self):
        state_dim = 7
        input_dim = 3
        output_dim = 4
        self.state_lambda, self.out_lambda, f_jacobian, h_jacobian = self.derive_lambdas()
        state_eq_constraints = [(lambda q: np.array([q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2 - 1]),
                                 1,
                                 lambda q: np.array([[2*q[0], 2*q[1], 2*q[2], 2*q[3], 0, 0, 0]]))]
        state_ineq_constraints = []

        super().__init__(state_dim, input_dim, output_dim, f_jacobian, h_jacobian, state_ineq_constraints,
                         state_eq_constraints)

    def discrete_state_func(self, x: Array, u: Array, w: Array, h: float) -> Array:
        return self.state_lambda(x, u, h).ravel() + w

    def output_func(self, x: Array) -> Array:
        return self.out_lambda(x).ravel()

    def derive_lambdas(self):
        a, b, c, d, bwx, bwy, bwz, wx, wy, wz, h = sp.symbols('a b c d bwx bwy bwz omega_x omega_y omega_z h')
        wrealx = wx-bwx
        wrealy = wy-bwy
        wrealz = wz-bwz
        g = 9.81
        w_mat = sp.Matrix([[0, -wrealx, -wrealy, -wrealz],
                           [wrealx, 0, wrealz, -wrealy],
                           [wrealy, -wrealz, 0, wrealx],
                           [wrealz, wrealy, -wrealx, 0]])
        q = sp.Matrix([[a, b, c, d]]).T
        x = sp.Matrix([[a, b, c, d, bwx, bwy, bwz]]).T
        dx = sp.Matrix([sp.S(1)/sp.S(2)*w_mat*q, sp.Matrix([0, 0, 0])])

        f = x + h * dx
        f_scalar = sp.lambdify((a, b, c, d, bwx, bwy, bwz, wx, wy, wz, h), f)
        f_lambda = lambda x_, u_, h_: f_scalar(*x_, *u_, h_)

        f_jac_scalar = sp.lambdify((a, b, c, d, bwx, bwy, bwz, wx, wy, wz, h), f.jacobian(x))
        f_jac = lambda x_, u_, h_: f_jac_scalar(*x_, *u_, h_)

        h_fun = sp.Matrix([[2*g*(b*d - a*c)],
                       [2*g*(a*b + c*d)],
                       [g*(a**2-b**2-c**2+d**2)],
                       [sp.atan2(2*(b*c+a*d), a**2+b**2-c**2-d**2)]])
        h_scalar = sp.lambdify((a, b, c, d, bwx, bwy, bwz), h_fun)
        h_lambda = lambda x_: h_scalar(*x_)

        h_jac_scalar = sp.lambdify((a, b, c, d, bwx, bwy, bwz), h_fun.jacobian(x))
        h_jac = lambda x_: h_jac_scalar(*x_)

        return f_lambda, h_lambda, f_jac, h_jac


pm.register_simulation_module(pm.Model, Dummy)
