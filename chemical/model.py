from collections import OrderedDict

import numpy as np
import pymoskito as pm
import sympy as sp

from extended_kalman_filter import ObserverModel, Array


class ChemicalModel(pm.Model):
    public_settings = OrderedDict([('initial state', [3.0, 1.0]),
                                   ('v', [-2.0, 1.0]),
                                   ('k', 0.16),
                                   ('output cov', 0.01)])

    def __init__(self, settings):
        settings.update(state_count=2)
        settings.update(input_count=1)  # in reality 0
        super().__init__(settings)

        self.v = np.array(self._settings['v'])
        self.k = self._settings['k']
        self.output_cov = self._settings['output cov']

    def state_function(self, t, x, args):
        dx = self.v * self.k * x[0]**2
        return dx

    def calc_output(self, x):
        return np.random.normal(x[0]+x[1], np.sqrt(self.output_cov))


class ChemicalObserverModel(ObserverModel):
    def __init__(self):
        state_dim = 2
        input_dim = 0
        output_dim = 1
        self.state_lambda, self.out_lambda, f_jacobian, h_jacobian = self.derive_lambdas()
        state_eq_constraints = []
        state_ineq_constraints = [(lambda x: x,
                                   2,
                                   lambda x: np.eye(2))]

        super().__init__(state_dim, input_dim, output_dim, f_jacobian, h_jacobian, state_ineq_constraints,
                         state_eq_constraints)

    def discrete_state_func(self, x: Array, u: Array, w: Array, h: float) -> Array:
        return self.state_lambda(x, u, h).ravel() + w

    def output_func(self, x: Array) -> Array:
        return self.out_lambda(x).ravel()

    def derive_lambdas(self):
        pA, pB, h = sp.symbols('p_A p_B h')
        k = 0.16
        x = sp.Matrix([pA, pB])

        f = sp.Matrix([pA/(2*k*h*pA+1), pB+(k*h*pA**2)/(2*k*h*pA+1)])
        f_scalar = sp.lambdify((pA, pB, h), f)
        f_lambda = lambda x_, u_, h_: f_scalar(*x_, h_)

        f_jac_scalar = sp.lambdify((pA, pB, h), f.jacobian(x))
        f_jac = lambda x_, u_, h_: f_jac_scalar(*x_, h_)

        h_fun = sp.Matrix([pA + pB])
        h_scalar = sp.lambdify((pA, pB), h_fun)
        h_lambda = lambda x_: h_scalar(*x_)

        h_jac_scalar = sp.lambdify((pA, pB), h_fun.jacobian(x))
        h_jac = lambda x_: h_jac_scalar(*x_)

        return f_lambda, h_lambda, f_jac, h_jac


pm.register_simulation_module(pm.Model, ChemicalModel)
