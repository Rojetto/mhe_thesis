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


pm.register_simulation_module(pm.Model, ChemicalModel)
