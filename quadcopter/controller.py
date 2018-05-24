from collections import OrderedDict

import pymoskito as pm
import numpy as np


class Dummy(pm.Controller):
    public_settings = OrderedDict([])

    def __init__(self, settings):
        settings.update(input_order=0)
        settings.update(input_type='system_output')
        super().__init__(settings)

    def _control(self, time, trajectory_values=None, feedforward_values=None, input_values=None, **kwargs):
        return np.zeros(1)


pm.register_simulation_module(pm.Controller, Dummy)