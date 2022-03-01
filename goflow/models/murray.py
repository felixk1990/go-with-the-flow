# @Author: Felix Kramer <kramer>
# @Date:   23-06-2021
# @Email:  kramer@mpi-cbg.de
# @Project: phd_network_remodelling
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-09-08T21:08:26+02:00

import numpy as np
from dataclasses import dataclass, field
# custom
from .base import model


@dataclass
class murray(model):

    pars: dict = field(default_factory=dict, init=True, repr=True)

    def __post_init__(self):

        self.init()

        self.model_args = [1., 1.]
        self.solver_options.update({'events': 'default'})
        if self.pars:
            self.set_model_parameters(self.pars)

    def set_model_parameters(self, model_pars):

        for k, v in model_pars.items():

            if 'alpha_0' == k :
                self.model_args[0] = v

            if 'alpha_1' == k :
                self.model_args[1] = v

    def set_solver_options(self, solv_opt):

        for k, v in solv_opt.items():

            self.solver_options[k] = v

        self.update_event_func()

    def calc_update_stimuli(self, t, x_0, flow, alpha_0, alpha_1):

        x_sq, p_sq = self.get_stimuli_pars(flow, x_0)

        s1 = 2.*alpha_1*np.multiply( p_sq, x_sq)
        s2 = alpha_0*np.ones(len(x_0))
        ds = np.subtract(s1, s2)
        dx = 2*np.multiply(ds, x_0)

        return dx

    def calc_cost_stimuli(self, t, x_0, flow, alpha_0, alpha_1):

        x_sq, p_sq = self.get_stimuli_pars(flow, x_0)

        f1 = alpha_1*np.multiply(p_sq, np.power(x_sq, 2))
        f2 = alpha_0 *x_sq
        F = np.sum(np.add(f1, f2))

        dF = -self.calc_update_stimuli(t, x_0, flow, alpha_0, alpha_1)

        return F, dF

    def get_stimuli_pars(self, flow, x_0):

        k = flow.circuit.scales['conductance']
        src = flow.circuit.nodes['source']

        x_sq = np.power(x_0, 2)
        conductivity = flow.calc_conductivity_from_cross_section(x_sq, k)
        p_sq, q_sq =flow.calc_sq_flow(conductivity, src)

        return x_sq, p_sq
