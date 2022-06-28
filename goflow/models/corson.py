# @Author: Felix Kramer <kramer>
# @Date:   23-06-2021
# @Email:  kramer@mpi-cbg.de
# @Project: phd_network_remodelling
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-09-08T21:08:26+02:00

import numpy as np
from dataclasses import dataclass, field

# custom
# from ..adapter.init_ivp import *
from .murray import murray

@dataclass
class corson(murray):

        def __post_init__(self):

            self.init()

            self.model_args = [1., 1., 1, 0.]
            self.solver_options.update({ 'events': 'dynamic'})
            if self.pars:
                self.set_model_parameters(self.pars)

        def set_model_parameters(self, model_pars):

            for k, v in model_pars.items():

                if 'alpha_0' == k :
                    self.model_args[0] = v

                if 'alpha_1' == k :
                    self.model_args[1] = v

                if 'gamma' == k :
                    self.model_args[2] = v

                if 'noise' == k :
                    self.model_args[3] = v

        def calc_update_stimuli(self, t, x_0, flow, a_0, a_1, gm, noise):

            flow.set_effective_source_matrix(noise)
            cnd, p_sq = self.get_stimuli_pars(flow, x_0)
            x_gamma = np.power(x_0, 4*gm)

            s1 = a_1*np.divide(np.multiply( p_sq, cnd), x_gamma)
            s2 = a_0*np.ones(len(x_0))*gm
            ds = np.subtract(s1, s2)
            dx = 4*np.multiply(ds, x_0)

            return dx

        def calc_cost_stimuli(self, t, x_0, flow, alpha_0, alpha_1, gamma, noise):

            flow.set_effective_source_matrix(noise)
            x_sq = np.power(x_0, 2)
            conductivity, p_sq = self.get_stimuli_pars(flow, x_0)

            f1 = alpha_1*np.multiply(p_sq, np.power(x_sq, 2))
            f2 = alpha_0 *np.power(x_sq, 2*gamma)
            F = np.sum(np.add(f1, f2))

            dF = -self.calc_update_stimuli(t, x_0, flow, alpha_0, alpha_1, gamma, noise)

            return F, dF

        def get_stimuli_pars(self, flow, x_0):

            k = flow.circuit.scales['conductance']

            c = flow.calc_conductivity_from_cross_section( np.power(x_0, 2), k)
            p_sq, q_sq = flow.calc_sq_flow_effective(c)

            return c, p_sq
