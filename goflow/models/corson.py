# @Author: Felix Kramer <kramer>
# @Date:   23-06-2021
# @Email:  kramer@mpi-cbg.de
# @Project: phd_network_remodelling
# @Last modified by:   felix
# @Last modified time: 2022-07-02T13:45:16+02:00

import numpy as np
from dataclasses import dataclass

# custom
# from ..adapter.init_ivp import *
from .murray import murray


@dataclass
class corson(murray):
    """
    The network adaptation model according to Bohn et al, based on flow
    dissipation-volume minimization with reweightedscales of conductivity costs
    as published for example:
        Corson, Fluctuations and Redundancy in Optimal Transport Networks, PRL,
        2010\n
        Hu and Cai, Adaptation and Optimization of Biological Transport
        Networks, PRL, 2013
    The approach assumes a varying sink-source landscape and a time scale
    separation between this variation and radial adaptation.
    The system's cost function is given as:\n
    .. math::
        \\Gamma = \\sum_{e}\\left(C_e\\langle \\Delta p_e^2\\rangle
        + aC_e^{\\gamma}\\right)
    The system's derived gradient descent
    dynamics are given as:\n
    .. math::
        \\frac{dr_e}{dt} \\propto \\left[\\alpha_1
        \\frac{\\langle \\Delta p_e^2 \\rangle }{L_e^2}
        \\frac{C_e}{r_e^{4\\gamma}}-\\alpha_0\\right] r_e
    Attributes:
        pars (dict):\n
         The specific model parameters \\alpha_0, \\alpha_1, \\gamma and
         effective noise strength \\eta. \n

        ivp_options (dict):\n
         Information to generate the internal solver_options. Providing t0,t1
         number of evaluation and x0.\n
        model_args (list):\n
         Model specific paramerets need to evaluate the update rules of the DS
         \n
        solver_options (dict):\n
         Specifying runtime and evaluation marks. \n
        events (dict):\n
         Events to consier, here in gernal flatlining events which allow for
         early termination of stiff simulations. \n
        null_decimals (in):\n
         description \n

    Methods:
        init()
            The model post_init function.
        update_event_func()
            Update the event function and ensures that solver_options are set.
        flatlining_default(t, x_0, *args)
            The default flatlining function for determining the terminal event
            of the dynamical system.
        flatlining_dynamic(t, x_0, *args)
            The dynamic flatlining function for determining the terminal event
            of the dynamical system.
        set_model_parameters(model_pars)
            Set internal model arguments array.
        set_solver_options(solv_opt)
            Set internal solver_options and update event function.
        calc_update_stimuli(t, x_0, *args)
            The dynamic system's temporal update rule, computing the gradient
            -dF for dx/dt.
        calc_cost_stimuli(t, x_0, *args)
            Computes the dynamic system's Lyapunov function and its gradient.
        get_stimuli_pars(self, flow, x_0)
            Update flow & pressure landscapes, recompute conductivity as well
            as squared quantities.

    """

    def __post_init__(self):

        self.init()

        self.model_args = [1., 1., 1, 0.]
        self.solver_options.update({'events': 'dynamic'})
        if self.pars:
            self.set_model_parameters(self.pars)

    def set_model_parameters(self, model_pars):

        for k, v in model_pars.items():

            if 'alpha_0' == k:
                self.model_args[0] = v

            if 'alpha_1' == k:
                self.model_args[1] = v

            if 'gamma' == k:
                self.model_args[2] = v

            if 'noise' == k:
                self.model_args[3] = v

    def calc_update_stimuli(self, t, x_0, flow, a_0, a_1, gm, noise):

        flow.set_effective_source_matrix(noise)
        cnd, p_sq = self.get_stimuli_pars(flow, x_0)
        x_gamma = np.power(x_0, 4*gm)

        s1 = a_1*np.divide(np.multiply(p_sq, cnd), x_gamma)
        s2 = a_0*np.ones(len(x_0))*gm
        ds = np.subtract(s1, s2)
        dx = 4*np.multiply(ds, x_0)

        return dx

    def calc_cost_stimuli(self, t, x_0, flow, alpha_0, alpha_1, gamma, noise):

        flow.set_effective_source_matrix(noise)
        x_sq = np.power(x_0, 2)
        conductivity, p_sq = self.get_stimuli_pars(flow, x_0)

        f1 = alpha_1*np.multiply(p_sq, np.power(x_sq, 2))
        f2 = alpha_0*np.power(x_sq, 2*gamma)
        F = np.sum(np.add(f1, f2))

        dF = -self.calc_update_stimuli(
            t, x_0, flow,
            alpha_0, alpha_1, gamma, noise
            )

        return F, dF

    def get_stimuli_pars(self, flow, x_0):

        k = flow.circuit.scales['conductance']

        c = flow.calc_conductivity_from_cross_section(np.power(x_0, 2), k)
        p_sq, q_sq = flow.calc_sq_flow_effective(c)

        return c, p_sq
