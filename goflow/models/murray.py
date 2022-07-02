# @Author: Felix Kramer <kramer>
# @Date:   23-06-2021
# @Email:  kramer@mpi-cbg.de
# @Project: phd_network_remodelling
# @Last modified by:   felix
# @Last modified time: 2022-07-02T13:15:02+02:00

import numpy as np
from dataclasses import dataclass, field
# custom
from .base import model


@dataclass
class murray(model):
    """
    The network adaptation model according to Murray, based on flow
    dissipation-volume minimization as published in:
     Murray, The Physiological Principle of Minimum Work, PNAS, 1926

    The system's cost function is given as:\n
    .. math::
        \\Gamma = \\sum_{e}\\left(\\frac{f_e^2}{C_e} + ar_e^2\\right)
    For minimizing this cost one iteratively computes the flow f_e,
    pressure \\Delta p_e landscapes. The system's derived gradient descent
    dynamics are given as:\n
    .. math::
        \\frac{dr_e}{dt} \\propto \\left[\\alpha_1\\left(
        \\frac{\\Delta p_er_e}{L_e}\\right)^2-\\alpha_0\\right] r_e
    Attributes:
        pars (dict):\n
         The specific model parameters \\alpha_0, \\alpha_1. \n

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

    pars: dict = field(default_factory=dict, init=True, repr=True)

    def __post_init__(self):

        self.init()

        self.model_args = [1., 1.]
        self.solver_options.update({'events': 'default'})
        if self.pars:
            self.set_model_parameters(self.pars)

    def set_model_parameters(self, model_pars):

        for k, v in model_pars.items():

            if 'alpha_0' == k:
                self.model_args[0] = v

            if 'alpha_1' == k:
                self.model_args[1] = v

    def set_solver_options(self, solv_opt):

        for k, v in solv_opt.items():

            self.solver_options[k] = v

        self.update_event_func()

    def calc_update_stimuli(self, t, x_0, flow, alpha_0, alpha_1):

        x_sq, p_sq = self.get_stimuli_pars(flow, x_0)

        s1 = 2.*alpha_1*np.multiply(p_sq, x_sq)
        s2 = alpha_0*np.ones(len(x_0))
        ds = np.subtract(s1, s2)
        dx = 2*np.multiply(ds, x_0)

        return dx

    def calc_cost_stimuli(self, t, x_0, flow, alpha_0, alpha_1):

        x_sq, p_sq = self.get_stimuli_pars(flow, x_0)

        f1 = alpha_1*np.multiply(p_sq, np.power(x_sq, 2))
        f2 = alpha_0 * x_sq
        F = np.sum(np.add(f1, f2))

        dF = -self.calc_update_stimuli(t, x_0, flow, alpha_0, alpha_1)

        return F, dF

    def get_stimuli_pars(self, flow, x_0):
        """
        Update flow & pressure landscapes, recompute conductivity as well as
        squared quantities.

        Invokes the internal flow circuit and recalcs flows and pressures as
        well as conductivities according to Hagen-Poiseuille's law of flows in
        cylindrical pipes.

        Args:
            flow (flow):\n
                A hailhydro flow object, describing the current flow landscape
                in the transport network.\n
            x_0 (array):\n
                Current state vector of the dynmic system.\n

        Returns:
            iterable: \n
                The squared state vector as well as the squared pressure
                landscape x_sq, p_sq

        """

        k = flow.circuit.scales['conductance']
        src = flow.circuit.nodes['source']

        x_sq = np.power(x_0, 2)
        conductivity = flow.calc_conductivity_from_cross_section(x_sq, k)
        p_sq, q_sq = flow.calc_sq_flow(conductivity, src)

        return x_sq, p_sq
