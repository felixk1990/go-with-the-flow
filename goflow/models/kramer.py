# @Author: Felix Kramer <kramer>
# @Date:   23-06-2021
# @Email:  kramer@mpi-cbg.de
# @Project: phd_network_remodelling
# @Last modified by:   felix
# @Last modified time: 2022-07-02T14:16:15+02:00

import numpy as np
import copy
from dataclasses import dataclass, field
# custom
from hailhydro.flow_random import FlowRandom as FlowRandom
from .base import model


def dualFlowRandom(dualCircuit, *args):

    dualFlow = []
    for i, cs in enumerate(dualCircuit.layer):

        new_args = copy.deepcopy(args)
        for arg in new_args:
            for k, v in arg.items():
                arg[k] = v[i]

        dualFlow.append(FlowRandom(cs, *new_args))
        dualFlow[-1].e_adj = dualCircuit.e_adj[:]
        dualFlow[-1].dist_adj = dualCircuit.dist_adj[:]

    return dualFlow


@dataclass
class kramer(model):
    """
    The network adaptation model for for two entangled networks, according
    to Kramer et al, based on minimization of noisy disspation-volume in
    combination with intertwinedness retrictions/couplings:
        Kramer and Modes, How to pare a pair: Topology control and pruning in
        intertwined complex networks, PRR, 2020

    The system's cost function is given as:\n
    .. math::
        \\Gamma = \\sum_{e} \\left( C_e \\langle \\Delta p_e^2 \\rangle
        + a r_e^{2} \\right) + b \\sum_{ee'} \\Delta r_{ee'}^{\\varepsilon}
    Attributes:
        pars (dict):\n
         The specific model parameters p0 (growth rate), p1 (coupling), p2
         (volume penalty), p3 (fluctuation) and coupling exponent \\varepsilon.

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

        self.solver_options.update({'events': 'dynamic'})
        if self.pars:
            self.set_model_parameters(self.pars)

    def update_event_func(self):

        self.solver_options['events'] = self.events[
                                            self.solver_options['events']
                                            ]

    def set_solver_options(self, solv_opt):

        for k, v in solv_opt.items():

            self.solver_options[k] = v

        self.update_event_func()

    def calc_update_stimuli(self, t, x_0, flow, p_0, p_1, p_2, p_3, coupling):

        # pruning
        sgl = np.where(x_0 <= 0.)[0]
        x_0[sgl] = np.power(10., -10)
        # x_0[sgl]

        idxSets = [len(f.circuit.edges['label']) for f in flow]
        sgn = coupling / np.absolute(coupling)

        dx_pre = []
        x_sep = [x_0[:idxSets[0]], x_0[idxSets[0]:]]

        for i, idx in enumerate(idxSets):

            f = flow[i]
            x = x_sep[i]
            f.set_effective_source_matrix(p_3[i])

            # calc flows
            x_sq, p_sq = self.get_stimuli_pars(f, x)
            x_cb = np.multiply(x_sq, x)

            # calc interaction
            cpl = np.zeros(idx)
            for j, e in enumerate(f.e_adj):

                dr = 1. - (x_sep[0][e[0]] + x_sep[1][e[1]])
                force = sgn * (dr**coupling)
                cpl[e[i]] += p_1[i] * force

            # calc total feedback
            shear_sq = np.multiply(p_sq, x_cb)
            vol = p_2[i] * x
            diff_shearvol = np.subtract(shear_sq, vol)

            dx_pre.append(p_0[i]*np.add(diff_shearvol, cpl))

        dx = np.concatenate((dx_pre[0], dx_pre[1]))

        # pruning
        dx[sgl] = 0.

        return dx

    def get_stimuli_pars(self, flow, x_0):

        k = flow.circuit.scales['conductance']

        x_sq = np.power(x_0, 2)
        conductivity = flow.calc_conductivity_from_cross_section(x_sq, k)
        p_sq, q_sq = flow.calc_sq_flow_effective(conductivity)

        return x_sq, p_sq

    def prune(self, t, x_0, flow, p_0, p_1, p_2, p_3, coupling):
        """
        Check whether a vessel collapsed, i.e. negative radii appear during
        integration. If so, handle it by pruning the vessel.

        Args:
            t (float):\n
             Current time step in numeric ODE evaluation \n
            x_0 (array):\n
             Current state vector of the DS. \n
            args (iterable):\n
             Model specific tuple of parameters, needed to evaluate stimulus ]
             functions \n

        Returns:
            int:
                1: do not prunr and keep updating
                0: vessel collapsed, stop updating

        """
        f = 1
        if np.any(x_0 < 0):
            f = 0

        return f
