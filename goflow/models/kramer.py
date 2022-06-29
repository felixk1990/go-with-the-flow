# @Author: Felix Kramer <kramer>
# @Date:   23-06-2021
# @Email:  kramer@mpi-cbg.de
# @Project: phd_network_remodelling
# @Last modified by:   felixk1990
# @Last modified time: 2022-06-29T17:54:15+02:00

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

    pars: dict = field(default_factory=dict, init=True, repr=True)

    def __post_init__(self):

        self.init()

        self.solver_options.update({'events': 'dynamic'})
        if self.pars:
            self.set_model_parameters(self.pars)

    def update_event_func(self):

    # try:
        # default = self.events[self.solver_options['events']]
        # self.solver_options['events'] = [ default, self.prune]
        self.solver_options['events'] = self.events[self.solver_options['events']]

    def set_solver_options(self, solv_opt):

        for k, v in solv_opt.items():

            self.solver_options[k] = v

        self.update_event_func()

    def calc_update_stimuli(self, t, x_0, flow, p_0, p_1, p_2, p_3, coupling):

        '''
            no implicit solvers! otherwise, pruning exceptions not handeled properly
            p0:timescale
            p1: coupling
            p2: volume penalty
            p3: fluctuation
        '''

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

            dx_pre.append(p_0[i] *np.add(diff_shearvol, cpl))

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

        f = 1
        if np.any(x_0 < 0):
            f = 0

        return f
