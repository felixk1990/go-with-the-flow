# @Author: Felix Kramer <felixk1990>
# @Date:   2022-06-29T13:16:22+02:00
# @Email:  felixuwekramer@proton.me
# @Filename: test_FluxAdaptation.py
@Last modified by:   felix
@Last modified time: 2022-07-01T16:06:05+02:00


import numpy as np
import os
import os.path as op

import kirchhoff.circuit_flow as kfc
from goflow.adapter import init_ivp as gi
from test_FlowAdaptation import setUp, cleanUp, calc_nsol, calc_graphState, calc_optimisation

from aux import *

locPath = './goflow/test/tmp'
pars_src = dict(modesSRC='dipole_border')
pars_plx = dict(modePLX='default')
modes = ['meigel', 'link']


def initEval(*args):

    mode = args[0]
    pars = args[1:]
    #initialize circuit+flow pattern
    C = kfc.initialize_circuit_from_crystal('square',3)
    # # initialize dynamic system and set integration parameters
    morpheus = gi.morph_dynamic(C, mode, pars)
    morpheus.evals = 10

    # numerically evaluate the system
    cnd = morpheus.flow.circuit.edges['conductivity']
    cnd_scale = morpheus.flow.circuit.scales['conductance']

    sp = {
        't0': 1e-05,
        't1': 1.,
        'x0': np.power(cnd/cnd_scale,0.25)*0.1,
    }
    nsol = morpheus.nlogSolve((sp['t0'],sp['t1']), sp['x0'])

    return nsol, morpheus

def test_nsol_Cyclic():

    setUp()

    pars_models = []

    # meigel
    pars_models.append([dict(phi0=0.5), pars_src, pars_plx])
    # link
    pars_models.append([dict(phi0=0.5, alpha_0=.1, alpha_1=.1), pars_src, pars_plx])

    for i in range(2):

        # set model and model parameters
        args = [modes[i], *pars_models[i]]
        nsol = calc_nsol(*args)
        assert nsol.success

def test_optimisation_Cyclic():

    for i in range(2):
        cnd2 = calc_optimisation(modes[i])
        assert cnd2
