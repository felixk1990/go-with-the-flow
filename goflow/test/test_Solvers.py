# @Author: Felix Kramer <felixk1990>
# @Date:   2022-06-29T13:21:34+02:00
# @Email:  felixuwekramer@proton.me
# @Filename: test_Solvers.py
# @Last modified by:   felixk1990
# @Last modified time: 2022-06-29T17:47:42+02:00


import numpy as np
import os
import os.path as op

import kirchhoff.circuit_flow as kfi
from hailhydro.flow_init import Flow
from goflow.adapter import init_ivp as gi
import goflow.models.binder as gfm
from test_FlowAdaptation import calc_optimisation, setUp, cleanUp
from aux import *

locPath = './goflow/test/tmp'

pars_model = dict(alpha_0=1., alpha_1=1.)
pars_src = dict(modesSRC='root_geometric')
pars_plx = dict(modePLX='default')
mode = 'murray'

def tmpDirectory(func):

    def modFunc():
        setUp()
        func()
        cleanUp()

    return modFunc

def eval_nsol(nsol, morpheus, tag):


    dataPoints = zip(nsol.t,nsol.y.transpose())
    model = morpheus.model
    pars = model.solver_options['args']
    cost = [model.calc_cost_stimuli(t, y, *pars)[0] for t, y in dataPoints]

    np.save(op.join(locPath, f'weights_{tag}'), nsol.y.transpose()[-1])
    np.save(op.join(locPath,f'cost_{tag}'), cost)
    #
    # g = morpheus.flow.circuit.G
    # saveGraphJson(g, op.join(locPath,f'graph_{args[0]}'))

    return nsol

@tmpDirectory
def test_AutoSolver():


    for kw in ['f1', 'f2']:
        #initialize circuit+flow pattern
        C = kfi.initialize_circuit_from_crystal('triagonal_planar',3).G
        # # initialize dynamic system and set integration parameters
        mrph = gi.morph_dynamic(C, mode, [pars_model, pars_src, pars_plx])
        mrph.evals = 10

        # numerically evaluate the system
        cnd = mrph.flow.circuit.edges['conductivity']
        cnd_scale = mrph.flow.circuit.scales['conductance']
        func_dict = dict(f1=mrph.autoSolve, f2=mrph.nlogSolve)
        sp = {
            't0': 1e-05,
            't1': .1,
            'x0': np.power(cnd/cnd_scale,0.25)*0.1,
        }
        nsol = func_dict[kw]((sp['t0'],sp['t1']), sp['x0'])
        eval_nsol(nsol, mrph, kw)

        cnd2 = calc_optimisation(kw)
        assert cnd2

@tmpDirectory
def test_CustomSolver():

    # #initialize circuit+flow pattern
    cfp={
        'type': 'hexagonal',
        'periods': 3,
    }
    pars_src = {
        'modeSRC': 'dipole_border',
    }
    pars_plx = {
        'modePLX':'default',
    }
    C = kfi.initialize_circuit_from_crystal(cfp['type'], cfp['periods'])
    F = Flow(C, pars_src, pars_plx)

    # set model and model parameters
    mp={
        'alpha_0': 1.,
        'alpha_1': 1.
    }
    model = gfm.modelBinder['murray'](pars=mp)

    # set solver options for custom integration (without event handling)
    so={
        'num_steps':500,
        'samples':100,
        'step':0.01,
    }
    model.solver_options.update(so)
    # initialize dynamic system and set integration parameters
    morpheus = gi.morph_dynamic(F, model,())
    k1 = morpheus.flow.circuit.edges['conductivity']
    k2 = morpheus.flow.circuit.scales['conductance']
    sp = {
         'x0': np.power(k1/k2,0.25)
    }

    nsol = morpheus.nsolve_custom(model.calc_update_stimuli,sp['x0'], **model.solver_options)

    #initialize circuit+flow pattern
    eval_nsol(nsol, morpheus, '')
    cnd2 = calc_optimisation('')
    assert cnd2
