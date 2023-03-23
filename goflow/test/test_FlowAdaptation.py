# @Author: Felix Kramer <felixk1990>
# @Date:   2022-06-28T17:04:33+02:00
# @Email:  felixuwekramer@proton.me
# @Filename: test_modelMurray.py
# @Last modified by:   kramer
# @Last modified time: 23-03-2023


import numpy as np
import os
import os.path as op

import kirchhoff.circuit_flow as kfc
# import kirchhoff.io_networkx as inx
from hailhydro.flow_init import Flow

from goflow.adapter import init_ivp as gi
from goflow.models.murray import murray
from helpFunc import *

locPath = './goflow/test/tmp'
pars_src = dict(modesSRC='root_geometric')
pars_plx = dict(modePLX='default')
modes = ['murray', 'bohn','corson']

def initEval(*args):

    mode = args[0]
    pars = args[1:]
    #initialize circuit+flow pattern
    # C = kfi.initialize_circuit_from_crystal('laves',3)
    C = kfc.initialize_flow_circuit_from_crystal('triagonal_planar',5).G
    # # initialize dynamic system and set integration parameters
    morpheus = gi.morph_dynamic(C, mode, pars)
    morpheus.evals = 10

    # numerically evaluate the system
    cnd = morpheus.flow.circuit.edges['conductivity']
    cnd_scale = morpheus.flow.circuit.scales['conductance']

    sp = {
        't0': 1e-05,
        't1': 4.,
        'x0': np.power(cnd/cnd_scale,0.25)*0.1,
    }
    nsol = morpheus.nlogSolve((sp['t0'],sp['t1']), sp['x0'])

    return nsol, morpheus

def calc_nsol(*args):

    nsol, morpheus = initEval(*args)

    dataPoints = zip(nsol.t,nsol.y.transpose())
    model = morpheus.model
    pars = model.solver_options['args']
    cost = [model.calc_cost_stimuli(t, y, *pars)[0] for t, y in dataPoints]

    np.save(op.join(locPath, f'weights_{args[0]}'), nsol.y.transpose()[-1])
    np.save(op.join(locPath,f'cost_{args[0]}'), cost)

    g = morpheus.flow.circuit.G
    saveGraphJson(g, op.join(locPath,f'graph_{args[0]}'))

    return nsol

def calc_graphState(mode, IsTree):

    conductivity = np.load(op.join(locPath, f'weights_{mode}.npy'))
    G = loadGraphJson(op.join(locPath, f'graph_{mode}'))

    circuit = kfc.FlowCircuit(G)
    circuit.edges['conductivity'] = conductivity
    circuit = clipp_graph(circuit)
    H = circuit.H
    nullity, CC = get_nullity(H)
    print(nullity)

    cnd1 = False
    if IsTree:
        if nullity == 0:
            cnd1 = True
    else:
        if nullity > 0:
            cnd1 = True

    print(f'Nullity test passed: {cnd1}')

    cnd3 = False
    if CC == 1:
        cnd3 = True
    print(f'Connectivity test passed: {cnd3}')

    return cnd1, cnd3

def calc_optimisation(mode):

    cost = np.load(op.join(locPath, f'cost_{mode}.npy'))
    cnd_set = []
    # print(cost[1:])
    for i, c in enumerate(cost[1:]):

        rslt = False
        if np.round(c,3) <= np.round(cost[i],3):
            rslt = True
        cnd_set.append(rslt)

    # print(cnd_set)
    cnd2 = False
    if all(cnd_set):
        cnd2 = True
    print(f'Optimisation test passed: {cnd2}')

    return cnd2

def setUp():

    os.system(f'mkdir {locPath}')

def cleanUp():

    os.system(f'rm -r {locPath}')

def test_nsol_Tree():

    setUp()
    pars_models = []
    # murray
    pars_models.append([dict(alpha_0=1., alpha_1=1.), pars_src, pars_plx])
    # bohn
    pars_models.append([dict(alpha_0=1., alpha_1=1., gamma=.5), pars_src, pars_plx])
    # corson
    pars_models.append([dict(alpha_0=1., alpha_1=1., gamma=.5, noise=.1), pars_src, pars_plx, dict(mode='default', noise=1.)])
    for i in range(3):

        # set model and model parameters


        args = [modes[i], *pars_models[i]]
        nsol = calc_nsol(*args)
        assert nsol.success

def test_optimisation_Tree():

    for i in range(3):
        cnd2 = calc_optimisation(modes[i])
        assert cnd2

def test_graphState_Tree():

    IsTree = True
    for i in range(3):
        cnd1, cnd3 = calc_graphState(modes[i],IsTree)
        assert (cnd1 and cnd3)

    cleanUp()

def test_nsol_Cyclic():

    setUp()
    modes = ['bohn','corson']
    pars_models = []

    # bohn
    pars_models.append([dict(alpha_0=1., alpha_1=1., gamma=1.5), pars_src, pars_plx])
    # corson
    pars_models.append([dict(alpha_0=1., alpha_1=1., gamma=.5, noise=10.), pars_src, pars_plx, dict(mode='default', noise=1.)])
    for i in range(2):

        # set model and model parameters
        args = [modes[i], *pars_models[i]]
        nsol = calc_nsol(*args)
        assert nsol.success

def test_optimisation_Cyclic():

    modes = ['bohn','corson']

    for i in range(2):
        cnd2 = calc_optimisation(modes[i])
        assert cnd2

def test_graphState_Cyclic():

    IsTree = False
    modes = ['bohn','corson']
    for i in range(2):
        cnd1, cnd3 = calc_graphState(modes[i],IsTree)
        assert (cnd1 and cnd3)

    cleanUp()
