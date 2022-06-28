import numpy as np
import os
import os.path as op

import kirchhoff.circuit_flow as kfc
from hailhydro.flow_init import Flow

from goflow.adapter import init_ivp as gi
from goflow.models.murray import murray

locPath = './goflow/test/tmp'
from aux import *


def initEval(mode, pars_src, pars_plx, pars_model):

    #initialize circuit+flow pattern
    # C = kfi.initialize_circuit_from_crystal('laves',3)
    C = kfc.initialize_circuit_from_crystal('triagonal_planar',5).G
    # # initialize dynamic system and set integration parameters
    morpheus = gi.morph_dynamic(C, mode, [pars_model, pars_src, pars_plx])
    morpheus.evals = 200

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

def get_nullity(H):

    E = nx.number_of_edges(H)
    N = nx.number_of_nodes(H)
    CC = nx.number_connected_components(H)
    nullity = E-N+CC

    return nullity, CC

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

def calc_graphState():

    conductivity = np.load(op.join(locPath, 'weights.npy'))
    G = loadGraphJson(op.join(locPath, 'graph'))

    circuit = kfc.FlowCircuit(G)
    circuit.edges['conductivity'] = conductivity
    circuit = clipp_graph(circuit)
    H = circuit.H
    nullity, CC = get_nullity(H)
    print(nullity)

    cnd1 = False
    if nullity == 0:
        cnd1 = True

    print(f'Nullity test passed: {cnd1}')

    cnd3 = False
    if CC == 1:
        cnd3 = True
    print(f'Connectivity test passed: {cnd3}')

    return cnd1, cnd3

def calc_optimisation():

    cost = np.load(op.join(locPath, 'cost.npy'))
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

def test_setUp():

    os.system(f'mkdir {locPath}')

def test_nsol():

    pars_src = dict(modesSRC='root_geometric')
    pars_plx = dict(modePLX='default')
    mode = ['murray', 'bohn']
    for i in range(2):

        # set model and model parameters
        pars_model = dict(
            alpha_0=1.,
            alpha_1=1.
        )

        args = [mode[i], pars_src, pars_plx, pars_model]
        nsol = calc_nsol(*args)
        assert nsol.success

def test_cleanUp():

    os.system(f'rm -r {locPath}')
# def test_optimisation():
#
#     cnd2 = calc_optimisation()
#     assert cnd2
#
# def test_graphState():
#
#     cnd1, cnd3 = calc_graphState()
#     assert (cnd1 and cnd3)
#
# def test_models_flow():
#
#
#     pars_src = dict(modesSRC='root_geometric')
#     pars_plx = dict(modePLX='default')
#
#     # set model and model parameters
#     pars_model = dict(
#         alpha_0=1.,
#         alpha_1=1.
#     )
#
#     args = ['murray', pars_src, pars_plx, pars_model]
#     nsol, morpheus = initEval(*args)
#     assert(True)
