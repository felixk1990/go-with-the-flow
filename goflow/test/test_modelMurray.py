import numpy as np
import os.path as op

import kirchhoff.circuit_flow as kfc
from hailhydro.flow_init import Flow

from goflow.adapter import init_ivp as gi
from goflow.models.murray import murray

locPath = './goflow/test/tmp'
from aux import *


def initEval():

    #initialize circuit+flow pattern
    # Circuit = kfc.initialize_flow_circuit_from_crystal('hexagonal',3)
    # pars_src = dict(mode='default')
    # pars_plx = dict(mode='default')
    # flowArgs= [Circuit, pars_src, pars_plx]
    # F = Flow(*flowArgs)
    #
    # # set model and model parameters
    # pars_model={
    #     'alpha_0':1.,
    #     'alpha_1':1.
    # }
    # murrayModel = murray(pars_model)
    #
    # # # initialize dynamic system and set integration parameters
    # morpheus = gi.morph_dynamic(flow=F, model=murrayModel)
    #
    # cnd = Circuit.edges['conductivity']
    # cnd_scale = Circuit.scales['conductance']
    # sp={
    #     't0': 0.,
    #     't1': 25.,
    #     'x0': np.power(cnd/cnd_scale,0.25),
    # }
    #
    # # numerically evaluate the system
    # update = murrayModel.calc_update_stimuli
    # nsol = morpheus.nsolve(update, (sp['t0'],sp['t1']), sp['x0'], **murrayModel.solver_options)

    #initialize circuit+flow pattern
    # C = kfi.initialize_circuit_from_crystal('laves',3)
    C = kfc.initialize_circuit_from_crystal('triagonal_planar',5).G
    pars_src = {
        'modesSRC': 'root_geometric'
    }
    pars_plx = {
        'modePLX':'default',
    }
    # set model and model parameters
    pars_model = {
        'alpha_0':1.,
        'alpha_1':1.
    }
    # # initialize dynamic system and set integration parameters
    morpheus = gi.morph_dynamic(C, 'murray', [pars_model, pars_src, pars_plx])
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

def test_nsol():

    nsol, morpheus = initEval()

    dataPoints = zip(nsol.t,nsol.y.transpose())
    murrayModel = morpheus.model
    args = murrayModel.solver_options['args']
    cost = [murrayModel.calc_cost_stimuli(t, y, *args)[0] for t, y in dataPoints]

    np.save(op.join(locPath, 'weights'), nsol.y.transpose()[-1])
    np.save(op.join(locPath,'cost'), cost)

    g = morpheus.flow.circuit.G
    saveGraphJson(g, op.join(locPath,'graph'))

    assert nsol.success

def test_graphState():

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

    assert (cnd1 and cnd3)

def test_optimisation():

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

    assert cnd2
