# @Author: Felix Kramer <felixk1990>
# @Date:   2022-06-28T16:24:41+02:00
# @Email:  felixuwekramer@proton.me
# @Filename: aux.py
# @Last modified by:   felixk1990
# @Last modified time: 2022-06-28T23:50:29+02:00


import networkx.readwrite.json_graph as nj
import json
import numpy as np
import networkx as nx

def loadGraphJson(pathInput):

    with open(pathInput+'.json',) as file:
        data = json.load(file)

    G = nj.node_link_graph(data)

    return G

def saveGraphJson(nxGraph , pathOutput):

    # convert to list types
    for component in [nxGraph.edges(), nxGraph.nodes()]:
        for u in component:
            for k, v in component[u].items():
                if isinstance(v, np.ndarray):
                    component[u][k] = v.tolist()

    data = nj.node_link_data(nxGraph)
    with open(pathOutput+'.json', 'w+') as file:
        json.dump(data, file)

def clipp_graph(circuit):

    """
    Prune the internal graph and generate a new internal variable
    represting the pruned based on an interanl threshold value.

    """

    #cut out edges which lie beneath a certain threshold value and export
     # this clipped structure
    circuit.set_network_attributes()
    circuit.threshold = 0.01
    list_graph_edges = list(circuit.G.edges())
    for e in list_graph_edges:
        if circuit.G.edges[e]['conductivity'] > circuit.threshold:
            circuit.H.add_edge(*e)
            for k in circuit.G.edges[e].keys():
                circuit.H.edges[e][k] = circuit.G.edges[e][k]

    list_pruned_nodes = list(circuit.H.nodes())
    list_pruned_edges = list(circuit.H.edges())

    for n in list_pruned_nodes:
        for k in circuit.G.nodes[n].keys():
            circuit.H.nodes[n][k] = circuit.G.nodes[n][k]
        circuit.H_J.append(circuit.G.nodes[n]['source'])
    for e in list_pruned_edges:
        circuit.H_C.append(circuit.H.edges[e]['conductivity'])

    circuit.H_C = np.asarray(circuit.H_C)
    circuit.H_J = np.asarray(circuit.H_J)

    return circuit

def get_nullity(H):

    E = nx.number_of_edges(H)
    N = nx.number_of_nodes(H)
    CC = nx.number_connected_components(H)
    nullity = E-N+CC

    return nullity, CC
