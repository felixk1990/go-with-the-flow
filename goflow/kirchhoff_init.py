# @Author:  Felix Kramer <kramer>
# @Date:   2021-05-08T20:34:30+02:00
# @Email:  kramer@mpi-cbg.de
# @Project: go-with-the-flow
# @Last modified by:   kramer
# @Last modified time: 2021-05-09T11:56:30+02:00
# @License: MIT

import networkx as nx
import numpy as np
import sys

class circuit:

    def __init__(self):

        self.dict_scales={
            'conductance':1,
            'flow':1,
            'length':1
        }

        self.dict_graph={
            'graph_mode':'',
            'threshold':0.001,
            'num_sources':1
        }

        self.dict_nodes={
            'source':[],
            'potential':[],
        }
        self.dict_edges={
            'conductivity':[],
            'flow_rate':[],
        }

        self.set_graph_containers()

    def set_graph_containers(self):

        self.G=nx.DiGraph()
        self.H=nx.Graph()
        self.H_C=[]
        self.H_J=[]


    # custom functions
    def initialize_circuit_from_networkx(self,input_graph):

        self.G=nx.convert_node_labels_to_integers(input_graph, first_label=0, ordering='default')
        self.initialize_circuit()

        self.list_graph_nodes=list(self.G.nodes())
        self.list_graph_edges=list(self.G.edges())

    def initialize_circuit(self):

        e=self.G.number_of_edges()
        n=self.G.number_of_nodes()

        init_val=['#269ab3',0,0,5]
        init_attributes=['color','source','potential','conductivity']
        for i,val in enumerate(init_val):
            nx.set_node_attributes(self.G, val , name=init_attributes[i])

        for k in dict_nodes.keys():
            dict_nodes[k]=np.zeros(n)

        for k in dict_edges.keys():
            dict_edges[k]=np.ones(n)

        self.set_network_attributes()
        print('circuit(): initialized and ready for (some) action :)')

    #get incidence atrix and its transpose
    def get_incidence_matrices(self):

        B=nx.incidence_matrix(self.G,nodelist=self.list_graph_nodes,edgelist=self.list_graph_edges,oriented=True).toarray()
        BT=np.transpose(B)

        return B,BT

    # update network traits from dynamic data
    def set_network_attributes(self):

        #set potential node values
        for i,n in enumerate(self.list_graph_nodes):
            self.G.nodes[n]['label']=i
            self.G.nodes[n]['potential']=self.dict_nodes['potential']
        #set conductivity matrix
        for j,e in enumerate(self.list_graph_edges):
            self.G.edges[e]['conductivity']=self.dict_edges['conductivity']
            self.G.edges[e]['label']=j

    # clipp small edges & translate conductance into general edge weight

    def clipp_graph(self):

        #cut out edges which lie beneath a certain threshold value and export this clipped structure
        self.set_network_attributes()

        for e in self.list_graph_edges:
            if self.G.edges[e]['conductivity'] > self.threshold:
                self.H.add_edge(*e)
                for k in self.G.edges[e].keys():
                    self.H.edges[e][k]=self.G.edges[e][k]

        self.list_pruned_nodes=list(self.H.nodes())
        self.list_pruned_edges=list(self.H.edges())

        for n in list_pruned_nodes:
            for k in self.G.nodes[n].keys():
                self.H.nodes[n][k]=self.G.nodes[n][k]
            self.H_J.append(self.G.nodes[n]['source'])
        for e inlist_pruned_edges:
            self.H_C.append(self.H.edges[e]['conductivity'])

        self.H_C=np.asarray(self.H_C)
        self.H_J=np.asarray(self.H_J)
        if len(list(self.H.nodes()))==0:
            sys.exit('FAILED PRUNING')

    def calc_root_incidence(self):

        root=0
        sink=0

        for i,n in enumerate(self.list_graph_nodes):
            if self.G.nodes[n]['source'] >  0:
                root=n
            if K.G.nodes[n]['source'] <  0:
                sink=n

        E_1=list(self.G.edges(root))
        E_2=list(self.G.edges(sink))
        E_ROOT=[]
        E_SINK=[]
        for e in E_1:
            if e[0]!=root:
                E_ROOT+=list(self.G.edges(e[0]))
            else:
                E_ROOT+=list(self.G.edges(e[1]))

        for e in E_2:
            if e[0]!=sink:
                E_SINK+=list(self.G.edges(e[0]))
            else:
                E_SINK+=list(self.edges(e[1]))

        return E_ROOT,E_SINK
