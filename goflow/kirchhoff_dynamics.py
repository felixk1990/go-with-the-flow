# @Author:  Felix Kramer <kramer>
# @Date:   2021-05-08T20:35:25+02:00
# @Email:  kramer@mpi-cbg.de
# @Project: go-with-the-flow
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-05-14T19:55:03+02:00
# @License: MIT

import random as rd
import networkx as nx
import numpy as np
import sys
import kirchhoff_init as kirchhoff_init

def initialize_flow_circuit_from_networkx(input_graph):

    kirchhoff_graph=dynamic_flow_circuit()
    kirchhoff_graph.default_init(input_graph)

    return kirchhoff_graph

def initialize_flux_circuit_from_networkx(input_graph):

    kirchhoff_graph=dynamic_flux_circuit()
    kirchhoff_graph.default_init(input_graph)

    return kirchhoff_graph


class dynamic_flow_circuit(kirchhoff_init.circuit,object):

    def __init__(self):
        super(dynamic_flow_circuit,self).__init__()

        self.graph_mode={

            'root_centrallity':self.init_source_root_central_centrallity,
            'root_geometric':self.init_source_root_central_geometric,
            'root_short':self.init_source_root_short,
            'root_long':self.init_source_root_long,
            'dipole_border':self.init_source_dipole_border,
            'dipole_point':self.init_source_dipole_point,
            'root_multi':self.init_source_root_multi,
            'custom':self.init_source_custom
        }

    # set a certain set of boundary conditions for the given networks
    def set_source_landscape(self,mode,**kwargs):

        # optional keywords
        if 'num_sources' in kwargs:
            self.graph['num_sources']= kwargs['num_sources']

        elif 'custom_sources' in kwargs:
            self.custom= kwargs['custom_sources']

        else:
            print('Warning: Not recognizing certain keywords')
        # call init sources
        if mode in self.graph_mode.keys():

            self.graph_mode[mode]()

        else :
            sys.exit('Whooops, Error: Define Input/output-flows for  the network.')

        self.test_consistency()

    def set_potential_landscape(self,mode):

        # todo
        return 0

    # different functions versus custom function
    def init_source_custom(self):
        # todo
        return 0

    def init_source_root_central_centrallity(self):

        centrality=nx.betweenness_centrality(self.G)
        centrality_sorted=sorted(centrality,key=centrality.__getitem__)

        self.set_root_leaves_relationship(centrality_sorted[-1])

    def init_source_root_central_geometric(self):

        pos=self.get_pos()
        X=np.mean(list(pos.values()),axis=0)

        dist={}
        for n in self.list_graph_nodes:
            dist[n]=np.linalg.norm(np.subtract(X,pos[n]))
        sorted_dist=sorted(dist,key=dist.__getitem__)

        self.set_root_leaves_relationship(sorted_dist[0])

    def init_source_root_short(self):

        # check whether geometric layout has been set
        pos=self.get_pos()

        # check for root closests to coordinate origin
        dist={}
        for n,p in pos.items():
            dist[n]=np.linalg.norm(p)
        sorted_dist=sorted(dist,key=dist.__getitem__)

        self.set_root_leaves_relationship(sorted_dist[0])

    def init_source_root_long(self):

        # check whether geometric layout has been set
        pos=self.get_pos()

        # check for root closests to coordinate origin
        dist={}
        for n,p in pos.items():
            dist[n]=np.linalg.norm(p)
        sorted_dist=sorted(dist,key=dist.__getitem__,reverse=True)

        self.set_root_leaves_relationship(sorted_dist[0])

    def init_source_dipole_border(self):

        pos=self.get_pos()
        dist={}
        for n,p in pos.items():
            dist[n]=np.linalg.norm(p)

        vals=list(dist.values())
        max_x=np.amax(vals)
        min_x=np.amin(vals)

        max_idx=[]
        min_idx=[]
        for k,v in dist.items():
            if v == max_x:
                max_idx.append(k)

            elif v == min_x:
                min_idx.append(k)

        self.set_poles_relationship(max_idx,min_idx)

    def init_source_dipole_point(self):

        pos=self.get_pos()
        dist={}
        for j,n in enumerate(self.list_graph_nodes[:-2]):
            for i,m in enumerate(self.list_graph_nodes[j+1:]):
                path=nx.shortest_path(self.G,source=n,target=m)
                dist[(n,m)]=len(path)
        max_len=np.amax(list(dist.values()))
        push=[]
        for key in dist.keys():
            if dist[key]==max_len:
                push.append(key)

        idx=np.random.choice(range(len(push)))
        source,sink=push[idx]

        self.set_poles_relationship([source],[sink])

    def init_source_root_multi(self):

        idx=np.random.choice( self.list_graph_nodes,size=self.graph['num_sources'] )
        self.node_color=['#ee2323','#1eb22f']
        self.nodes_source=[self.G.number_of_nodes()/self.graph['num_sources']-1,-1]

        for j,n in enumerate(self.list_graph_nodes):

            if n in idx:

                self.set_source_attributes(j,n,0)

            else:

                self.set_source_attributes(j,n,1)

    # auxillary function for the block above
    def set_root_leaves_relationship(self,root):

        self.node_color=['#ee2323','#1eb22f']
        self.nodes_source=[self.G.number_of_nodes()-1,-1]
        for j,n in enumerate(self.list_graph_nodes):

            if n==root:
                idx=0

            else:
                idx=1

            self.set_source_attributes(j,n,idx)

    def set_poles_relationship(self,sources,sinks):

        print(sources)
        print(sinks)
        self.node_color=['#ee2323','#ee2323','#1eb22f']
        self.nodes_source=[1,-1,0]

        for j,n in enumerate(self.list_graph_nodes):
            self.set_source_attributes(j,n,2)

        for i,s in enumerate(sources):
            for j,n in enumerate(self.list_graph_nodes):
                if n==s:
                    self.set_source_attributes(j,s,0)

        for i,s in enumerate(sinks):
            for j,n in enumerate(self.list_graph_nodes):
                if n==s:
                    self.set_source_attributes(j,s,1)

    def set_source_attributes(self,j,node,idx):

        self.G.nodes[node]['source']=self.nodes_source[idx]*self.scales['flow']
        self.G.nodes[node]['color']=self.node_color[idx]
        self.nodes['source'][j]=self.nodes_source[idx]*self.scales['flow']

    # todo
    def init_conductivity_plexus(self):

        d=np.amax(np.absolute(self.J)) * 0.5
        M=self.G.number_of_edges()
        for m in range(M):

            x=int(0.5+rd.random())
            sign=(-1)**x
            self.C[m]+=sign*d*rd.random()

class dynamic_flux_circuit(dynamic_flow_circuit,object):

    def __init__(self):
        super(dynamic_flux_circuit,self).__init__()
        self.J_C=[]
    def set_solute_flux_boundary(self,flux):

        for j,n in enumerate(nx.nodes(self.G)):

            if self.G.nodes[n]['source'] >0:
                self.J_C[j]=flux
            elif self.G.nodes[n]['source'] < 0:
                self.J_C[j]=-flux
            else:
                self.J_C[j]=0.

    def initiate_adv_diff_abs_network(self,dict_pars):

        self.threshold=0.01
        self.set_source_landscape(dict_pars['graph_mode'])
        self.set_solute_flux_boundary(dict_pars['flux'])
        self.sum_flux=np.sum( self.J_C[j] for j,n in enumerate(self.G.nodes()) if self.G.nodes[n]['source']>0 )
        self.C0=dict_pars['c0']
        self.phi0=dict_pars['phi']*self.sum_flux/nx.number_of_edges(self.G)
        self.D=dict_pars['D']

        m=nx.number_of_edges(self.G)
        n=nx.number_of_nodes(self.G)
        self.beta=np.ones(m)*dict_pars['beta']
        self.R=np.add(np.ones(m),np.random.rand(m))*dict_pars['R']
        self.C=np.power(self.R,4)*self.k
        self.G.graph['conductance']=self.k

    def set_terminals_potentials(self,p0):
        idx_potential=[]
        idx_sources=[]
        for j,n in enumerate(nx.nodes(self.G)):

            if self.G.nodes[n]['source']>0:

                self.G.nodes[n]['potential']=1
                self.G.nodes[n]['color']='#ee2323'
                self.V[j]=p0
                idx_potential.append(j)
            elif self.G.nodes[n]['source']<0:

                self.G.nodes[n]['potential']=0.
                self.G.nodes[n]['color']='#ee2323'
                self.V[j]=0.
                idx_potential.append(j)
            else:

                self.G.nodes[n]['source']=0.
                self.G.nodes[n]['color']='#1eb22f'
                self.J[j]=0.
                idx_sources.append(j)

        self.G.graph['sources']=idx_sources
        self.G.graph['potentials']=idx_potential
