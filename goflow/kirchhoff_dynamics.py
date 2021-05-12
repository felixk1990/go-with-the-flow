# @Author:  Felix Kramer <kramer>
# @Date:   2021-05-08T20:35:25+02:00
# @Email:  kramer@mpi-cbg.de
# @Project: go-with-the-flow
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-05-12T23:14:30+02:00
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

            'centralspawn':self.init_source_centralspawn,
            'root_short':self.init_source_root_short,
            'root_long':self.init_source_root_long,
            'terminals':self.init_source_terminals,
            'multi_source':self.init_source_multi,
            'multi_source_cluster':self.init_source_multi_cluster,
            'terminals_dipole':self.init_source_terminals_dipole,
            'terminals_monopole':self.init_source_terminals_monopole

        }

    # set a certain set of boundary conditions for the given networks
    def set_source_landscape(self,mode):

        if mode in self.graph_mode.keys():

            self.graph_mode[mode]()

        else :
            sys.exit('Whooops, Error: Define Input/output-flows for  the network.')

        self.test_consistency()

    def get_pos(self):

        pos_key='pos'
        reset_layout=False
        for j,n in enumerate(self.G.nodes()):
            if pos_key not in self.G.nodes[n]:
                reset_layout=True
        if reset_layout:
            print('set networkx.spring_layout()')
            pos = nx.spring_layout(self.G)
        else:
            pos = nx.get_node_attributes(self.G,'pos')

        return pos

    # different functions versus custom function
    def init_source_custom(self):
        return 0

    def init_source_centralspawn(self):

        centrality=nx.betweenness_centrality(self.G)
        centrality_sorted=sorted(centrality,key=centrality.__getitem__)

        for j,n in enumerate(self.G.nodes()):

            if n==centrality_sorted[-1]:

                self.G.nodes[n]['source']=((self.G.number_of_nodes()-1))*self.scales['flow']
                self.G.nodes[n]['color']='#ee2323'
                self.nodes['source'][j]=(self.G.number_of_nodes()-1)*self.scales['flow']

            else:

                self.G.nodes[n]['source']=-1*self.scales['flow']
                self.G.nodes[n]['color']='#1eb22f'
                self.nodes['source'][j]=-1*self.scales['flow']

    def init_source_root_short(self):

        # check whether geometric layout has been set
        pos=self.get_pos()

        # check for root closests to coordinate origin
        dist={}
        for n,p in pos.items():
            dist[n]=np.linalg.norm(p)
        sorted_dist=sorted(dist,key=dist.__getitem__)

        for j,n in enumerate(self.G.nodes()):

            if n==sorted_dist[0]:

                self.G.nodes[n]['source']=((self.G.number_of_nodes()-1))*self.scales['flow']
                self.G.nodes[n]['color']='#ee2323'
                self.nodes['source'][j]=(self.G.number_of_nodes()-1)*self.scales['flow']

            else:

                self.G.nodes[n]['source']=-1*self.scales['flow']
                self.G.nodes[n]['color']='#1eb22f'
                self.nodes['source'][j]=-1*self.scales['flow']

    def init_source_root_long(self):

        # check whether geometric layout has been set
        pos=self.get_pos()

        # check for root closests to coordinate origin
        dist={}
        for n,p in pos.items():
            dist[n]=np.linalg.norm(p)
        sorted_dist=sorted(dist,key=dist.__getitem__,reverse=True)

        for j,n in enumerate(self.G.nodes()):

            if n==sorted_dist[0]:

                self.G.nodes[n]['source']=((self.G.number_of_nodes()-1))*self.scales['flow']
                self.G.nodes[n]['color']='#ee2323'
                self.nodes['source'][j]=(self.G.number_of_nodes()-1)*self.scales['flow']

            else:

                self.G.nodes[n]['source']=-1*self.scales['flow']
                self.G.nodes[n]['color']='#1eb22f'
                self.nodes['source'][j]=-1*self.scales['flow']

    def init_source_terminals(self):

        dist=[]
        adj=[]
        list_n=list(nx.nodes(self.G))
        for j,n in enumerate(list_n):
            p=self.G.nodes[n]['pos']
            dist.append(np.linalg.norm(p[0]))
            if j<len(list_n)-1:
                for i,m in enumerate(list_n[j+1:]):
                    q=self.G.nodes[m]['pos']
                    adj.append(np.linalg.norm(np.subtract(p,q)))
        max_x=np.amax(dist)
        min_x=np.amin(dist)

        max_idx=np.where(dist==max_x)[0]
        min_idx=np.where(dist==min_x)[0]
        madj=np.amin(adj)

        self.initialize_circuit()

        for j,n in enumerate(nx.nodes(self.G)):

            self.G.nodes[n]['source']=0.
            self.G.nodes[n]['color']='#1eb22f'
            self.J[j]=0.

        for j in max_idx:
            n=list_n[j]
            self.G.nodes[n]['source']=-1*self.f
            self.G.nodes[n]['color']='#ee2323'
            self.J[j]=-1.*self.f
        for j in min_idx:
            n=list_n[j]
            self.G.nodes[n]['source']=1*self.f
            self.G.nodes[n]['color']='#ee2323'
            self.J[j]=1*self.f

    def init_source_terminals_dipole(self):

        dist={}
        list_n=list(nx.nodes(self.G))
        for j,n in enumerate(list_n[:-2]):
            for i,m in enumerate(list_n[j+1:]):
                path=nx.shortest_path(self.G,source=n,target=m)
                dist[(n,m)]=len(path)
        max_len=np.amax(list(dist.values()))
        push=[]
        for key in dist.keys():
            if dist[key]==max_len:
                push.append(key)

        idx=np.random.choice(range(len(push)))
        source,sink=push[idx]
        for j,n in enumerate(nx.nodes(self.G)):

            if n==source:

                self.G.nodes[n]['source']=1*self.f
                self.G.nodes[n]['color']='#ee2323'
                self.J[j]=1*self.f

            elif n==sink:

                self.G.nodes[n]['source']=-1*self.f
                self.G.nodes[n]['color']='#ee2323'
                self.J[j]=-1.*self.f

            else:

                self.G.nodes[n]['source']=0.
                self.G.nodes[n]['color']='#1eb22f'
                self.J[j]=0.

    def init_source_terminals_monopole(self):

        dist={}
        list_n=list(nx.nodes(self.G))
        X=np.zeros(len(self.G.nodes[list_n[0]]['pos']))
        for n in list_n:
            X=np.add(self.G.nodes[n]['pos'],X)
        X=X/len(list_n)
        for n in list_n:
            dist[n]=np.linalg.norm(np.subtract(X,self.G.nodes[n]['pos']))
        sorted_dist=sorted(dist,key=dist.__getitem__)


        for j,n in enumerate(nx.nodes(self.G)):

            if n==sorted_dist[0]:

                self.G.nodes[n]['source']=(nx.number_of_nodes(self.G)-1)*self.f
                self.G.nodes[n]['color']='#ee2323'
                self.J[j]=(nx.number_of_nodes(self.G)-1)*self.f

            else:

                self.G.nodes[n]['source']=-1*self.f
                self.G.nodes[n]['color']='#ee2323'
                self.J[j]=-1.*self.f

    def init_source_multi(self):

        # dist={}
        # for j,n in enumerate(self.G.nodes()):
        #     dist[n]=np.linalg.norm(self.G.nodes[n]['pos'])
        #
        # sorted_close=sorted(dist,key=dist.__getitem__)
        #
        # sorted_far=sorted(dist,key=dist.__getitem__,reverse=True)
        idx=np.random.choice( list(self.G.nodes()),size=self.num_sources )

        for j,n in enumerate(self.G.nodes()):

            # if (n==sorted_close[0] or n==sorted_far[0]):
            if n in idx:

                self.G.nodes[n]['source']=1.*self.f
                self.G.nodes[n]['color']='#ee2323'
                self.J[j]=1.*self.f

            else:

                self.G.nodes[n]['source']=-1*self.f
                self.G.nodes[n]['color']='#1eb22f'
                self.J[j]=-1*self.f

    def init_source_multi_cluster(self):


        idx=np.random.choice( list(self.G.nodes()),size=self.num_sources )
        dict_nodes={}
        for j,n in enumerate(self.G.nodes()):
            dict_nodes[n]=j
            if n in idx:

                self.G.nodes[n]['source']=1.*self.f
                self.G.nodes[n]['color']='#ee2323'
                self.J[j]=1.*self.f

            else:

                self.G.nodes[n]['source']=-1*self.f
                self.G.nodes[n]['color']='#1eb22f'
                self.J[j]=-1*self.f

        for j,n in enumerate(self.G.neighbors(idx[0])):

            self.G.nodes[n]['source']=1.*self.f
            self.G.nodes[n]['color']='#ee2323'
            self.J[dict_nodes[n]]=1.*self.f

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
