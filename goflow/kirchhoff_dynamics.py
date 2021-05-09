# @Author:  Felix Kramer <kramer>
# @Date:   2021-05-08T20:35:25+02:00
# @Email:  kramer@mpi-cbg.de
# @Project: go-with-the-flow
# @Last modified by:   kramer
# @Last modified time: 2021-05-09T12:16:06+02:00
# @License: MIT

import random as rd
import networkx as nx
import numpy as np
import sys
import goflow.kirchhoff_init as kirchhoff_init

class dynamic_flow_circuit(kirchoff_init.circuit,object):

    def __init__(self):
        super(dynamic_flow_circuit,self).__init__()

        self.graph_mode={
            'simpleleaf':self.init_source_simpleleaf,
            'pointleaf':self.init_source_pointleaf_L,
            'rootleaf':self.init_source_rootleaf,
            'bilehex':self.init_source_bilehex,
            'bloodhex':self.init_source_bloodhex,
            'centralspawn':self.init_source_centralspawn,
            'pointleaf':self.init_source_pointleaf_R,
            'pointleaf':self.init_source_short_distance,
            'pointleaf':self.init_source_short_distance,
            'terminals':self.init_source_terminals,
            'multi':self.init_source_multi,
            'multi_cluster':self.init_source_multi_cluster,
            'max_distance':self.init_source_max_distance,
            'terminals_dipole':self.init_source_terminals_dipole,
            'terminals_monopole':self.init_source_terminals_monopole
        }
    # test consistency of conductancies & sources
    def test_consistency(self):

        self.set_network_attributes()
        tolerance=0.000001
        # check value consistency
        conductivities=np.fromiter(nx.get_edge_attributes(self.G, 'conductivity').values(),float)
        if len(np.where(conductivities <=0 )[0]) !=0:
            sys.exit('Error, conductivities negaitve/zero!')

        sources=np.fromiter(nx.get_node_attributes(self.G, 'source').values(),float)
        if np.sum(sources) > tolerance:
            sys.exit('Error, input and ouput flows not balanced!')
        else:
            print('set_source_landscape(): '+self.graph_mode+' is set and consistent :)')

    # set a certain set of boundary conditions for the given networks
    def set_source_landscape(self,mode):

        if mode in self.graph_mode.keys():

            self.graph_mode[mode]()

        else :
            sys.exit('Whooops, Error: Define Input/output-flows for  the network.')

        self.test_consistency()

    def init_source_simpleleaf(self):

        spine_nodes=1+2*int((-0.5+np.sqrt(0.25+(self.G.number_of_nodes()-1)/3.)))

        for j,n in enumerate(nx.nodes(self.G)):

            if n==(0,0):
                self.G.nodes[n]['source']=1*self.f
                self.G.nodes[n]['color']='#ee2323'
                self.J[j]=1*self.f
            elif n==(spine_nodes-1,0):
                self.G.nodes[n]['source']=-1*self.f
                self.G.nodes[n]['color']='#1eb22f'
                self.J[j]=-1*self.f
            else:
                self.J[j]=0

    def init_source_pointleaf_L(self):

        pos_x=[]
        for n in nx.nodes(self.G):
            pos_x.append(self.G.nodes[n]['pos'][0])
        min_x=np.argmin(pos_x)
        for j,n in enumerate(nx.nodes(self.G)):

            if j==min_x:

                self.G.nodes[n]['source']=((self.G.number_of_nodes()-1))*self.f
                self.G.nodes[n]['color']='#ee2323'
                self.J[j]=(self.G.number_of_nodes()-1)*self.f

            else:

                self.G.nodes[n]['source']=-1*self.f
                self.G.nodes[n]['color']='#1eb22f'
                self.J[j]=-1*self.f

    def init_source_pointleaf_R(self):

        pos_x=[]
        for n in nx.nodes(self.G):
            pos_x.append(self.G.nodes[n]['pos'][0])
        max_x=np.argmax(pos_x)
        for j,n in enumerate(nx.nodes(self.G)):

            if j==max_x:

                self.G.nodes[n]['source']=((self.G.number_of_nodes()-1))*self.f
                self.G.nodes[n]['color']='#ee2323'
                self.J[j]=(self.G.number_of_nodes()-1)*self.f

            else:

                self.G.nodes[n]['source']=-1*self.f
                self.G.nodes[n]['color']='#1eb22f'
                self.J[j]=-1*self.f

    def init_source_rootleaf(self):

        diam=int((self.G.number_of_nodes())**0.25)
        stack=0
        n_stack=0
        for j,n in enumerate(nx.nodes(self.G)):

            r_pos=self.G.nodes[keys]['pos']
            d=np.dot(r_pos,r_pos)
            if d <= diam :
                self.G.node[n]['source']=1*self.f
                self.G.node[n]['color']='#ee2323'
                self.J[j]=np.exp(-d)*self.G.number_of_nodes()*self.f
                stack+=self.J[j]
                n_stack+=1

            else:
                self.G.node[n]['source']=-1
                self.G.node[n]['color']='#1eb22f'
                self.J[j]=-1

        n_stack=self.G.number_of_nodes()-n_stack

        for j,n in enumerate(nx.nodes(self.G)):

            if self.G.node[n]['source']==-1 :
                self.G.node[n]['source']=-stack*self.f/n_stack
                self.J[j]=-stack*self.f/n_stack

    def init_source_bilehex(self):

        self.remove_center()
        N=self.G.number_of_nodes()
        for i,n in enumerate(nx.nodes(self.G)):
            #fill the tissue
            self.J[i]=1*self.f
            self.G.nodes[n]['source']=1*self.f
            self.G.nodes[n]['color']='#1eb22f'
        x=((6. - N)*self.f/6.)
        for i,n in enumerate(nx.nodes(self.G)):
            #test whether node is an edgepoint
            if self.G.degree(n)==3:
                self.G.nodes[n]['source']=x
                self.G.nodes[n]['color']='#ee2323'
                self.J[i]=x

    def init_source_bloodhex(self):

        # self.remove_center()
        N=self.G.number_of_nodes()

        # if K.stacks == 1:
        #
        # for i,n in enumerate(nx.nodes(self..G)):
        #     #fill the tissue
        #     # self.J[i]=-1
        #     # self.G.nodes[n]['source']=-1
        #     self.J[i]=0.
        #     self.G.nodes[n]['source']=0.
        #     self.G.nodes[n]['color']='#1eb22f'
        x=N*self.f/6.
        dict_central=nx.betweenness_centrality(self.G)
        central_sorted=sorted(dict_central,key=dict_central.__getitem__)

        for i,n in enumerate(self.G.nodes()):
            #test whether node is an edgepoint
            if self.G.degree(n)==3:
                self.G.nodes[n]['source']=x
                self.G.nodes[n]['color']='#ee2323'
                self.J[i]=x
            elif n==central_sorted[-1]:
                self.G.nodes[n]['source']=(-6.)*x
                self.G.nodes[n]['color']='#ee2323'
                self.J[i]=(-6.)*x
            else:
                self.G.nodes[n]['source']=0.
                self.G.nodes[n]['color']='#ee2323'
                self.J[i]=0.
        self.num_sources=6
        # x=-arb_add
        # for i,n in enumerate(nx.nodes(self.G)):
        #     #test whether node is inner circle
        #     if self.G.degree(n)==5:
        #         self.G.nodes[n]['source']+=x
        #         self.G.nodes[n]['color']='#1eb22f'
        #         self.J[i]+=x

    def init_source_centralspawn(self):

        dict_central=nx.betweenness_centrality(self.G)
        central_sorted=sorted(dict_central,key=dict_central.__getitem__)

        for j,n in enumerate(self.G.nodes()):

            if n==central_sorted[-1]:

                self.G.nodes[n]['source']=((self.G.number_of_nodes()-1))*self.f
                self.G.nodes[n]['color']='#ee2323'
                self.J[j]=(self.G.number_of_nodes()-1)*self.f

            else:

                self.G.nodes[n]['source']=-1*self.f
                self.G.nodes[n]['color']='#1eb22f'
                self.J[j]=-1*self.f

    def init_conductivity_plexus(self):

        d=np.amax(np.absolute(self.J)) * 0.5
        M=self.G.number_of_edges()
        for m in range(M):

            x=int(0.5+rd.random())
            sign=(-1)**x
            self.C[m]+=sign*d*rd.random()

    def init_source_short_distance(self):

        dist={}
        for j,n in enumerate(self.G.nodes()):
            dist[n]=np.linalg.norm(self.G.nodes[n]['pos'])

        sorted_dist=sorted(dist,key=dist.__getitem__)

        for j,n in enumerate(self.G.nodes()):

            if n==sorted_dist[0]:

                self.G.nodes[n]['source']=((self.G.number_of_nodes()-1))*self.f
                self.G.nodes[n]['color']='#ee2323'
                self.J[j]=(self.G.number_of_nodes()-1)*self.f

            else:

                self.G.nodes[n]['source']=-1*self.f
                self.G.nodes[n]['color']='#1eb22f'
                self.J[j]=-1*self.f

    def init_source_long_distance(self):

        dist={}
        for j,n in enumerate(self.G.nodes()):
            dist[n]=np.linalg.norm(self.G.nodes[n]['pos'])
        sorted_dist=sorted(dist,key=dist.__getitem__,reverse=True)

        for j,n in enumerate(self.G.nodes()):

            if n==sorted_dist[0]:

                self.G.nodes[n]['source']=((self.G.number_of_nodes()-1))*self.f
                self.G.nodes[n]['color']='#ee2323'
                self.J[j]=(self.G.number_of_nodes()-1)*self.f

            else:

                self.G.nodes[n]['source']=-1*self.f
                self.G.nodes[n]['color']='#1eb22f'
                self.J[j]=-1*self.f

    def init_source_max_distance(self):

        dist={}
        for j,n in enumerate(self.G.nodes()):
            dist[n]=np.linalg.norm(self.G.nodes[n]['pos'])
        sorted_dist=sorted(dist,key=dist.__getitem__,reverse=True)

        for j,n in enumerate(self.G.nodes()):

            if n==sorted_dist[0]:

                self.G.nodes[n]['source']=self.G.number_of_nodes()*self.f
                self.G.nodes[n]['color']='#ee2323'
                self.J[j]=self.G.number_of_nodes()*self.f

            elif n==sorted_dist[-1]:

                self.G.nodes[n]['source']=-(self.G.number_of_nodes())*self.f
                self.G.nodes[n]['color']='#1eb22f'
                self.J[j]=-(self.G.number_of_nodes())*self.f

            else:
                self.G.nodes[n]['source']=0.
                self.G.nodes[n]['color']='k'
                self.J[j]=0.

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
