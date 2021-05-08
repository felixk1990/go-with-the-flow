# @Author:  Felix Kramer <kramer>
# @Date:   2021-05-08T20:35:25+02:00
# @Email:  kramer@mpi-cbg.de
# @Project: go-with-the-flow
# @Last modified by:   kramer
# @Last modified time: 2021-05-09T00:42:05+02:00
# @License: MIT

import random as rd
import networkx as nx
import numpy as np
import sys
import goflow.kirchhoff_init as kirchhoff_init

class kirchhoff_dynamics(kirchoff_init,object):

    # generate an edge weight pattern to calibrate for cycle coalescence algorithm
    def generate_pattern(self,mode):
        if 'random' in mode:
            for e in self.G.edges():
                self.G.edges[e]['conductivity']=rd.uniform(0.,1.)*5.
        if 'gradient' in mode:
            list_n=list(self.G.nodes())
            # ref_p=self.G.nodes[list_n[0]]['pos']
            ref_p=np.array([0,0,1])
            for e in self.G.edges():
                p=(np.array(self.G.edges[e]['slope'][0])+np.array(self.G.edges[e]['slope'][1]))/2.
                r=np.linalg.norm(p-ref_p)
                self.G.edges[e]['conductivity']=3./r
        if 'bigradient' in mode:
            ref1=0
            ref2=int((self.G.number_of_nodes()-1)/2)
            list_n=list(self.G.nodes())
            ref_p1=self.G.nodes[list_n[ref1]]['pos']
            ref_p2=self.G.nodes[list_n[ref2]]['pos']
            for e in self.G.edges():
                p=(np.array(self.G.edges[e]['slope'][0])+np.array(self.G.edges[e]['slope'][1]))/2.
                r1=np.linalg.norm(p-ref1)
                r2=np.linalg.norm(p-ref2)
                self.G.edges[e]['conductivity']=(3./r1 +2./r2)

        if 'nested_square' in mode:
            # so far only for cube /square
            dim=len(list(nx.get_node_attributes(self.G,'pos').values())[0])
            w=5.
            nx.set_edge_attributes(self.G,w,'conductivity')
            nx.set_edge_attributes(self.G,False,'tracked')
            if dim==2:
                corners=[n for n in self.G.nodes() if self.G.degree(n)==2]
            if dim==3:
                corners=[n for n in self.G.nodes() if self.G.degree(n)==3]
            outskirt_paths=[]

            for i,c1 in enumerate(corners[:-1]):
                for j,c2 in enumerate(corners[i+1:]):
                    connection_existent=False
                    p1=self.G.nodes[c1]['pos']

                    p2=self.G.nodes[c2]['pos']

                    if dim==2:
                        connection_existent = (p1[0]==p2[0])  or  (p1[1]==p2[1])
                    if dim==3:
                        connection_existent = ((p1[0]==p2[0] and p1[1]==p2[1]) or (p1[0]==p2[0] and p1[2]==p2[2]) or (p1[1]==p2[1] and p1[2]==p2[2]))
                    if connection_existent:
                        outskirt_paths.append(nx.shortest_path(self.G,source=c1,target=c2))

            for p in outskirt_paths:
                for i,n in enumerate(p[1:]):
                    self.G.edges[(p[i],n)]['conductivity']=w
                    self.G.edges[(p[i],n)]['tracked']=True


            system_scale=len(outskirt_paths[0])-1
            divisions=1.
            divide_conquer=True
            while divide_conquer:

                divisions*=2.
                system_scale/=2.
                if system_scale==1.:
                    divide_conquer=False
                parts_of_the_line=[]
                divine_paths=[]
                delta_w_paths=[]
                for p in outskirt_paths:
                    for j in range(int(divisions)):
                        if j%2==1:
                            parts_of_the_line.append(p[int(len(p)*j/divisions)])
                dict_face={}
                for i,c1 in enumerate(parts_of_the_line[:-1]):
                    for j,c2 in enumerate(parts_of_the_line[i+1:]):
                        connection_existent=False
                        p1=self.G.nodes[c1]['pos']
                        p2=self.G.nodes[c2]['pos']

                        if dim==2:
                            connection_existent = (p1[0]==p2[0])  or  (p1[1]==p2[1])
                        if dim==3:
                            connection_existent = ((p1[0]==p2[0] and p1[1]==p2[1]) or (p1[0]==p2[0] and p1[2]==p2[2]) or (p1[1]==p2[1] and p1[2]==p2[2]))

                        if connection_existent:
                            divine_paths.append(nx.shortest_path(self.G,source=c1,target=c2))
                            delta_w_paths.append(0.)
                            if (p1[0]==p2[0]):
                                delta_w_paths[-1]=0.5
                                if dim==3:
                                    if (p1[2]!=p2[2]):
                                        delta_w_paths[-1]=0.
                max_d=np.amax([len(d) for d in divine_paths])
                for id_d,d in enumerate(divine_paths):
                    if len(d)==max_d:
                        for i,n in enumerate(d[1:]):
                            if not self.G.edges[(d[i],n)]['tracked']:
                                self.G.edges[(d[i],n)]['conductivity']=(delta_w_paths[id_d]+w)/divisions
                                self.G.edges[(d[i],n)]['tracked']=True
                # outskirt_paths+=divine_paths

        self.translate_weight('conductivity')

    # test consistency of conductancies & sources
    def test_consistency(self):

        self.set_network_attributes()
        sumit=0
        tolerance=0.00001

        for e in nx.edges(self.G):

            if self.G.edges[e]['conductivity']<=0:
                sys.exit('Error, conductivity fatality!')

        for n in nx.nodes(self.G):
            sumit+=self.G.nodes[n]['source']

        if sumit>tolerance:
            sys.exit('Error, input and ouput flows not balanced!')
        else:
            print('set_source_landscape(): '+self.graph_mode+' is set and consistent :)')

    # set a certain set of boundary conditions for the given networks
    def set_source_landscape(self,mode):

        # reinitialze circuit
        H=nx.Graph(self.G)
        self.G=nx.Graph()
        self.counter_e=0
        self.counter_n=0
        dict_nodes={}
        for i,n in enumerate(H.nodes()):
            idx_n=self.count_n()-1
            self.G.add_node(idx_n,pos=H.nodes[n]['pos'],label=H.nodes[n]['label'])
            dict_nodes.update({n:idx_n})

        for i,e in enumerate(H.edges()):
            idx_e=self.count_e()
            self.G.add_edge(dict_nodes[e[0]],dict_nodes[e[1]],slope=(H.nodes[e[0]]['pos'],H.nodes[e[1]]['pos']),label=H.edges[e]['label'])
        self.initialize_circuit()
        #modes:
        #1: single source-sink
        #2: single source -all sinks apart from that, left-handed
        #3: small area of sources
        #4: bile, lobule structure sinks at edgepoints, sources everywehere else
        #5: blood, lobule structure sourcse at edgepoints, sinks everywehere else
        #6: central source, all sinks part from that
        #7: single source -all sinks apart from that, right-handed
        #8: single source -all sinks apart from that, shortest distance
        #9: single source -all sinks apart from that, longest distance
        if mode == 1    :
            self.init_source_simpleleaf()
            self.graph_mode='simpleleaf'
        elif mode == 2  :
            self.init_source_pointleaf_L()
            self.graph_mode='pointleaf'
        elif mode == 3  :
            self.init_source_rootleaf()
            self.graph_mode='rootleaf'
        elif mode == 4  :
            self.init_source_bilehex()
            self.graph_mode='bilehex'
        elif mode == 5  :
            self.init_source_bloodhex()
            self.graph_mode='bloodhex'
        elif mode == 6  :
            self.init_source_centralspawn()
            self.graph_mode='centralspawn'
        elif mode == 7 :
            self.init_source_pointleaf_R()
            self.graph_mode='pointleaf'
        elif mode == 8 :
            self.init_source_short_distance()
            self.graph_mode='pointleaf'
        elif mode == 9 :
            self.init_source_long_distance()
            self.graph_mode='pointleaf'
        elif mode == 10 :
            self.init_source_terminals()
            self.graph_mode='terminals'
        elif mode == 11 :
            self.init_source_multi()
            self.graph_mode='multi'
        elif mode == 12:
            self.init_source_multi_cluster()
            self.graph_mode='multi_cluster'

        elif mode == 13:
            self.init_source_max_distance()
            self.graph_mode='max_distance'

        elif mode == 14 :
            self.init_source_terminals_dipole()
            self.graph_mode='terminals_dipole'

        elif mode == 15 :
            self.init_source_terminals_monopole()
            self.graph_mode='terminals_monopole'

        else :
            sys.exit('Whooops, Error: Define Input/output-flows for  the network.')

        self.C=np.ones(self.G.number_of_edges())*np.amax(np.absolute(self.J))
        self.test_consistency()
        self.B, self.BT=self.get_incidence_matrices()

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

    def calc_root_incidence(self):
        root=0
        sink=0
        # q=[]
        for i,n in enumerate(self.G.nodes()):
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
