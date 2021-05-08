import networkx as nx
import numpy as np
import scipy as sc
from scipy import sparse
import scipy.linalg as lina
from scipy.spatial import Voronoi
import random as rd
import sys

class kirchhoff_network:

    def __init__(self):
        self.counter_n=0
        self.counter_e=0

        self.C=[]
        self.V=[]
        self.F=[]
        self.J=[]
        self.E=[]

        self.G=nx.DiGraph()

        self.k=1
        self.f=1
        self.l=1
        self.stacks=1
        self.spine=1
        self.radius=1
        self.graph_mode=''
        self.threshold=0.001
        self.num_sources=1
        self.epsilon=1.
        # clipped graph_mod
        self.H=nx.Graph()
        self.H_C=[]
    # custom functions
    def count_n(self):
        self.counter_n+=1
        return self.counter_n

    def count_e(self):
        self.counter_e+=1
        return self.counter_e

    def initialize_circuit(self):

        nx.set_node_attributes(self.G, '#269ab3', name='color')
        nx.set_node_attributes(self.G, 0, name='source')
        nx.set_node_attributes(self.G, 0, name='potential')
        nx.set_edge_attributes(self.G, 5, name='conductivity')

        self.C=np.ones(self.G.number_of_edges())
        self.V=np.zeros(self.G.number_of_nodes())
        self.J=np.zeros(self.G.number_of_nodes())
        self.J_C=np.zeros(self.G.number_of_nodes())
        self.F=np.zeros(self.G.number_of_edges())
        self.set_network_attributes()
        print('kirchhoff_network(): initialized and ready for (some) action :)')

    #get incidence atrix and its transpose
    def get_incidence_matrices(self):

        B=nx.incidence_matrix(self.G,nodelist=list(self.G.nodes()),edgelist=list(self.G.edges()),oriented=True).toarray()
        BT=np.transpose(B)

        return B,BT

    # update network traits from dynamic data
    def set_network_attributes(self):
        #set potential node values
        for i,n in enumerate(self.G.nodes()):
            self.G.nodes[n]['label']=i
            self.G.nodes[n]['potential']=self.V[i]
        #set conductivity matrix
        for j,e in enumerate(self.G.edges()):
            self.G.edges[e]['conductivity']=self.C[j]
            self.G.edges[e]['label']=j

    # clipp small edges & translate conductance into general edge weight
    def set_structure_coefficient_fluctuation(self):


        B=nx.incidence_matrix(self.H,nodelist=self.list_n,edgelist=self.list_e,oriented=True).toarray()
        BT=np.transpose(B)
        num_n=nx.number_of_nodes(self.G)
        x=np.where(self.H_J > 0)[0]
        idx=np.where(self.H_J < 0)[0]
        N=len(idx)
        M=len(x)

        U=np.zeros((num_n,num_n))
        V=np.zeros((num_n,num_n))

        m_sq=1./float(M*M)
        m=1./float(M)
        n_sq_m_sq=N*N*m_sq
        nm=N*m
        n_m_sq=N*m_sq

        for i in range(num_n):
            for j in range(num_n):
                f=0.
                g1=0.
                g2=0.
                h=0.
                delta=0.

                if i==j and (i in idx) and (j in idx):
                    delta=1.

                if (i in x) and (j in idx):
                    g1=1.

                if (j in x) and (i in idx):
                    g2=1.

                if (i in x) and (j in x):
                    f=1.

                if (i in idx) and (j in idx):
                    h=1.

                U[i,j]= ( f*n_sq_m_sq - nm*(g1+g2) + h )
                V[i,j]= ( f*n_m_sq - m*(g1+g2) + delta )
        D=np.matmul(B,np.matmul(np.diag(self.H_C),BT))
        ID=lina.pinv(D)
        BID=np.matmul(BT,ID)
        BIDT=np.transpose(BID)
        VA=np.matmul(BID,np.matmul(V,BIDT))
        UA=np.matmul(BID,np.matmul(U,BIDT))

        for j,e in enumerate(self.list_e):
            self.H.edges[e]['coefficient_l3']=VA[j,j]/UA[j,j]

    def clipp_graph(self):
        #cut out edges which lie beneath a certain threshold value and export this clipped structure
        self.set_network_attributes()
        self.H_C=[]
        self.H_J=[]
        self.H=nx.Graph()

        # for n in self.G.nodes():
        #     self.H.add_node(n,pos=self.G.nodes[n]['pos'],source=self.G.nodes[n]['source'],label=self.G.nodes[n]['label'])
        # for e in list(self.G.edges()):
        #     if self.G.edges[e]['conductivity'] > self.threshold:
        #         self.H.add_edge(*e,conductivity=self.G.edges[e]['conductivity'],slope=self.G.edges[e]['slope'],label=self.G.edges[e]['label'],sign=self.G.edges[e]['sign'])
        for e in list(self.G.edges()):
            if self.G.edges[e]['conductivity'] > self.threshold:
                # self.H.add_edge(*e,conductivity=self.G.edges[e]['conductivity'],slope=self.G.edges[e]['slope'],label=self.G.edges[e]['label'])
                self.H.add_edge(*e)
                for k in self.G.edges[e].keys():
                    self.H.edges[e][k]=self.G.edges[e][k]

        self.list_n=self.H.nodes()
        self.list_e=self.H.edges()
        for n in self.list_n:
            # self.H.nodes[n]['pos']=self.G.nodes[n]['pos']
            # self.H.nodes[n]['source']=self.G.nodes[n]['source']
            # self.H.nodes[n]['label']=self.G.nodes[n]['label']
            for k in self.G.nodes[n].keys():
                self.H.nodes[n][k]=self.G.nodes[n][k]
            self.H_J.append(self.G.nodes[n]['source'])
        for e in self.list_e:
            self.H_C.append(self.H.edges[e]['conductivity'])
        self.H_C=np.asarray(self.H_C)
        self.H_J=np.asarray(self.H_J)
        if len(list(self.H.nodes()))==0:
            sys.exit('FAILED PRUNING')

    def translate_weight(self,attribute):

        for e in self.G.edges():
            self.G.edges[e]['weight']=self.G.edges[e][attribute]
        for e in self.H.edges():
            self.H.edges[e]['weight']=self.H.edges[e][attribute]

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

    # extra tools, defining boundaries & initialization for solute transport networks
    def refine_edges(self,periods):

        self.epsilon=1./(periods-1)
        AUX=nx.Graph(self.G)
        T=np.linspace(0.,1.,num=periods)
        for e in self.G.edges():

            new_node_set=[e[0]]
            for i,t in enumerate(T[1:-1]):
                new_node_set.append(self.count_n())
                pos_new=self.G.nodes[e[0]]['pos']+t*np.subtract(self.G.nodes[e[1]]['pos'],self.G.nodes[e[0]]['pos'])
                AUX.add_node( new_node_set[-1],pos=pos_new,label=new_node_set[-1])
            new_node_set.append(e[1])
            for i,t in enumerate(T[:-1]):
                AUX.add_edge(new_node_set[i],new_node_set[i+1],slope=(AUX.nodes[new_node_set[i]]['pos'],AUX.nodes[new_node_set[i+1]]['pos']),label=self.count_e())

            AUX.remove_edge(*e)
        self.G=nx.Graph()
        dict_nodes={}
        for idx_n,n in enumerate(AUX.nodes()):
            self.G.add_node(idx_n,pos=AUX.nodes[n]['pos'],label=AUX.nodes[n]['label'])
            dict_nodes.update({n:idx_n})
        for idx_e,e in enumerate(AUX.edges()):
            self.G.add_edge(dict_nodes[e[0]],dict_nodes[e[1]],slope=(AUX.nodes[e[0]]['pos'],AUX.nodes[e[1]]['pos']),label=AUX.edges[e]['label'])
        # initialze circuit
        self.initialize_circuit()

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
