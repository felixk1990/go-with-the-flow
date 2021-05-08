import networkx as nx
import numpy as np
import scipy.spatial as ssp
import scipy.linalg as lina
import sys
import init_graph
import init_graph_crystal
import random as rd

class bilayer_graph(init_graph.kirchhoff_network,object):

    def __init__(self):

        super(bilayer_graph,self).__init__()
        self.layer=[]
        self.e_adj=[]
        self.e_adj_idx=[]
        self.n_adj=[]
        self.num_sources=1

    def distance_edges(self):

        self.D=np.zeros(len(self.e_adj_idx))
        for i,e in enumerate(self.e_adj_idx):
            n=self.layer[0].G.edges[e[0]]['slope'][0]-self.layer[0].G.edges[e[0]]['slope'][1]
            m=self.layer[1].G.edges[e[1]]['slope'][0]-self.layer[1].G.edges[e[1]]['slope'][1]
            q=np.cross(n,m)
            q/=np.linalg.norm(q)
            d=(self.layer[0].G.edges[e[0]]['slope'][0]-self.layer[1].G.edges[e[1]]['slope'][0])
            self.D[i]= np.linalg.norm(np.dot(d,q))

        self.D/=((self.layer[0].l+self.layer[1].l)/2.)

    def periodic_cell_structure(self,cell,num_periods,lattice_constant,translation_length):
        C=init_graph_crystal.crystal_graph()
        L=nx.Graph()
        periods=range(num_periods)
        for i in periods:
            for j in periods:
                for k in periods:
                    TD=C.lattice_translation(translation_length*np.array([i,j,k]),cell)
                    L.add_nodes_from(TD.nodes(data=True))
        return L

    def dual_flow_network_crystal(self,unit_cell,translation_length,lattice_constant,num_periods):

        # create primary point cloud with lattice structure
        G=self.periodic_cell_structure(unit_cell,num_periods,lattice_constant,translation_length)
        points=[G.nodes[n]['pos'] for i,n in enumerate(G.nodes())]

        # creating voronoi cells, with defined ridge structure
        V = ssp.Voronoi(points)
        list_p=np.array(list(V.points))
        list_v=np.array(list(V.vertices))

        # construct caged networks from given point clouds, with corresponding adjacency list of edges
        H=nx.Graph()
        G=nx.Graph()
        rv_aux=[]
        rp_aux=[]
        adj=[]
        aff=[{},{}]

        for rv,rp in zip(V.ridge_vertices,V.ridge_points):
            if np.any(np.array(rv)==-1):
                continue
            else:
                rv_aux.append(rv)
                rp_aux.append(rp)

        for j,v in enumerate(list_v):
            H.add_node(j,pos=v,label=self.count_n())

        self.counter_n=0
        for j,p in enumerate(list_p):
            G.add_node(j,pos=p,label=self.count_n())
        for i,n in enumerate(list_p[:-1]):
            for j,m in enumerate(list_p[(i+1):]):
                dist=np.linalg.norm(n-m)
                if dist==lattice_constant:
                    G.add_edge(i,(i+1)+j,slope=(n,m),label=self.count_e())
        self.counter_e=0
        for i,rv in enumerate(rv_aux):
            E1=(rp_aux[i][0],rp_aux[i][1])
            for j,v in enumerate(rv):
                e1=rv[-1+j]
                e2=rv[-1+(j+1)]
                E2=(e1,e2)
                if not H.has_edge(*E2):
                    H.add_edge(*E2,slope=(V.vertices[e1],V.vertices[e2]),label=self.count_e())
                if G.has_edge(*E1):
                    adj.append([G.edges[E1]['label'],H.edges[E2]['label']])

        # cut off redundant (non-connected or neigborless) points/edges
        K=[G,H]
        for i in range(2):
            adj_x=np.array(adj)[:,i]
            list_e=list(K[i].edges())
            for e in list_e:
                if np.any( np.array(adj_x) == K[i].edges[e]['label']):
                    continue
                else:
                    K[i].remove_edge(*e)

            list_n=list(K[i].nodes())
            for n in list_n:
                if not K[i].degree(n)> 0:
                    K[i].remove_node(n)

        # relabeling network nodes/edges & adjacency-list
        P=[nx.Graph(),nx.Graph()]
        dict_P=[[{},{},{}],[{},{},{}]]
        for i in range(2):
            for idx_n,n in enumerate(K[i].nodes()):
                P[i].add_node(idx_n,pos=K[i].nodes[n]['pos'],label=K[i].nodes[n]['label'])
                dict_P[i][0].update({n:idx_n})
                aff[i][idx_n]=[]
            for idx_e,e in enumerate(K[i].edges()):
                P[i].add_edge(dict_P[i][0][e[0]],dict_P[i][0][e[1]],slope=[K[i].nodes[e[0]]['pos'],K[i].nodes[e[1]]['pos']],label=K[i].edges[e]['label'])
            for j,e in enumerate(P[i].edges()):
                dict_P[i][1].update({P[i].edges[e]['label']:j})
                dict_P[i][2].update({P[i].edges[e]['label']:e})

        for a in adj:

            e=[dict_P[0][1][a[0]],dict_P[1][1][a[1]]]
            E=[dict_P[0][2][a[0]],dict_P[1][2][a[1]]]
            self.e_adj.append([e[0],e[1]])
            self.e_adj_idx.append([E[0],E[1]])
            for i in range(2):
                n0=E[i][0]
                n1=E[i][1]
                aff[i][n0].append(a[-(i+1)])
                aff[i][n1].append(a[-(i+1)])

        for i in range(2):

            for key in aff[i].keys():
                aff[i][key]=list(set(aff[i][key]))
                aux=[]
                for l in aff[i][key]:
                    aux.append(dict_P[-(i+1)][1][l])
                aff[i][key]=aux

        self.n_adj=[aff[0],aff[1]]

        return P[0],P[1]

    def construct_dual_networks_crystal(self,bilayer_mode,num_periods,scale,graph_mode,parameters):

        self.layer=[]
        G=[]
        a=1.
        if 'diamond' in bilayer_mode:
            a=0.5
            lattice_constant=np.sqrt(3.)/2.
            translation_length=2
            C=init_graph_crystal.crystal_graph()
            unit_cell=C.diamond_unit_cell()
            g1,g2=self.dual_flow_network_crystal(unit_cell,translation_length,lattice_constant,num_periods)
            G=[g1,g2]
        elif 'bcc' in bilayer_mode:
            lattice_constant=np.sqrt(3.)/2.
            translation_length=1
            C=init_graph_crystal.crystal_graph()
            unit_cell=C.bcc_unit_cell()
            g1,g2=self.dual_flow_network_crystal(unit_cell,translation_length,lattice_constant,num_periods)
            G=[g1,g2]
        elif 'fcc' in bilayer_mode:
            lattice_constant=np.sqrt(2.)/2.
            translation_length=1
            C=init_graph_crystal.crystal_graph()
            unit_cell=C.fcc_unit_cell()
            g1,g2=self.dual_flow_network_crystal(unit_cell,translation_length,lattice_constant,num_periods)
            G=[g1,g2]
        elif 'simple' in bilayer_mode:
            lattice_constant=1
            translation_length=1
            C=init_graph_crystal.crystal_graph()
            unit_cell=C.simple_unit_cell()
            g1,g2=self.dual_flow_network_crystal(unit_cell,translation_length,lattice_constant,num_periods)
            G=[g1,g2]

        else:
            sys.exit('bilayer_graph.construct_dual_networks_crystal(): invalid graph mode')

        for i in range(2):
            self.layer.append( init_graph.kirchhoff_network())
            self.layer[i].G=G[i]
            self.layer[i].l=parameters[0]
            self.layer[i].k=parameters[1]
            self.layer[i].f=parameters[2]
            for n in self.layer[i].G.nodes():
                self.layer[i].G.nodes[n]['pos']*=(scale*self.layer[i].l*a)

            self.layer[i].initialize_circuit()
            self.layer[i].set_source_landscape(graph_mode[i])
            for j,c in enumerate(self.layer[i].C):
                self.layer[i].C[j]=0.01
                #1./np.power(1.,5)
            self.layer[i].threshold=0.0000001
        self.distance_edges()

    def check_no_overlap(self,scale):

        check=True
        K1=self.layer[0]
        K2=self.layer[1]

        for e in self.e_adj:
            r1=K1.C[e[0],e[0]]
            r2=K2.C[e[1],e[1]]

            if r1+r2 > scale*0.5:
                check=False
                break
        return check

    def clipp_graph(self):
        for i in range(2):
            self.layer[i].clipp_graph()

    def set_structure_coefficient_coupling(self,exp):

        f={}
        K=[self.layer[0],self.layer[1]]
        C=np.array([K[0].C[:],K[1].C[:]])
        R=np.array([np.power(C[i],0.25) for i,c in enumerate(C)])
        for j in range(2):
            f[j]=np.zeros(nx.number_of_edges(K[j].G))
        for j,e in enumerate(self.e_adj):
            r=1.-(R[0][e[0]]+R[1][e[1]])
            d0=r**exp

            f[0][e[0]]+=d0
            f[1][e[1]]+=d0
        for i in range(2):
            for j,e in enumerate(K[i].G.edges()):
                K[i].G.edges[e]['coefficient_l1']=np.sign(exp)*f[i][j]/R[i][j]
                if K[i].H.has_edge(*e):
                    K[i].H.edges[e]['coefficient_l1']=K[i].G.edges[e]['coefficient_l1']

    def randomize_plexus(self):
        d=1
        for i in range(2):
            for j,c in enumerate(self.layer[i].C):
                d=self.layer[i].C[j]
                x=int(0.5+rd.random())
                sign=(-1)**x

                self.layer[i].C[j]+=sign*d*rd.random()*0.5

    # test new minimal surface graph_sets
    def dual_minimal_surface_graphs(self,bilayer_mode,num_periods,scale,graph_mode,parameters):

        self.layer=[]
        G=[]
        a=1.
        if 'diamond' in bilayer_mode:
            a=0.5
            lattice_constant=np.sqrt(3.)/2.
            translation_length=1
            C=init_graph_crystal.crystal_graph()
            unit_cell=C.diamond_unit_cell()
            g1,g2=self.dual_diamond(unit_cell,translation_length,lattice_constant,num_periods)
            G=[g1,g2]

        elif 'simple' in bilayer_mode:
            lattice_constant=1
            translation_length=1
            C=init_graph_crystal.crystal_graph()
            unit_cell=C.simple_unit_cell()
            g1,g2=self.dual_flow_network_crystal(unit_cell,translation_length,lattice_constant,num_periods)
            G=[g1,g2]

        elif 'laves' in bilayer_mode:
            lattice_constant=2.
            g1,g2=self.dual_laves(lattice_constant,num_periods)
            G=[g1,g2]

        else:
            sys.exit('bilayer_graph.construct_dual_networks_crystal(): invalid graph mode')

        for i in range(2):
            self.layer.append( init_graph.kirchhoff_network())
            self.layer[i].G=G[i]
            self.layer[i].num_sources=self.num_sources
            self.layer[i].l=parameters[0]
            self.layer[i].k=parameters[1]
            self.layer[i].f=parameters[2]
            self.layer[i].initialize_circuit()
            self.layer[i].set_source_landscape(graph_mode[i])
            for j,c in enumerate(self.layer[i].C):
                self.layer[i].C[j]=0.01
                #1./np.power(1.,5)
            self.layer[i].threshold=0.0000001
        self.distance_edges()

    def dual_diamond(self,unit_cell,translation_length,lattice_constant,num_periods):

        # create primary point cloud with lattice structure
        adj=[]
        adj_idx=[]
        aff=[{},{}]

        G_aux=self.periodic_cell_structure_offset(unit_cell,num_periods,lattice_constant,translation_length,[0,0,0])
        H_aux=self.periodic_cell_structure_offset(unit_cell,num_periods,lattice_constant,translation_length,[1,0,0])
        H=nx.Graph()
        G=nx.Graph()
        points_G=[G_aux.nodes[n]['pos'] for i,n in enumerate(G_aux.nodes())]
        points_H=[H_aux.nodes[n]['pos'] for i,n in enumerate(H_aux.nodes())]
        for i,n in enumerate(G_aux.nodes()):
            G.add_node(i,pos=G_aux.nodes[n]['pos'],label=self.count_n() )
        for i,n in enumerate(points_G[:-1]):
            for j,m in enumerate(points_G[(i+1):]):
                dist=np.linalg.norm(np.subtract(n,m))
                if dist==lattice_constant:
                    G.add_edge(i,(i+1)+j,slope=(n,m),label=self.count_e())

        self.counter_e=0
        self.counter_n=0
        for i,n in enumerate(H_aux.nodes()):
            H.add_node(i,pos=H_aux.nodes[n]['pos'],label=self.count_n() )
        for i,n in enumerate(points_H[:-1]):
            for j,m in enumerate(points_H[(i+1):]):
                dist=np.linalg.norm(np.subtract(n,m))
                if dist==lattice_constant:
                    H.add_edge(i,(i+1)+j,slope=(n,m),label=self.count_e())

        for i,e in enumerate(G.edges()):
            a=np.add(G.edges[e]['slope'][0],G.edges[e]['slope'][1])
            for j,f in enumerate(H.edges()):
                b=np.add(H.edges[f]['slope'][0],H.edges[f]['slope'][1])
                c=np.subtract(a,b)
                if np.dot(c,c)==2.:
                    adj.append([G.edges[e]['label'],H.edges[f]['label']])
                    adj_idx.append([e,f])
        K=[G,H]
        for i in range(2):
            adj_x=np.array(adj)[:,i]
            list_e=list(K[i].edges())
            for e in list_e:
                if np.any( np.array(adj_x) == K[i].edges[e]['label']):
                    continue
                else:
                    K[i].remove_edge(*e)

            list_n=list(K[i].nodes())
            for n in list_n:
                if not K[i].degree(n)> 0:
                    K[i].remove_node(n)

        P=[nx.Graph(),nx.Graph()]
        dict_P=[[{},{},{}],[{},{},{}]]
        for i in range(2):
            for idx_n,n in enumerate(K[i].nodes()):
                P[i].add_node(idx_n,pos=K[i].nodes[n]['pos'],label=K[i].nodes[n]['label'])
                dict_P[i][0].update({n:idx_n})
                aff[i][idx_n]=[]
            for idx_e,e in enumerate(K[i].edges()):
                P[i].add_edge(dict_P[i][0][e[0]],dict_P[i][0][e[1]],slope=[K[i].nodes[e[0]]['pos'],K[i].nodes[e[1]]['pos']],label=K[i].edges[e]['label'])
            for j,e in enumerate(P[i].edges()):
                dict_P[i][1].update({P[i].edges[e]['label']:j})
                dict_P[i][2].update({P[i].edges[e]['label']:e})

        for a in adj:

            e=[dict_P[0][1][a[0]],dict_P[1][1][a[1]]]
            E=[dict_P[0][2][a[0]],dict_P[1][2][a[1]]]
            self.e_adj.append([e[0],e[1]])
            self.e_adj_idx.append([E[0],E[1]])
            for i in range(2):
                n0=E[i][0]
                n1=E[i][1]
                aff[i][n0].append(a[-(i+1)])
                aff[i][n1].append(a[-(i+1)])

        for i in range(2):

            for key in aff[i].keys():
                aff[i][key]=list(set(aff[i][key]))
                aux=[]
                for l in aff[i][key]:
                    aux.append(dict_P[-(i+1)][1][l])
                aff[i][key]=aux

        self.n_adj=[aff[0],aff[1]]


        return P[0],P[1]

    def dual_laves(self,lattice_constant,num_periods):


        adj=[]
        adj_idx=[]
        aff=[{},{}]

        G_aux=self.laves_graph(num_periods,'R',[0.,0.,0.])
        list_nodes=list(G_aux.nodes())
        H=nx.Graph()
        G=nx.Graph()
        points_G=[G_aux.nodes[n]['pos'] for i,n in enumerate(G_aux.nodes()) ]
        for i,n in enumerate(G_aux.nodes()) :
            # if n in largest_cc:
                G.add_node(n,pos=G_aux.nodes[n]['pos'],label=self.count_n() )
        for i,n in enumerate(list_nodes[:-1]):
            # if n in largest_cc:
                for j,m in enumerate(list_nodes[(i+1):]):
                    # if m in largest_cc:
                        v=np.subtract(n,m)
                        dist=np.dot(v,v)
                        if dist==lattice_constant:
                            G.add_edge(n,m,slope=(G_aux.nodes[n]['pos'],G_aux.nodes[m]['pos']),label=self.count_e())

        self.counter_e=0
        self.counter_n=0
        H_aux=self.laves_graph(num_periods,'L',[3.,2.,0.])
        list_nodes=list(H_aux.nodes())
        points_H=[H_aux.nodes[n]['pos'] for i,n in enumerate(H_aux.nodes())]
        for i,n in enumerate(H_aux.nodes()):
            H.add_node(n,pos=H_aux.nodes[n]['pos'],label=self.count_n() )
        for i,n in enumerate(list_nodes[:-1]):
            for j,m in enumerate(list_nodes[(i+1):]):
                v=np.subtract(n,m)
                dist=np.dot(v,v)
                if dist==lattice_constant:
                    H.add_edge(n,m,slope=(H_aux.nodes[n]['pos'],H_aux.nodes[m]['pos']),label=self.count_e())

        for i,e in enumerate(G.edges()):
            a=np.add(G.edges[e]['slope'][0],G.edges[e]['slope'][1])
            for j,f in enumerate(H.edges()):
                b=np.add(H.edges[f]['slope'][0],H.edges[f]['slope'][1])
                c=np.subtract(a,b)
                if np.dot(c,c)==14.:
                    adj.append([G.edges[e]['label'],H.edges[f]['label']])
                    adj_idx.append([e,f])
        K=[G,H]

        for i in range(2):
            adj_x=np.array(adj)[:,i]
            list_e=list(K[i].edges())
            for e in list_e:
                if np.any( np.array(adj_x) == K[i].edges[e]['label']):
                    continue
                else:
                    K[i].remove_edge(*e)

            list_n=list(K[i].nodes())
            for n in list_n:
                if not K[i].degree(n)> 0:
                    K[i].remove_node(n)

        P=[nx.Graph(),nx.Graph()]
        dict_P=[[{},{},{}],[{},{},{}]]
        for i in range(2):
            for idx_n,n in enumerate(K[i].nodes()):
                P[i].add_node(idx_n,pos=K[i].nodes[n]['pos'],label=K[i].nodes[n]['label'])
                dict_P[i][0].update({n:idx_n})
                aff[i][idx_n]=[]
            for idx_e,e in enumerate(K[i].edges()):
                P[i].add_edge(dict_P[i][0][e[0]],dict_P[i][0][e[1]],slope=[K[i].nodes[e[0]]['pos'],K[i].nodes[e[1]]['pos']],label=K[i].edges[e]['label'])
            for j,e in enumerate(P[i].edges()):
                dict_P[i][1].update({P[i].edges[e]['label']:j})
                dict_P[i][2].update({P[i].edges[e]['label']:e})

        for a in adj:

            e=[dict_P[0][1][a[0]],dict_P[1][1][a[1]]]
            E=[dict_P[0][2][a[0]],dict_P[1][2][a[1]]]
            self.e_adj.append([e[0],e[1]])
            self.e_adj_idx.append([E[0],E[1]])
            for i in range(2):
                n0=E[i][0]
                n1=E[i][1]
                aff[i][n0].append(a[-(i+1)])
                aff[i][n1].append(a[-(i+1)])

        for i in range(2):

            for key in aff[i].keys():
                aff[i][key]=list(set(aff[i][key]))
                aux=[]
                for l in aff[i][key]:
                    aux.append(dict_P[-(i+1)][1][l])
                aff[i][key]=aux

        self.n_adj=[aff[0],aff[1]]


        return P[0],P[1]

    def periodic_cell_structure_offset(self,cell,num_periods,lattice_constant,translation_length,offset):
        C=init_graph_crystal.crystal_graph()
        L=nx.Graph()
        periods=range(num_periods)
        for i in periods:
            for j in periods:
                for k in periods:
                    if (i+j+k)%2==0:
                        TD=C.lattice_translation(offset+translation_length*np.array([i,j,k]),cell)
                        L.add_nodes_from(TD.nodes(data=True))

        return L

    def laves_graph(self,num_periods,chirality,offset):
        counter=0
        L=nx.Graph()
        periods=range(num_periods)
        # fundamental_points=[[0,0,0],[1,2,3],[2,3,1],[3,1,2],[2,2,2],[3,0,1],[0,1,3],[1,3,0]]
        # for l,fp in enumerate(fundamental_points):
        #     for i in periods:
        #         for j in periods:
        #             for k in periods:
        #
        #                 pos_n=np.array(fp)
        #                 if i!=0 or j!=0 or k!=0:
        #                     pos_n=np.add(pos_n,[2.+4*i,2.+4*j,2.+4*k])
        #                     L.add_node(tuple(pos_n),pos=pos_n)
        #                 else:
        #                     L.add_node(tuple(pos_n),pos=pos_n)
        fundamental_points=[[0,0,0],[1,1,0],[1,2,1],[0,3,1],[2,2,2],[3,3,2],[3,0,3],[2,1,3]]
        if chirality=='R':

            for l,fp in enumerate(fundamental_points):
                for i in periods:
                    for j in periods:
                        for k in periods:

                            pos_n=np.add(np.add(fp,[4.*i,4.*j,4.*k]),offset)
                            L.add_node(tuple(pos_n),pos=pos_n)
        if chirality=='L':
            # fundamental_points=[np.add(fp,[2.,0.,0.]) for fp in fundamental_points]
            for l,fp in enumerate(fundamental_points):
                for i in periods:
                    for j in periods:
                        for k in periods:

                            pos_n=np.add(np.add(np.multiply(fp,[-1.,1.,1.]),[4.*i,4.*j,4.*k]),offset)
                            L.add_node(tuple(pos_n),pos=pos_n)
            # for n in L.nodes():
            #     L.nodes[n]['pos']+=np.add(np.array([1.,1.,1.])*4.*num_periods,[-2.,0.,0.])

        return L
