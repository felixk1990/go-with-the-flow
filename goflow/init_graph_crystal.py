import networkx as nx
import numpy as np
import sys
import init_graph
import analyze_graph

class crystal_graph(init_graph.kirchhoff_network,object):

    # construct one of the following crystal topologies
    def lattice_translation(self,t,T):
        D=nx.Graph()
        for n in T.nodes():
            D.add_node(tuple(n+t),pos=T.nodes[n]['pos']+t)
        return D

    def periodic_cell_structure(self,cell,num_periods,lattice_constant,translation_length):
        DL=nx.Graph()

        if type(num_periods) is not int :
            periods=[range(num_periods[0]),range(num_periods[1]),range(num_periods[2])]
        else:
            periods=[range(num_periods),range(num_periods),range(num_periods)]
        for i in periods[0]:
            for j in periods[1]:
                for k in periods[2]:
                    TD=self.lattice_translation(translation_length*np.array([i,j,k]),cell)
                    DL.add_nodes_from(TD.nodes(data=True))
                    self.dict_cells[(i,j,k)]=list(TD.nodes())
        for n in DL.nodes():
            DL.nodes[n]['label']=self.count_n()
        list_n=np.array(list(DL.nodes()))
        for i,n in enumerate(list_n[:-1]):
            for m in list_n[(i+1):]:
                dist=np.linalg.norm(DL.nodes[tuple(n)]['pos']-DL.nodes[tuple(m)]['pos'])
                if dist==lattice_constant:
                    DL.add_edge(tuple(n),tuple(m),slope=(DL.nodes[tuple(n)]['pos'],DL.nodes[tuple(m)]['pos']),label=self.count_e())

        dict_nodes={}
        for idx_n,n in enumerate(DL.nodes()):
            self.G.add_node(idx_n,pos=self.l*DL.nodes[n]['pos'],label=DL.nodes[n]['label'])
            dict_nodes.update({n:idx_n})
        for idx_e,e in enumerate(DL.edges()):
            self.G.add_edge(dict_nodes[e[0]],dict_nodes[e[1]],slope=(self.l*DL.nodes[e[0]]['pos'],self.l*DL.nodes[e[1]]['pos']),label=DL.edges[e]['label'])

        self.dict_cubes={}
        dict_aux={}
        for i,k in enumerate(self.dict_cells.keys()):
            dict_aux[i]=[ dict_nodes[n] for n in self.dict_cells[k] ]
        for i,k in enumerate(dict_aux.keys()):
            self.dict_cubes[k]=nx.Graph()
            n_list=list(dict_aux[k])
            for u in n_list[:-1]:
                for v in n_list[1:]:
                    if self.G.has_edge(u,v):
                        self.dict_cubes[k].add_edge(u,v)
    #construct full triangulated hex grid as skeleton
    def simple_unit_cell(self):
        D=nx.Graph()
        for i in [0,1]:
            for j in [0,1]:
                for k in [0,1]:
                    D.add_node(tuple((i,j,k)),pos=np.array([i,j,k]))

        return D

    def simple_cubic_lattice(self,num_periods, sidelength, conductance,flow):
        self.dict_cells={  }
        self.k=conductance
        self.l=sidelength
        self.f=flow
        #construct single box
        lattice_constant=1.
        translation_length=1.
        D=self.simple_unit_cell()
        self.periodic_cell_structure(D,num_periods,lattice_constant,translation_length)

        # initialze circuit
        self.initialize_circuit()

    def simple_chain(self,num_periods, sidelength, conductance,flow):

        self.k=conductance
        self.l=sidelength
        self.f=flow
        #construct single box
        lattice_constant=1.
        for i in range(num_periods):
          self.G.add_node(i, pos=self.l*np.array([i,0,0]),label=self.count_n())
        for i in range(num_periods-1):
          self.G.add_edge(i+1,i, label=self.count_e(),slope=(self.l*self.G.nodes[i+1]['pos'],self.l*self.G.nodes[i]['pos']))

        # initialze circuit
        self.initialize_circuit()

    def bcc_unit_cell(self):
        D=nx.Graph()
        for i in [0,1]:
            for j in [0,1]:
                for k in [0,1]:
                    D.add_node(tuple((i,j,k)),pos=np.array([i,j,k]))
        D.add_node(tuple((0.5,0.5,0.5)),pos=np.array([0.5,0.5,0.5]))
        return D

    def simple_bcc_lattice(self,n, sidelength, conductance,flow):

        self.k=conductance
        self.l=sidelength
        self.f=flow
        #construct single box
        lattice_constant=np.sqrt(3.)/2.
        translation_length=1.
        D=self.bcc_unit_cell()
        self.periodic_cell_structure(D,n,lattice_constant,translation_length)
        # initialze circuit
        self.initialize_circuit()

    def fcc_unit_cell(self):
        D=nx.Graph()
        for i in [0,1]:
            for j in [0,1]:
                for k in [0,1]:
                    D.add_node(tuple((i,j,k)),pos=np.array([i,j,k]))
        for i in [0.,1.]:
            D.add_node(tuple((0.5,i,0.5)),pos=np.array([0.5,i,0.5]))
        for i in [0.,1.]:
            D.add_node(tuple((0.5,0.5,i)),pos=np.array([0.5,0.5,i]))
        for i in [0.,1.]:
            D.add_node(tuple((i,0.5,0.5)),pos=np.array([i,0.5,0.5]))

        return D

    def simple_fcc_lattice(self,n, sidelength, conductance,flow):

        self.k=conductance
        self.l=sidelength
        self.f=flow
        #construct spine
        lattice_constant=np.sqrt(2.)/2.
        translation_length=1.
        D=self.fcc_unit_cell()
        self.periodic_cell_structure(D,n,lattice_constant,translation_length)

        # initialze circuit
        self.initialize_circuit()

    def diamond_unit_cell(self):

        D=nx.Graph()
        T=[nx.Graph() for i in range(4)]
        T[0].add_node((0,0,0),pos=np.array([0,0,0]))
        T[0].add_node((1,1,0),pos=np.array([1,1,0]))
        T[0].add_node((1,0,1),pos=np.array([1,0,1]))
        T[0].add_node((0,1,1),pos=np.array([0,1,1]))
        T[0].add_node((0.5,0.5,0.5),pos=np.array([0.5,0.5,0.5]))
        translation=[np.array([1,1,0]),np.array([1,0,1]),np.array([0,1,1])]
        for i,t in enumerate(translation):
            for n in T[0].nodes():
                T[i+1].add_node(tuple(n+t),pos=T[0].nodes[n]['pos']+t)
        for t in T:
            D.add_nodes_from(t.nodes(data=True))

        return D

    def diamond_lattice(self,num_periods, sidelength, conductance,flow):

        self.dict_cells={  }
        self.k=conductance
        self.l=sidelength
        self.f=flow

        lattice_constant=np.sqrt(3.)/2.
        translation_length=2.
        D=self.diamond_unit_cell()
        self.periodic_cell_structure(D,num_periods,lattice_constant,translation_length)

         # initialze circuit
        self.initialize_circuit()

    def laves_lattice(self,num_periods, sidelength, conductance,flow):

        self.k=conductance
        self.l=sidelength
        self.f=flow
        #construct single box
        counter=0
        G_aux=nx.Graph()
        # periods=range(-num_periods,num_periods)
        # periods=range(num_periods)
        if type(num_periods) is not int :
            periods=[range(num_periods[0]),range(num_periods[1]),range(num_periods[2])]
        else:
            periods=[range(num_periods),range(num_periods),range(num_periods)]
        lattice_constant=2.
        fundamental_points=[[0,0,0],[1,1,0],[1,2,1],[0,3,1],[2,2,2],[3,3,2],[3,0,3],[2,1,3]]
        for l,fp in enumerate(fundamental_points):
            for i in periods[0]:
                for j in periods[1]:
                    for k in periods[2]:

                        pos_n=np.add(fp,[4.*i,4.*j,4.*k])
                        G_aux.add_node(tuple(pos_n),pos=pos_n)



        list_nodes=list(G_aux.nodes())
        self.G=nx.Graph()
        H=nx.Graph()
        points_G=[G_aux.nodes[n]['pos'] for i,n in enumerate(G_aux.nodes()) ]
        for i,n in enumerate(G_aux.nodes()) :

              H.add_node(n,pos=G_aux.nodes[n]['pos'],label=self.count_n() )
        for i,n in enumerate(list_nodes[:-1]):
              for j,m in enumerate(list_nodes[(i+1):]):

                      v=np.subtract(n,m)
                      dist=np.dot(v,v)
                      if dist==lattice_constant:
                          H.add_edge(n,m,slope=(G_aux.nodes[n]['pos'],G_aux.nodes[m]['pos']),label=self.count_e())

        dict_nodes={}
        for idx_n,n in enumerate(H.nodes()):
          self.G.add_node(idx_n,pos=self.l*H.nodes[n]['pos'],label=H.nodes[n]['label'])
          dict_nodes.update({n:idx_n})
        for idx_e,e in enumerate(H.edges()):
          self.G.add_edge(dict_nodes[e[0]],dict_nodes[e[1]],slope=(self.l*H.nodes[e[0]]['pos'],self.l*H.nodes[e[1]]['pos']),label=H.edges[e]['label'])
        # initialze circuit
        self.initialize_circuit()

    def laves_set_up_volumes(self,periodic_bool):

        # set up volumes be aware to work on properly iniced systems
        if periodic_bool:
            XYZ=[[],[],[]]
            min_xi=[]
            max_xi=[]
            pos = nx.get_node_attributes(self.G, 'pos')
            for p in pos.values():
                for i,xi in enumerate(p):

                    XYZ[i].append(xi)
            for i in range(3):
                min_xi.append(np.amin(XYZ[i]))
                max_xi.append(np.amax(XYZ[i]))

            list_dangeling=[]
            for n in self.G.nodes():
                if self.G.degree(n)<3:
                    list_dangeling.append(n)

            for n in list_dangeling:
                p= self.G.nodes[n]['pos']
                for i,xi in enumerate(p):
                    if xi == min_xi[i]:
                        dxi=np.zeros(3)
                        dxi[i]=max_xi[i]+self.l
                        q=np.add(p,dxi)
                        for m in list_dangeling:
                            p_test= self.G.nodes[m]['pos']
                            dist=np.linalg.norm(np.subtract(p_test,q))

                            if dist==np.sqrt(2.)*self.l:
                                self.G.add_edge(n,m,slope=(self.G.nodes[n]['pos'],self.G.nodes[m]['pos']),label=self.count_e())


            list_dangeling=[]
            for n in self.G.nodes():
                if self.G.degree(n)<3:
                    list_dangeling.append(n)
            for n in list_dangeling:
                p= self.G.nodes[n]['pos']
                for i,xi in enumerate(p):
                    diff_probe_idx=[j for j in range(3) if j!=i ]
                    if xi == min_xi[i]:
                        dxi=np.zeros(3)
                        dxi[i]=max_xi[i]+self.l
                        for j in diff_probe_idx:
                            dxi[j]=max_xi[j]+self.l
                            q=np.add(p,dxi)
                            for m in list_dangeling:
                                p_test= self.G.nodes[m]['pos']
                                dist=np.linalg.norm(np.subtract(p_test,q))

                                if dist==np.sqrt(2.)*self.l:
                                    self.G.add_edge(n,m,slope=(self.G.nodes[n]['pos'],self.G.nodes[m]['pos']),label=self.count_e())




            T=analyze_graph.tool_box()
            cycle_basis=T.construct_minimum_basis(self.G)
            dict_volumes={}
            dict_idx={}
            dict_counter={}
            for i,e in enumerate(self.G.edges()):
                dict_idx[e]=i
                dict_counter[e]=0
            for i,c in enumerate(cycle_basis):
                    dict_volumes[i]=[]
                    for j,e in enumerate( c.edges() ):

                        if e in dict_idx:
                            dict_volumes[i].append(dict_idx[e])
                            dict_counter[e]+=1
                        else:
                            dict_volumes[i].append(dict_idx[(e[-1],e[0])])
                            dict_counter[(e[-1],e[0])]+=1
            dict_volumes[len(cycle_basis)]=[]
            for i,e in enumerate(self.G.edges()):
                if dict_counter[e]<2:
                    dict_volumes[len(cycle_basis)].append(dict_idx[e])

            keys=list(dict_volumes.keys())
            for k in keys:
                if len(dict_volumes[k])!=10:
                    del dict_volumes[k]
            self.dict_volumes=dict_volumes

        else:
            print('why bother?')
