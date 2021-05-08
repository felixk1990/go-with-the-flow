import networkx as nx
import numpy as np
import sys
import init_graph
import analyze_graph

class hex_graph(init_graph.kirchhoff_network,object):

    def __init__(self):
        super(hex_graph,self).__init__()

    #I) construct and define one-layer hex
    # auxillary function, construct triangulated hex grid upper and lower wings
    def construct_wing(self,a,n):

      for m in range(n-1):
          #m-th floor
          floor_m_nodes = self.spine - (m+1)
          self.G.add_node((0,a*(m+1)),pos=np.array([self.l*(m+1)/2.,a*(np.sqrt(3.)/2.)*self.l*(m+1)]),label=self.count_n())
          self.G.add_edge((0,a*(m+1)),(0,a*m),slope=(self.G.nodes[(0,a*(m+1))]['pos'],self.G.nodes[(0,a*m)]['pos']),label=self.count_e())
          self.G.add_edge((0,a*(m+1)),(1,a*m),slope=(self.G.nodes[(0,a*(m+1))]['pos'],self.G.nodes[(1,a*m)]['pos']),label=self.count_e())

          for p in range(floor_m_nodes):
              #add 3-junctions
              self.G.add_node((p+1,a*(m+1)),pos=np.array([self.l*((p+1)+(m+1)/2.),a*(np.sqrt(3.)/2.)*self.l*(m+1)]),label=self.count_n())
              self.G.add_edge((p+1,a*(m+1)),(p+1,a*m),slope=(self.G.nodes[(p+1,a*(m+1))]['pos'],self.G.nodes[(p+1,a*m)]['pos']),label=self.count_e())
              self.G.add_edge((p+1,a*(m+1)),(p+2,a*m),slope=(self.G.nodes[(p+1,a*(m+1))]['pos'],self.G.nodes[(p+2,a*m)]['pos']),label=self.count_e())
              self.G.add_edge((p+1,a*(m+1)),(p,a*(m+1)),slope=(self.G.nodes[(p+1,a*(m+1))]['pos'],self.G.nodes[(p,a*(m+1))]['pos']),label=self.count_e())

    #construct full triangulated hex grid as skeleton
    def triangulated_hexagon_lattice(self,n, sidelength, conductance,flow):

      self.k=conductance
      self.l=sidelength
      self.f=flow
      #construct spine
      self.spine = 2*(n-1)
      self.G.add_node((0,0),pos=np.array([0.,0.]), label=self.count_n())

      for m in range(self.spine):

          self.G.add_node((m+1,0),pos=np.array([(m+1)*self.l,0.]),label=self.count_n())
          self.G.add_edge((m,0),(m+1,0),slope=(self.G.nodes[(m,0)]['pos'],self.G.nodes[(m+1,0)]['pos']),label=self.count_e())

      #construct lower/upper halfspace
      self.construct_wing(-1,n)
      self.construct_wing( 1,n)

      # initialze circuit
      H=nx.Graph(self.G)
      self.G=nx.Graph()
      dict_nodes={}
      for idx_n,n in enumerate(H.nodes()):
          self.G.add_node(idx_n,pos=H.nodes[n]['pos'],label=H.nodes[n]['label'])
          dict_nodes.update({n:idx_n})
      for idx_e,e in enumerate(H.edges()):
          self.G.add_edge(dict_nodes[e[0]],dict_nodes[e[1]],slope=(H.nodes[e[0]]['pos'],H.nodes[e[1]]['pos']),label=H.edges[e]['label'])
      self.initialize_circuit()


    #II) construct and define multi-layer hex

    #define crosslinking procedure between the generated single-layers
    def crosslink_stacks(self):

      if self.stacks > 1 :
          labels_n = nx.get_node_attributes(self.G,'label')
          sorted_label_n_list=sorted(labels_n ,key=labels_n.__getitem__)
          # for n in nx.nodes(self.G):
          for n in sorted_label_n_list:
              if n[2]!=self.stacks-1:

                  self.G.add_edge((n[0],n[1],n[2]),(n[0],n[1],n[2]+1),slope=(self.G.nodes[(n[0],n[1],n[2])]['pos'],self.G.nodes[(n[0],n[1],n[2]+1)]['pos']),label=self.count_e())

    # auxillary function, construct triangulated hex grid upper and lower wings
    def construct_spine_stack(self,z,n):
        self.spine = 2*(n-1)
        # self.spine=2*n
        self.G.add_node((0,0,z),pos=(0.,0.,z*self.l),label=self.count_n())

      # for m in range(self.spine-1):
        for m in range(self.spine):

            self.G.add_node((m+1,0,z),pos=((m+1)*self.l,0.,z*self.l),label=self.count_n())
            self.G.add_edge((m,0,z),(m+1,0,z),slope=(self.G.nodes[(m,0,z)]['pos'],self.G.nodes[(m+1,0,z)]['pos']),label=self.count_e())

    def construct_wing_stack(self,z,a,n):
        for m in range(n-1):
            #m-th floor
            floor_m_nodes=self.spine-(m+1)
      # for m in range(n):
      #     #m-th floor
      #     floor_m_nodes=self.spine-(m+2)
            self.G.add_node((0,a*(m+1),z),pos=(self.l*(m+1)/2.,a*(np.sqrt(3.)/2.)*self.l*(m+1),z*self.l),label=self.count_n())
            self.G.add_edge((0,a*(m+1),z),(0,a*m,z),slope=(self.G.nodes[(0,a*(m+1),z)]['pos'],self.G.nodes[(0,a*m,z)]['pos']),label=self.count_e())
            self.G.add_edge((0,a*(m+1),z),(1,a*m,z),slope=(self.G.nodes[(0,a*(m+1),z)]['pos'],self.G.nodes[(1,a*m,z)]['pos']),label=self.count_e())

            for p in range(floor_m_nodes):
              #add 3-junctions
                self.G.add_node((p+1,a*(m+1),z),pos=(self.l*((p+1)+(m+1)/2.),a*(np.sqrt(3.)/2.)*self.l*(m+1),z*self.l),label=self.count_n())
                self.G.add_edge((p+1,a*(m+1),z),(p+1,a*m,z),slope=(self.G.nodes[(p+1,a*(m+1),z)]['pos'],self.G.nodes[(p+1,a*m,z)]['pos']),label=self.count_e())
                self.G.add_edge((p+1,a*(m+1),z),(p+2,a*m,z),slope=(self.G.nodes[(p+1,a*(m+1),z)]['pos'],self.G.nodes[(p+2,a*m,z)]['pos']),label=self.count_e())
                self.G.add_edge((p+1,a*(m+1),z),(p,a*(m+1),z),slope=(self.G.nodes[(p+1,a*(m+1),z)]['pos'],self.G.nodes[(p,a*(m+1),z)]['pos']),label=self.count_e())

    #construct full triangulated hex grids as skeleton of a stacked structure
    def triangulated_hexagon_lattice_stack(self,stack,n, sidelength, conductance,flow):

      self.k=conductance
      self.l=sidelength
      self.f=flow
      self.stacks=stack
      for z in range(self.stacks):

          #construct spine for different levels of lobule
          self.construct_spine_stack(z,n)

          #construct lower/upper halfspace
          self.construct_wing_stack( z,-1, n)
          self.construct_wing_stack( z, 1, n)

      self.crosslink_stacks()

      # initialze circuit & add attributes
      self.initialize_circuit()

    def remove_center(self):

        if self.stacks == 1 :
         self.G.remove_node((int((self.spine+1)/2),0))
        elif self.stacks > 1 :
         for z in range(self.stacks):
             self.G.remove_node((int((self.spine+1)/2),0,z))
        else:
         sys.exit('Error, stacks not correctly defined.')
        self.C=np.identity(self.G.number_of_edges())
        self.V=np.zeros(self.G.number_of_nodes())
        self.J=np.zeros(self.G.number_of_nodes())
        self.F=np.zeros(self.G.number_of_edges())

    def square_grid(self, tiling_factor, sidelength, conductance,flow):

        self.k=conductance
        self.l=sidelength
        self.f=flow
        a=range(0,tiling_factor+1)

        for x in a:
            for y in a:
                self.G.add_node((x,y),pos=(x,y,0),label=self.count_n())

        list_n=list(self.G.nodes())
        dict_d={}
        threshold=1.
        for idx_n,n in enumerate(list_n[:-1]):
            for m in list_n[idx_n+1:]:
                dict_d[(n,m)]=np.linalg.norm(np.array(self.G.nodes[n]['pos'])-np.array(self.G.nodes[m]['pos']))
        for nm in dict_d:
            if dict_d[nm] <= threshold:
                self.G.add_edge(*nm,slope=[self.G.nodes[nm[0]]['pos'],self.G.nodes[nm[1]]['pos']],label=self.count_e())

        # initialze circuit & add attributes
        self.initialize_circuit()

class hex_tiles_graph(init_graph.kirchhoff_network,object):

        def __init__(self):
            super(hex_tiles_graph,self).__init__()

        def hexagonal_grid(self, *args):

            tiling_factor, sidelength, conductance,flow,periodic_bool=args
            self.k=conductance
            self.l=sidelength
            self.f=flow

            m=2*tiling_factor+1
            n=2*tiling_factor
            self.G=nx.hexagonal_lattice_graph(m, n, periodic=periodic_bool, with_positions=True)
            for n in self.G.nodes():
                self.G.nodes[n]['label']=self.count_n
                self.G.nodes[n]['pos']=self.l*np.array(self.G.nodes[n]['pos'])
            for e in self.G.edges():

                self.G.edges[e]['label']=self.count_e
                self.G.edges[e]['slope']=[self.G.nodes[e[0]]['pos'],self.G.nodes[e[1]]['pos']]

            # initialze circuit & add attributes
            self.initialize_circuit()

        def set_up_volumes(self,periodic_bool):

            # set up volumes be aware to work on properly iniced systems
            if periodic_bool:
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
                    if len(dict_volumes[k])>6:
                        del dict_volumes[k]
                self.dict_volumes=dict_volumes
            else:
                print('why bother?')
