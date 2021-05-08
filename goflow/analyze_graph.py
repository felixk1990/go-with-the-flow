import networkx as nx
import numpy as np
import random as rd
import scipy as sc
import scipy.stats
import os
import os.path as op
import re
import sys
from IPython.display import Image, display
import multiprocessing as mp
class tool_box:

    #internal memory variables
    def __init__(self):
        self.mem_n=[]
        self.mem_e=[]
        self.label=1
        self.num=20
        self.tol=np.power(10.,-8.)

        self.list_n_phy=[]
        self.list_e_phy=[]
        self.list_n_topo=[]
        self.list_e_topo=[]
        self.list_pearson_phy=[]
        self.list_pearson_topo=[]
        self.list_exp_phy=[]
        self.list_exp_topo=[]
        self.list_nullity=[]
        self.list_robustness=[]
        self.list_cycle_tree=[]
        self.list_path=[]
        self.list_scaling=[]
        self.list_delta=[]
        self.list_branching=[]

        self.aux_list1=[]
        self.aux_list2=[]
        self.aux_list3=[]
        self.aux_list4=[]

        self.list_bond=[]
        self.list_site=[]
        self.list_A=[]
        self.list_coverage=[]
        self.list_cycle_density=[]
        self.basis_mode=''
        # self.basis_mode='shortest_topological'
        self.input_dir=''
        self.input_data_dir=''
        self.output_dir=''
        self.output_tag=''

        self.NULLITY=False
        self.ROBUSTNESS=False
        self.RENTIAN_PHYS=False
        self.RENTIAN_TOPO=False
        self.CYCLE_COAL=False
        self.CYCLE_COAL_AVG=False
        self.SCALING=False
        self.PERCOLATION_BOND=False
        self.PERCOLATION_SITE=False
        self.PATH_HIST=False
        self.BRANCHING=False
        self.CYCLE_NEMATIC=False

        self.reference_graph=nx.Graph()

    #setting variables, flags & pathways
    def set_tool(self,setting_dict):
        self.NULLITY=setting_dict['nullity']
        self.ROBUSTNESS=setting_dict['robustness']
        self.RENTIAN_PHYS=setting_dict['rentian_physical']
        self.RENTIAN_TOPO=setting_dict['rentian_topological']
        self.CYCLE_COAL=setting_dict['cycle_coalescence']
        self.CYCLE_COAL_AVG=setting_dict['cycle_coalescence_avg']
        self.SCALING=setting_dict['scaling']
        self.PERCOLATION_BOND=setting_dict['percolation_bond']
        self.PERCOLATION_SITE=setting_dict['percolation_site']
        self.PATH_HIST=setting_dict['path_hist']
        self.BRANCHING=setting_dict['branching']
        self.CYCLE_NEMATIC=setting_dict['nematic']

    def set_path(self,IO):
        self.input_dir=IO.DIR_OUT_PROGRAM_DATA
        self.input_data_dir=IO.DIR_OUT_DATA
        self.output_dir=IO.DIR_OUT_PROGRAM_PLOT

    def set_path_custom(self,path_input,path_output):
        self.input_dir=path_input
        self.output_dir=path_output

    def get_path(self):
        return self.input_dir,self.output_dir

    def analyze_graph(self,G):

        #check active tics
        if self.NULLITY:
            print('...Caculating graph nullity...')
            N=self.calc_nullity(G)
            self.list_nullity.append(N)
            print('...Done...')
        if self.ROBUSTNESS:
            print('...Caculating graph robustness...')
            RG=self.calc_robustness(G)
            MST,GT=self.construct_nullgraphs(G)
            RGT=self.calc_robustness(GT)
            RMST=self.calc_robustness(MST)
            R=(RG-RMST)/(RGT-RMST)
            self.list_robustness.append(R)
            print('...Done...')
        if self.RENTIAN_PHYS:
            print('...Caculating Rent coefficients, physical...')
            n_aux,e_aux,p_aux,exp_aux = self.calc_rentian(G,'physical')

            self.list_n_phy.append(n_aux)
            self.list_e_phy.append(e_aux)

            self.list_pearson_phy.append(p_aux)
            self.list_exp_phy.append(exp_aux)
            print('...Done...')
        if self.RENTIAN_TOPO:
            print('...Caculating Rent coefficients, topological...')
            n_aux,e_aux,p_aux,exp_aux = self.calc_rentian(G,'topological')

            self.list_n_topo.append(n_aux)
            self.list_e_topo.append(e_aux)

            self.list_pearson_topo.append(p_aux)
            self.list_exp_topo.append(exp_aux)
            print('...Done...')
        if self.CYCLE_COAL:
            print('...Caculating graph cycle coalescence...')

            cycle_basis=self.construct_minimum_basis(G)
            CT,A=self.calc_cycle_coalescence(G,cycle_basis)

            self.list_cycle_tree.append(CT)
            self.list_A.append(A)
            print('...Done...')

        if self.CYCLE_COAL_AVG:
            print('...Caculating graph cycle coalescence on average...')

            cycle_basis_list=self.construct_minimum_basis_multi(G,self.num)
            CT,A=self.calc_cycle_coalescence_avg(G,cycle_basis_list)
            self.list_cycle_tree.append(CT)
            self.list_A.append(A)
            print('...Done...')
        if self.PERCOLATION_BOND:
            print('...Caculating graph bond percolation...')
            phi,S,err_phi,err_S=self.calc_percolation_bond(G)
            self.list_bond.append([phi,S,err_phi,err_S])

            print('...Done...')
        if self.PERCOLATION_SITE:
            print('...Caculating graph site percolation...')
            phi,S,err_phi,err_S=self.calc_percolation_site(G)

            self.list_site.append([phi,S,err_phi,err_S])

            print('...Done...')
        if self.PATH_HIST:
            print('...Caculating graph path length distribution ...')
            root=0
            try:
                for n in G.nodes():
                    if G.nodes[n]['source'] > 0:
                        root=n
            except:
                root=list(G.nodes())[0]
            path=self.path_list(G,root)
            self.list_path.append(list(path.values()))
            print('...Done...')
        if self.BRANCHING:
            print('...Caculating graph branching amount ...')
            rb=self.track_branchings(G)
            self.list_branching.append(rb)
            print('...Done...')

        if self.SCALING:
            print('...Caculating branch scaling...')
            n,d=self.calc_scaling(G)
            self.list_scaling.append(n)
            self.list_delta.append(d)

            print('...Done...')

    def analyze_export(self):
        #check active tics
        if self.NULLITY:
            np.save(op.join(self.output_dir,self.output_tag+'nullity'),self.list_nullity)
        if self.ROBUSTNESS:
            np.save(op.join(self.output_dir,self.output_tag+'robustness'),self.list_robustness)
        if self.RENTIAN_PHYS:
            np.save(op.join(self.output_dir,self.output_tag+'rent_nodes_phy'),self.list_n_phy)
            np.save(op.join(self.output_dir,self.output_tag+'rent_edge_phy'),self.list_e_phy)
            np.save(op.join(self.output_dir,self.output_tag+'rent_pearson_phy'),self.list_pearson_phy)
            np.save(op.join(self.output_dir,self.output_tag+'rent_exp_phy'),self.list_exp_phy)
        if self.RENTIAN_TOPO:
            np.save(op.join(self.output_dir,self.output_tag+'rent_nodes_topo'),self.list_n_topo)
            np.save(op.join(self.output_dir,self.output_tag+'rent_edge_topo'),self.list_e_topo)
            np.save(op.join(self.output_dir,self.output_tag+'rent_pearson_topo'),self.list_pearson_topo)
            np.save(op.join(self.output_dir,self.output_tag+'rent_exp_topo'),self.list_exp_topo)
        if self.CYCLE_COAL:
            np.save(op.join(self.output_dir,self.output_tag+'cycle_asymmetry'),self.list_A)
        if self.CYCLE_COAL_AVG:
            np.save(op.join(self.output_dir,self.output_tag+'cycle_asymmetry_avg'),self.list_A)
        if self.PERCOLATION_BOND:
            np.save(op.join(self.output_dir,self.output_tag+'percolation_bond'),self.list_bond)
        if self.PERCOLATION_SITE:
            np.save(op.join(self.output_dir,self.output_tag+'percolation_site'),self.list_site)
        if self.PATH_HIST:
            np.save(op.join(self.output_dir,self.output_tag+'path_hist'),self.list_path)
        if self.BRANCHING:
            np.save(op.join(self.output_dir,self.output_tag+'branching'),self.list_branching)
        if self.SCALING:
            np.save(op.join(self.output_dir,self.output_tag+'scaling'),self.list_scaling)

    def save_nparray(self, nparray,label):

        np.save(op.join(self.output_dir,label),nparray)
    #analyze various graph properties
    def calc_hist_weights(self,G):
        #calc histogram for final conductivity values and return histogram and bins, with customized bin size

        w=nx.get_edge_attributes(G,'weights')
        seq=[]
        seq.append(pow(10,-25))
        for i in range(24):
            seq.append(pow(10,-24+i))
        for i in range(int(np.amax(w))):
            seq.append(i+1)
        hist_w, bins =np.histogram(w,bins=seq)

        slots=[]
        j=0
        while True:
            slots.append((bins[j]+bins[j+1])/2.)
            j+=1
            if (j+1)==len(bins):
                break
        return slots, hist_w

    def calc_nullity(self,G):
        #calc nullity, the number of cyclic vectors
        H=G.to_undirected()
        nullity=0
        if nx.number_of_nodes(H) > 5:
            c=nx.number_connected_components(H)
            GT=self.reference_graph
            nullity=(c+nx.number_of_edges(H)-nx.number_of_nodes(H))/(1.+nx.number_of_edges(GT)-nx.number_of_nodes(GT))

        return nullity

    def construct_nullgraphs(self,G):
        #get minimum spanning tree from pre-constructed network
        H=G.to_undirected()
        MST=nx.minimum_spanning_tree(H,weight='weight')
        #get nodes from graph and perform a greedy triangulation
        pos=nx.get_node_attributes(H,'pos')
        node_points=list(pos.values())

        V=sc.spatial.Voronoi(node_points)
        GT=nx.Graph()

        for j,i in enumerate(V.points):
            GT.add_node(j,pos=i, label=j)
        for j,i in enumerate(V.ridge_points):
            GT.add_edge(i[0],i[1],slope=(V.points[i[0]],V.points[i[1]]), label=j)
        V.close()
        return MST,GT

    def calc_robustness(self,G):
        # calc robustness of graph by randomly cutting edges, till largest component is N/2
        iterations=50
        iteration_R=np.zeros(iterations)

        N_init=G.number_of_nodes()
        E_init=G.number_of_edges()
        N_2=N_init*0.5
        GU=G.to_undirected()
        for i in range(iterations):
            #create auxilarry graph object from largest connected component
            #cut out random edges until largest recognizable component falls under threshold, average over several realizations
            J=GU.copy()
            num_e_cut=0
            num_c=N_init
            E=E_init

            while True:

                idx_e=rd.randrange((E-num_e_cut))
                list_e=list(J.edges())
                J.remove_edge(*list_e[idx_e])
                list_components=nx.connected_components(J)
                list_size=np.array([len(j) for j in list_components])
                num_c=np.amax(list_size)
                num_e_cut+=1
                if num_c <= N_2:
                    iteration_R[i] = float(num_e_cut)/E_init
                    break

        avg_R=np.mean(iteration_R)
        return avg_R

    def calc_rentian(self,G,mode):
        # calc Rentian scaling exponents for physical boxing and avstract topological partionining
        list_n=[]
        list_e=[]

        if mode=='physical':
            #set system size x,yz wise
            iterations=5000
            iter_count=0
            N=G.number_of_nodes()
            nodes=list(G.nodes())
            pos=np.array([G.nodes[n]['pos'] for n in nodes])
            #determine dimensionality and set global box_hull
            dim=len(pos[0,:])
            coord=[(np.amax(pos[:,i])-np.amin(pos[:,i])) for i,p in enumerate(pos[0,:])]
            avg_l=np.mean(np.array(coord))
            hull = sc.spatial.Delaunay(pos)

            while True:

                #pick random node and create box of random size around it
                idx_n=rd.randrange(N)
                box_l=rd.uniform(0.,avg_l/2.)
                point=pos[idx_n]
                box_edge_points=[point]

                #check whether box is in convex hull
                if dim==3:
                    for i in [-1,1]:
                        for j in [-1,1]:
                            for k in [-1,1]:
                                box_edge_points.append(point+np.array([i,j,k])*box_l)
                elif dim==2:
                    for i in [-1,1]:
                        for j in [-1,1]:
                                box_edge_points.append(point+np.array([i,j])*box_l)
                else:
                    print('NO SUITABLE DIMENSION!')
                    break
                box_edge_points=np.array(box_edge_points)

                c1,c2=0,0
                list_nodes=[]
                bool_hull=hull.find_simplex(box_edge_points)

                if np.all(bool_hull >= 0 ):

                    #when box in hull, count how many nodes inside & how many edges cross the borderfaces
                    iter_count+=1
                    box_hull=sc.spatial.Delaunay(box_edge_points)

                    list_bool=box_hull.find_simplex(pos)
                    for i,n in enumerate(nodes):
                        #check how many nodes are inside the box
                        if list_bool[i] >= 0:

                            c1+=1
                            list_nodes.append(n)
                    list_n.append(c1)
                    list_edges=list(G.edges(list_nodes))
                    for e in list_edges:
                        #check wheter edge ends are inside the box
                        bool_boxhull=box_hull.find_simplex(np.array([G.nodes[e[0]]['pos'],G.nodes[e[1]]['pos']]))
                        if np.any(bool_boxhull < 0):
                            c2+=1
                    list_e.append(c2)
                    box_hull.close()

                if iter_count==iterations:
                    break
            hull.close()

        if mode=='topological':

            self.mem_n=[]
            self.mem_e=[]

            dict_nodes={}
            G_fin=nx.MultiGraph()
            for idx_n,n in enumerate(G.nodes()):
                G_fin.add_node(idx_n+1)
                dict_nodes.update({n:idx_n+1})
            for idx_e,e in enumerate(G.edges()):
                G_fin.add_edge(dict_nodes[e[0]],dict_nodes[e[1]])

            fname=op.join(self.output_dir,'hypergraphfile.txt')
            fo=open(fname,'w')
            num_N=nx.number_of_nodes(G_fin)
            fo.write(str(nx.number_of_edges(G_fin))+' '+str(num_N))
            for e in G_fin.edges():
                fo.write('\n'+str(e[0])+' '+str(e[1]))
            fo.close()

            UBfactor=1
            Nparts=[2**(i+1) for i in range(num_N) if 2**(i+1) < num_N]
            # Nparts=[2*(i+1) for i in range(num_N/2) ]
            for j,N in enumerate(Nparts):
                if j > 2 and j < (len(Nparts)-1):
                    os.system("'/Users/kramer/hmetis-1.5-osx-i686/shmetis' "+str(fname)+" "+str(N)+" "+str(UBfactor)+' > '+self.output_dir+'proxy_file')
                    # reading data directly from bash output
                    f=open(self.output_dir+'proxy_file')
                    list_data=[]
                    data_line=False
                    for lines in f:
                        # print(lines)
                        if 'Timing' in lines:
                            break
                        elif data_line:

                            numbs=re.findall(r'\d+', lines)
                            for n in numbs:
                                 list_data.append(int(n))

                        elif 'Partition Sizes & External Degrees' in lines:
                            data_line=True

                    for n,e in zip(list_data[0::2],list_data[1::2]):
                        if n!=0 and e!=0:
                            self.mem_n.append(n)
                            self.mem_e.append(e)

            os.system('rm '+fname+'*')

            list_n = self.mem_n
            list_e = self.mem_e

        p,exp = self.calc_rent_exponent(list_n,list_e)

        return list_n,list_e,p,exp

    def calc_rent_exponent(self,list_n,list_e):

        # calc the scaling exponents for the given datasets
        X=list_n
        Y=list_e

        X_log=np.log10(X)
        Y_log=np.log10(Y)

        p=scipy.stats.linregress(X_log,Y_log)
        return p[2],p[0]

    def calc_cycle_coalescence(self,G,cycle_basis):

        #create cycle_map_tree with cycles' edges as tree nodes
        Cycle_Tree=nx.Graph()
        for idx_c,c in enumerate(cycle_basis):
             Cycle_Tree.add_node(tuple(c.edges(keys=True)),label='base',weight=1.,branch_type='none',pos=(-1,-1))

        edges=nx.get_edge_attributes(G,'weight')
        sorted_edges=sorted(edges,key=edges.__getitem__)
        counter_c=0

        for e in sorted_edges:
            #check whether all cycles are merged

            if len(cycle_basis)== 1:
                break
            cycles={}
            for idx_c,c in enumerate(cycle_basis):
                if c.has_edge(*e):
                    if 'minimum_weight' in self.basis_mode:
                        cycles.update({idx_c:c.graph['cycle_weight']})
                    else:
                        cycles.update({idx_c:nx.number_of_edges(c)})

            if len(cycles.values()) >= 2:

                idx_list=sorted(cycles,key=cycles.__getitem__)

                c1=cycle_basis[idx_list[0]]
                c2=cycle_basis[idx_list[1]]
                c1_edges=c1.edges(keys=True)
                c2_edges=c2.edges(keys=True)
                # print(len(c1_edges))
                # print(len(c2_edges))
                merged_cycle=nx.MultiGraph()
                merged_cycle.graph['cycle_weight']=0
                for m in c1_edges:
                    merged_cycle.add_edge(*m)

                for m in c2_edges:
                    if merged_cycle.has_edge(*m):
                        merged_cycle.remove_edge(*m)
                    else:
                        merged_cycle.add_edge(*m)
                for m in merged_cycle.edges():
                    merged_cycle.graph['cycle_weight']+=G.edges[m]['weight']
                list_merged=list(merged_cycle.nodes())
                for n in list_merged:
                    if merged_cycle.degree(n)==0:
                        merged_cycle.remove_node(n)
                # build merging tree
                if Cycle_Tree.nodes[tuple(c1_edges)]['label']=='base':
                    Cycle_Tree.nodes[tuple(c1_edges)]['pos']=(counter_c,0)
                    # Cycle_Tree.nodes[tuple(c1_edges)]['branch_type']='none'
                    counter_c+=1
                if Cycle_Tree.nodes[tuple(c2_edges)]['label']=='base':
                    Cycle_Tree.nodes[tuple(c2_edges)]['pos']=(counter_c,0)
                    # Cycle_Tree.nodes[tuple(c2_edges)]['branch_type']='none'
                    counter_c+=1

                cycle_basis.remove(c1)
                cycle_basis.remove(c2)
                cycle_basis.append(merged_cycle)
                # build up the merging tree, set leave weights to nodes, set asymetry value to binary branchings

                mc_edges=merged_cycle.edges(keys=True)
                c_x=(Cycle_Tree.nodes[tuple(c1_edges)]['pos'][0]+Cycle_Tree.nodes[tuple(c2_edges)]['pos'][0])/2.
                c_y=np.amax([Cycle_Tree.nodes[tuple(c1_edges)]['pos'][1],Cycle_Tree.nodes[tuple(c2_edges)]['pos'][1]])+2
                c1_weight=Cycle_Tree.nodes[tuple(c1_edges)]['weight']
                c2_weight=Cycle_Tree.nodes[tuple(c2_edges)]['weight']

                Cycle_Tree.add_node(tuple(mc_edges),pos=(c_x,c_y),label='merged',weight=c1_weight+c2_weight)
                Cycle_Tree.add_edge(tuple(c1_edges),tuple(mc_edges))
                Cycle_Tree.add_edge(tuple(c2_edges),tuple(mc_edges))
                # criterium for avoiding redundant branchings
                if c_y>=6:
                    Cycle_Tree.nodes[tuple(mc_edges)]['branch_type']='vanpelt_2'
                    Cycle_Tree.nodes[tuple(mc_edges)]['asymmetry']=np.absolute((c1_weight-c2_weight))/(c1_weight+c2_weight-2.)
                else:
                    Cycle_Tree.nodes[tuple(mc_edges)]['branch_type']='none'

            else:
                continue
        # calc topological asymmetry per binary branching point
        list_asymmetry=[]
        # print(nx.number_connected_components(Cycle_Tree))
        for n in Cycle_Tree.nodes():
            if Cycle_Tree.nodes[n]['pos'][0]==-1:
                Cycle_Tree.nodes[n]['pos']=(counter_c,0)
                counter_c+=1
        # print(nx.get_node_attributes(Cycle_Tree,'branch_type'))
        for n in Cycle_Tree.nodes():
            # if Cycle_Tree.nodes[n]['label']=='merged':
            if Cycle_Tree.nodes[n]['branch_type']=='vanpelt_2':
                list_asymmetry.append(Cycle_Tree.nodes[n]['asymmetry'])

        return Cycle_Tree, list_asymmetry

    def calc_cycle_coalescence_avg(self,G,cycle_basis_list):

        tree_list=[]
        superlist_asymmetry=[]
        for cbl in cycle_basis_list:
            #create cycle_map_tree with cycles' edges as tree nodes
            Cycle_Tree=nx.Graph()
            for idx_c,c in enumerate(cbl):
                 Cycle_Tree.add_node(tuple(c.edges(keys=True)),label='base',weight=1.,branch_type='none',pos=(-1,-1))

            edges=nx.get_edge_attributes(G,'weight')
            sorted_edges=sorted(edges,key=edges.__getitem__)
            counter_c=0

            for e in sorted_edges:
                #check whether all cycles are merged
                if len(cbl)== 1:
                    break
                cycles={}
                for idx_c,c in enumerate(cbl):
                    if c.has_edge(*e):
                        if 'minimum_weight' in self.basis_mode:
                            cycles.update({idx_c:c.graph['cycle_weight']})
                        else:
                            cycles.update({idx_c:nx.number_of_edges(c)})

                if len(cycles.values()) >= 2:

                    idx_list=sorted(cycles,key=cycles.__getitem__)

                    c1=cbl[idx_list[0]]
                    c2=cbl[idx_list[1]]
                    c1_edges=c1.edges(keys=True)
                    c2_edges=c2.edges(keys=True)
                    # print(len(c1_edges))
                    # print(len(c2_edges))
                    merged_cycle=nx.MultiGraph()
                    merged_cycle.graph['cycle_weight']=0
                    for m in c1_edges:
                        merged_cycle.add_edge(*m)

                    for m in c2_edges:
                        if merged_cycle.has_edge(*m):
                            merged_cycle.remove_edge(*m)
                        else:
                            merged_cycle.add_edge(*m)
                    for m in merged_cycle.edges():
                        merged_cycle.graph['cycle_weight']+=G.edges[m]['weight']
                    list_merged=list(merged_cycle.nodes())
                    for n in list_merged:
                        if merged_cycle.degree(n)==0:
                            merged_cycle.remove_node(n)
                    # build merging tree
                    if Cycle_Tree.nodes[tuple(c1_edges)]['label']=='base':
                        Cycle_Tree.nodes[tuple(c1_edges)]['pos']=(counter_c,0)
                        # Cycle_Tree.nodes[tuple(c1_edges)]['branch_type']='none'
                        counter_c+=1
                    if Cycle_Tree.nodes[tuple(c2_edges)]['label']=='base':
                        Cycle_Tree.nodes[tuple(c2_edges)]['pos']=(counter_c,0)
                        # Cycle_Tree.nodes[tuple(c2_edges)]['branch_type']='none'
                        counter_c+=1

                    cbl.remove(c1)
                    cbl.remove(c2)
                    cbl.append(merged_cycle)
                    # build up the merging tree, set leave weights to nodes, set asymetry value to binary branchings

                    mc_edges=merged_cycle.edges(keys=True)
                    c_x=(Cycle_Tree.nodes[tuple(c1_edges)]['pos'][0]+Cycle_Tree.nodes[tuple(c2_edges)]['pos'][0])/2.
                    c_y=np.amax([Cycle_Tree.nodes[tuple(c1_edges)]['pos'][1],Cycle_Tree.nodes[tuple(c2_edges)]['pos'][1]])+2
                    c1_weight=Cycle_Tree.nodes[tuple(c1_edges)]['weight']
                    c2_weight=Cycle_Tree.nodes[tuple(c2_edges)]['weight']

                    Cycle_Tree.add_node(tuple(mc_edges),pos=(c_x,c_y),label='merged',weight=c1_weight+c2_weight)
                    Cycle_Tree.add_edge(tuple(c1_edges),tuple(mc_edges))
                    Cycle_Tree.add_edge(tuple(c2_edges),tuple(mc_edges))
                    # criterium for avoiding redundant branchings

                    if c_y>=6:
                        Cycle_Tree.nodes[tuple(mc_edges)]['branch_type']='vanpelt_2'
                        Cycle_Tree.nodes[tuple(mc_edges)]['asymmetry']=np.absolute((c1_weight-c2_weight))/(c1_weight+c2_weight-2.)
                    else:
                        Cycle_Tree.nodes[tuple(mc_edges)]['branch_type']='none'

                else:
                    continue
            # calc topological asymmetry per binary branching point
            list_asymmetry=[]
            # print(nx.number_connected_components(Cycle_Tree))
            for n in Cycle_Tree.nodes():
                if Cycle_Tree.nodes[n]['pos'][0]==-1:
                    Cycle_Tree.nodes[n]['pos']=(counter_c,0)
                    counter_c+=1

            for n in Cycle_Tree.nodes():
                # if Cycle_Tree.nodes[n]['label']=='merged':
                if Cycle_Tree.nodes[n]['branch_type']=='vanpelt_2':
                    list_asymmetry.append(Cycle_Tree.nodes[n]['asymmetry'])
            superlist_asymmetry.append(list_asymmetry)
            tree_list.append(Cycle_Tree)
        return tree_list, superlist_asymmetry

    def merge_tiles(self,G,c1,c2):

        c1_edges=c1.edges(keys=True)
        c2_edges=c2.edges(keys=True)
        merged_cycle=nx.MultiGraph()
        merged_cycle.graph['cycle_weight']=0
        for m in c1_edges:
            merged_cycle.add_edge(*m)

        for m in c2_edges:
            if merged_cycle.has_edge(*m):
                merged_cycle.remove_edge(*m)
            else:
                merged_cycle.add_edge(*m)
        for m in merged_cycle.edges():
            merged_cycle.graph['cycle_weight']+=G.edges[m]['weight']
        list_merged=list(merged_cycle.nodes())
        for n in list_merged:
            if merged_cycle.degree(n)==0:
                merged_cycle.remove_node(n)
        return merged_cycle
    # @profile
    def generate_cycle_lists(self,G):

        total_cycle_dict={}
        total_cycle_list=[]
        super_list=[]
        # check for graph_type, then check for paralles in the Graph, if existent insert dummy nodes to resolve conflict, cast the network onto simple graph afterwards
        # choose method to perform construction of minimal basis
        if 'minimum_weight' in self.basis_mode:
            counter=0
            T=nx.minimum_spanning_tree(G,weight='weight')
            D=nx.difference(G,T)
            for n in G.nodes():
                for e in D.edges():
                    p_in=nx.shortest_path(G,source=n,target=e[0],weight='weight')
                    p_out=nx.all_shortest_paths(G,source=n,target=e[1],weight='weight')
                    for p in p_out:
                        simple_cycle=nx.Graph(cycle_weight=0.)
                        nx.add_path(simple_cycle,p_in)
                        nx.add_path(simple_cycle,p)
                        simple_cycle.add_edge(*e)
                        if nx.is_eulerian(simple_cycle):
                            # relabeling and weighting graph
                            simple_cycle=nx.MultiGraph(simple_cycle)
                            for m in simple_cycle.edges():
                                simple_cycle.graph['cycle_weight']+=G.edges[m]['weight']
                            total_cycle_list.append(simple_cycle)

                            total_cycle_dict.update({counter:simple_cycle.graph['cycle_weight']})
                            counter+=1
                            break
        if 'minimum_tile' in self.basis_mode:
            counter=0
            labels_n = nx.get_node_attributes(G,'label')
            sorted_label_n_list=sorted(labels_n ,key=labels_n.__getitem__)
            for n in sorted_label_n_list:
                # building new tree using breadth first
                TR=self.breadth_first_tree(G,n)
                D=nx.difference(G,TR)
                labels_e={}
                for e in D.edges():
                    labels_e[e]=G.edges[e]['label']
                sorted_label_e_list=sorted(labels_e ,key=labels_e.__getitem__)

                for e in sorted_label_e_list:
                    p_in=nx.shortest_path(TR,source=n,target=e[0])
                    p_out=nx.shortest_path(TR,source=n,target=e[1])
                    # label pathways
                    simple_cycle=nx.MultiGraph(cycle_weight=0.)
                    nx.add_path(simple_cycle,p_in)
                    nx.add_path(simple_cycle,p_out)
                    simple_cycle.add_edge(*e)

                    list_n=list(simple_cycle.nodes())
                    seen={}
                    for m in list(simple_cycle.edges()):
                        num_conncetions=simple_cycle.number_of_edges(*m)
                        if num_conncetions > 1 and m not in seen.keys():
                            seen[m]=1
                        elif num_conncetions > 1:
                            seen[m]+=1
                    for m in seen:
                        for i in range(seen[m]):
                            simple_cycle.remove_edge(m[0],m[1],i)
                    for q in list_n:
                        if simple_cycle.degree(q)==0:
                            simple_cycle.remove_node(q)

                    if nx.is_eulerian(simple_cycle):
                        # relabeling and weighting graph
                        for m in simple_cycle.edges():
                            simple_cycle.graph['cycle_weight']+=G.edges[m]['weight']
                        total_cycle_list.append(simple_cycle)
                        total_cycle_dict.update({counter:nx.number_of_edges(simple_cycle)})

                        counter+=1

        else:
            # create cycle subgraphs from super_list
            for n in G.nodes():
                if G.degree(n) > 1:
                    c_list=nx.cycle_basis(G,n)
                    if not super_list:
                        super_list=list(c_list)
                    else:
                        super_list+=c_list

            for idx_c,c in enumerate(super_list):
                J=nx.MultiGraph()
                nx.add_cycle(J,c)
                total_cycle_list.append(J)
                total_cycle_dict.update({idx_c:nx.number_of_edges(J)})

        return total_cycle_dict,total_cycle_list,super_list

    def construct_minimum_basis(self,G):
        # calc minimum weight basis and construct dictionary for weights of edges, takes a leave-less, connected, N > 1 SimpleGraph as input, no self-loops optimally, deviations are not raising any warnings
        # total_cycle_dict={}
        # total_cycle_list=[]
        # super_list=[]
        # # check for graph_type, then check for paralles in the Graph, if existent insert dummy nodes to resolve conflict, cast the network onto simple graph afterwards
        # # choose method to perform construction of minimal basis
        # if 'minimum_weight' in self.basis_mode:
        #     counter=0
        #     T=nx.minimum_spanning_tree(G,weight='weight')
        #     D=nx.difference(G,T)
        #     for n in G.nodes():
        #         for e in D.edges():
        #             p_in=nx.shortest_path(G,source=n,target=e[0],weight='weight')
        #             p_out=nx.all_shortest_paths(G,source=n,target=e[1],weight='weight')
        #             for p in p_out:
        #                 simple_cycle=nx.Graph(cycle_weight=0.)
        #                 nx.add_path(simple_cycle,p_in)
        #                 nx.add_path(simple_cycle,p)
        #                 simple_cycle.add_edge(*e)
        #                 if nx.is_eulerian(simple_cycle):
        #                     # relabeling and weighting graph
        #                     simple_cycle=nx.MultiGraph(simple_cycle)
        #                     for m in simple_cycle.edges():
        #                         simple_cycle.graph['cycle_weight']+=G.edges[m]['weight']
        #                     total_cycle_list.append(simple_cycle)
        #
        #                     total_cycle_dict.update({counter:simple_cycle.graph['cycle_weight']})
        #                     counter+=1
        #                     break
        # if 'minimum_tile' in self.basis_mode:
        #     cycles_id_dict={}
        #     counter=0
        #     labels_n = nx.get_node_attributes(G,'label')
        #     sorted_label_n_list=sorted(labels_n ,key=labels_n.__getitem__)
        #     for n in sorted_label_n_list:
        #         # building new tree using breadth first
        #         TR=self.breadth_first_tree(G,n)
        #         D=nx.difference(G,TR)
        #         labels_e={}
        #         for e in D.edges():
        #             labels_e[e]=G.edges[e]['label']
        #         sorted_label_e_list=sorted(labels_e ,key=labels_e.__getitem__)
        #
        #         for e in sorted_label_e_list:
        #             p_in=nx.shortest_path(TR,source=n,target=e[0])
        #             p_out=nx.shortest_path(TR,source=n,target=e[1])
        #             # label pathways
        #             simple_cycle=nx.MultiGraph(cycle_weight=0.)
        #             nx.add_path(simple_cycle,p_in)
        #             nx.add_path(simple_cycle,p_out)
        #             simple_cycle.add_edge(*e)
        #
        #             list_n=list(simple_cycle.nodes())
        #             seen={}
        #             for m in list(simple_cycle.edges()):
        #                 num_conncetions=simple_cycle.number_of_edges(*m)
        #                 if num_conncetions > 1 and m not in seen.keys():
        #                     seen[m]=1
        #                 elif num_conncetions > 1:
        #                     seen[m]+=1
        #             for m in seen:
        #                 for i in range(seen[m]):
        #                     simple_cycle.remove_edge(m[0],m[1],i)
        #             for q in list_n:
        #                 if simple_cycle.degree(q)==0:
        #                     simple_cycle.remove_node(q)
        #
        #             if nx.is_eulerian(simple_cycle):
        #                 # relabeling and weighting graph
        #                 for m in simple_cycle.edges():
        #                     simple_cycle.graph['cycle_weight']+=G.edges[m]['weight']
        #                 total_cycle_list.append(simple_cycle)
        #                 total_cycle_dict.update({counter:nx.number_of_edges(simple_cycle)})
        #
        #                 counter+=1
        #
        # else:
        #     # create cycle subgraphs from super_list
        #     cycles_id_dict={}
        #     for n in G.nodes():
        #         if G.degree(n) > 1:
        #             c_list=nx.cycle_basis(G,n)
        #             if not super_list:
        #                 super_list=list(c_list)
        #             else:
        #                 super_list+=c_list
        #
        #     for idx_c,c in enumerate(super_list):
        #         J=nx.MultiGraph()
        #         nx.add_cycle(J,c)
        #         total_cycle_list.append(J)
        #         total_cycle_dict.update({idx_c:nx.number_of_edges(J)})
        #sort basis vectors according to weight, creating a new minimum weight basis from the total_cycle_list
        nullity=nx.number_of_edges(G)-nx.number_of_nodes(G)+nx.number_connected_components(G)
        total_cycle_dict,total_cycle_list,super_list=self.generate_cycle_lists(G)
        sorted_cycle_list=sorted(total_cycle_dict,key=total_cycle_dict.__getitem__)
        minimum_basis=[]
        EC=nx.MultiGraph()
        counter=0
        for c in sorted_cycle_list:

            cycle_edges_in_basis=True

            for e in total_cycle_list[c].edges(keys=True):
                if not EC.has_edge(*e):
                    EC.add_edge(*e,label=counter)
                    counter+=1
                    cycle_edges_in_basis=False
            #if cycle edges where not part of the supergraph yet then it becomes automatically part of the basis
            if not cycle_edges_in_basis:
                minimum_basis.append(total_cycle_list[c])
            #if cycle edges are already included we check for linear dependece
            else:
                linear_independent=False
                rows=len(list(EC.edges()))
                columns=len(minimum_basis)+1
                E=np.zeros((rows,columns))
                # translate the existent basis vectors into z2 representation
                for idx_c,cycle in enumerate(minimum_basis+[total_cycle_list[c]]):
                    for m in cycle.edges(keys=True):
                        if EC.has_edge(*m):
                            E[EC.edges[m]['label'],idx_c]=1

                # calc echelon form
                a_columns=np.arange(columns-1)
                zwo=np.ones(columns)*2
                for column in a_columns:
                    idx_nz=np.nonzero(E[column:,column])[0]
                    if idx_nz.size:
                        if len(idx_nz)==1:
                            E[column,:],E[idx_nz[0]+column,:]=E[idx_nz[0]+column,:].copy(),E[column,:].copy()
                        else:
                            for r in idx_nz[1:]:
                                aux_E=np.add(E[r+column],E[idx_nz[0]+column])
                                E[r+column]=np.mod(aux_E,zwo)
                            E[column,:],E[idx_nz[0]+column,:]=E[idx_nz[0]+column,:].copy(),E[column,:].copy()
                    else:
                        sys.exit('Error: minimum_weight_basis containing inconsistencies ...')
                # test echelon form for inconsistencies
                for r in range(rows):
                    line_check=np.nonzero(E[r])[0]
                    if len(line_check)==1 and line_check[0]==(columns-1):
                        linear_independent=True

                        break
                if linear_independent:
                    minimum_basis.append(total_cycle_list[c])

            if len(minimum_basis)==nullity:
                break

        if len(minimum_basis)<nullity:
            sys.exit('Error: Cycle basis badly constructed')

        return minimum_basis

    def construct_minimum_basis_multi(self,G,num_cb):
        cycle_basis_list=[]

        for num in range(num_cb):
            total_cycle_dict={}
            total_cycle_list=[]
            super_list=[]
            # check for graph_type, then check for paralles in the Graph, if existent insert dummy nodes to resolve conflict, cast the network onto simple graph afterwards
            # choose method to perform construction of minimal basis
            if 'minimum_weight' in self.basis_mode:
                counter=0
                T=nx.minimum_spanning_tree(G,weight='weight')
                D=nx.difference(G,T)
                for n in G.nodes():
                    for e in D.edges():
                        p_in=nx.shortest_path(G,source=n,target=e[0],weight='weight')
                        p_out=nx.all_shortest_paths(G,source=n,target=e[1],weight='weight')
                        for p in p_out:
                            simple_cycle=nx.Graph(cycle_weight=0.)
                            nx.add_path(simple_cycle,p_in)
                            nx.add_path(simple_cycle,p)
                            simple_cycle.add_edge(*e)
                            if nx.is_eulerian(simple_cycle):
                                # relabeling and weighting graph
                                simple_cycle=nx.MultiGraph(simple_cycle)
                                for m in simple_cycle.edges():
                                    simple_cycle.graph['cycle_weight']+=G.edges[m]['weight']
                                total_cycle_list.append(simple_cycle)

                                total_cycle_dict.update({counter:simple_cycle.graph['cycle_weight']})
                                counter+=1
                                break
            if 'minimum_tile' in self.basis_mode:
                counter=0
                labels_n = nx.get_node_attributes(G,'label')
                sorted_label_n_list=sorted(labels_n ,key=labels_n.__getitem__)
                for n in sorted_label_n_list:
                    # building new tree using breadth first
                    TR=self.breadth_first_tree_random(G,n)
                    D=nx.difference(G,TR)
                    labels_e={}
                    for e in D.edges():
                        labels_e[e]=G.edges[e]['label']
                    sorted_label_e_list=sorted(labels_e ,key=labels_e.__getitem__)

                    for e in sorted_label_e_list:
                        p_in=nx.shortest_path(TR,source=n,target=e[0])
                        p_out=nx.shortest_path(TR,source=n,target=e[1])
                        # label pathways
                        simple_cycle=nx.MultiGraph(cycle_weight=0.)
                        nx.add_path(simple_cycle,p_in)
                        nx.add_path(simple_cycle,p_out)
                        simple_cycle.add_edge(*e)

                        list_n=list(simple_cycle.nodes())
                        seen={}
                        for m in list(simple_cycle.edges()):
                            num_conncetions=simple_cycle.number_of_edges(*m)
                            if num_conncetions > 1 and m not in seen.keys():
                                seen[m]=1
                            elif num_conncetions > 1:
                                seen[m]+=1
                        for m in seen:
                            for i in range(seen[m]):
                                simple_cycle.remove_edge(m[0],m[1],i)
                        for q in list_n:
                            if simple_cycle.degree(q)==0:
                                simple_cycle.remove_node(q)

                        if nx.is_eulerian(simple_cycle):
                            # relabeling and weighting graph
                            for m in simple_cycle.edges():
                                simple_cycle.graph['cycle_weight']+=G.edges[m]['weight']
                            total_cycle_list.append(simple_cycle)
                            total_cycle_dict.update({counter:nx.number_of_edges(simple_cycle)})

                            counter+=1

            else:
                # create cycle subgraphs from super_list
                for n in G.nodes():
                    if G.degree(n) > 1:
                        c_list=nx.cycle_basis(G,n)
                        if not super_list:
                            super_list=list(c_list)
                        else:
                            super_list+=c_list

                for idx_c,c in enumerate(super_list):
                    J=nx.MultiGraph()
                    nx.add_cycle(J,c)
                    total_cycle_list.append(J)
                    total_cycle_dict.update({idx_c:nx.number_of_edges(J)})
            #sort basis vectors according to weight, creating a new minimum weight basis from the total_cycle_list
            nullity=nx.number_of_edges(G)-nx.number_of_nodes(G)+nx.number_connected_components(G)

            sorted_cycle_list=sorted(total_cycle_dict,key=total_cycle_dict.__getitem__)
            minimum_basis=[]
            EC=nx.MultiGraph()
            counter=0
            for c in sorted_cycle_list:

                cycle_edges_in_basis=True

                for e in total_cycle_list[c].edges(keys=True):
                    if not EC.has_edge(*e):
                        EC.add_edge(*e,label=counter)
                        counter+=1
                        cycle_edges_in_basis=False
                #if cycle edges where not part of the supergraph yet then it becomes automatically part of the basis
                if not cycle_edges_in_basis:
                    minimum_basis.append(total_cycle_list[c])
                #if cycle edges are already included we check for linear dependece
                else:
                    linear_independent=False
                    rows=len(list(EC.edges()))
                    columns=len(minimum_basis)+1
                    E=np.zeros((rows,columns))
                    # translate the existent basis vectors into z2 representation
                    for idx_c,cycle in enumerate(minimum_basis+[total_cycle_list[c]]):
                        for m in cycle.edges(keys=True):
                            if EC.has_edge(*m):
                                E[EC.edges[m]['label'],idx_c]=1

                    # calc echelon form
                    a_columns=np.arange(columns-1)
                    zwo=np.ones(columns)*2
                    for column in a_columns:
                        idx_nz=np.nonzero(E[column:,column])[0]
                        if idx_nz.size:
                            if len(idx_nz)==1:
                                E[column,:],E[idx_nz[0]+column,:]=E[idx_nz[0]+column,:].copy(),E[column,:].copy()
                            else:
                                for r in idx_nz[1:]:
                                    aux_E=np.add(E[r+column],E[idx_nz[0]+column])
                                    E[r+column]=np.mod(aux_E,zwo)
                                E[column,:],E[idx_nz[0]+column,:]=E[idx_nz[0]+column,:].copy(),E[column,:].copy()
                        else:
                            sys.exit('Error: minimum_weight_basis containing inconsistencies ...')
                    # test echelon form for inconsistencies
                    for r in range(rows):
                        line_check=np.nonzero(E[r])[0]
                        if len(line_check)==1 and line_check[0]==(columns-1):
                            linear_independent=True
                            break
                    if linear_independent:
                        minimum_basis.append(total_cycle_list[c])

                if len(minimum_basis)==nullity:
                    break

            if len(minimum_basis)<nullity:
                sys.exit('Error: Cycle basis badly constructed')
            cycle_basis_list.append(minimum_basis)
        return cycle_basis_list

    def calc_percolation_site(self,G):
        # to discuss: is averaging allowed here for phi?

        raw_s=[]
        raw_p=[]
        iterations=25

        N_init=float(nx.number_of_nodes(G))
        E_init=nx.number_of_edges(G)

        for i in range(iterations):

            f=1
            s=[]
            p=[]

            J=nx.MultiGraph(G)
            num_e_cut=0
            num_c=N_init
            E=E_init

            while True:

                if f <= 0.0001:
                    break

                idx_e=rd.randrange((E-num_e_cut))
                list_e=list(J.edges())
                J.remove_edge(*list_e[idx_e])
                list_components=nx.connected_components(J)
                list_size=np.array([len(j) for j in list_components])
                num_c=float(np.amax(list_size))
                num_e_cut+=1.
                phi=1.-num_e_cut/E_init
                if phi <= f:
                    p.append(phi)
                    s.append(num_c/N_init)
                    f-=0.05
                raw_s.append(s)
                raw_p.append(p)

        s=np.mean(raw_s,axis=0)
        p=np.mean(raw_p,axis=0)
        err_s=np.std(raw_s,axis=0)
        err_p=np.std(raw_p,axis=0)

        return p,s,err_p,err_s

    def calc_percolation_bond(self,G):

        raw_s=[]
        raw_p=[]
        iterations=50
        N_init=float(nx.number_of_nodes(G))

        for i in range(iterations):

            f=1

            s=[]
            p=[]

            J=nx.MultiGraph(G)
            num_n_cut=0
            num_c=N_init
            N=N_init

            while True:

                if f <= 0.0001:
                    break
                idx_n=rd.randrange(N-num_n_cut)
                list_n=list(J.nodes())
                J.remove_node(list_n[idx_n])
                list_components=nx.connected_components(J)
                list_size=np.array([len(j) for j in list_components])
                num_c=np.amax(list_size)
                num_n_cut+=1
                phi=1.-num_n_cut/N_init
                if phi <= f:
                    p.append(phi)
                    s.append(num_c/N_init)
                    f-=0.05
                raw_s.append(s)
                raw_p.append(p)

        s=np.mean(raw_s,axis=0)
        p=np.mean(raw_p,axis=0)
        err_s=np.std(raw_s,axis=0)
        err_p=np.std(raw_p,axis=0)

        return p,s,err_p,err_s

    def breadth_first_tree(self,G,root):

        T=nx.Graph()
        push_down={}
        for n in G.nodes():
            push_down[n]=False

        push_down[root]=True
        root_queue=[]
        labels_e = G.edges(root,'label')
        dict_labels_e={}
        for le in labels_e:
            dict_labels_e[(le[0],le[1])]=le[2]
        sorted_label_e_list=sorted(dict_labels_e,key=dict_labels_e.__getitem__)
        # print(sorted_label_e_list)
        for e in sorted_label_e_list:

            if e[0]==root:
                if not push_down[e[1]]:
                    T.add_edge(*e)
                    root_queue.append(e[1])
                    push_down[e[1]]=True

            else:
                if not push_down[e[0]]:
                    T.add_edge(*e)
                    root_queue.append(e[1])
                    push_down[e[0]]=True

        while T.number_of_nodes() < G.number_of_nodes():
            new_queue=[]
            for q in root_queue:
                labels_e = G.edges(q,'label')
                dict_labels_e={}
                for le in labels_e:
                    dict_labels_e[(le[0],le[1])]=le[2]
                sorted_label_e_list=sorted(dict_labels_e,key=dict_labels_e.__getitem__)
                # print(sorted_label_e_list)
                for e in sorted_label_e_list:
                    # print('edge#order:'+str(dict_labels_e[e]))
                    if e[0]==q:
                        if not push_down[e[1]]:
                            T.add_edge(*e)
                            new_queue.append(e[1])
                            push_down[e[1]]=True

                    else:
                        if not push_down[e[0]]:
                            T.add_edge(*e)
                            new_queue.append(e[1])
                            push_down[e[0]]=True
            root_queue=new_queue[:]
        return T

    def breadth_first_tree_random(self,G,root):

        T=nx.Graph()
        push_down={}
        for n in G.nodes():
            push_down[n]=False

        push_down[root]=True
        root_queue=[]
        labels_e = G.edges(root,'label')
        dict_labels_e={}
        for le in labels_e:
            dict_labels_e[(le[0],le[1])]=le[2]
        sorted_label_e_list=sorted(dict_labels_e,key=dict_labels_e.__getitem__)
        # print(sorted_label_e_list)
        for e in rd.shuffle(sorted_label_e_list):

            if e[0]==root:
                if not push_down[e[1]]:
                    T.add_edge(*e)
                    root_queue.append(e[1])
                    push_down[e[1]]=True

            else:
                if not push_down[e[0]]:
                    T.add_edge(*e)
                    root_queue.append(e[1])
                    push_down[e[0]]=True

        while T.number_of_nodes() < G.number_of_nodes():
            new_queue=[]
            for q in root_queue:
                labels_e = G.edges(q,'label')
                dict_labels_e={}
                for le in labels_e:
                    dict_labels_e[(le[0],le[1])]=le[2]
                sorted_label_e_list=sorted(dict_labels_e,key=dict_labels_e.__getitem__)
                # print(sorted_label_e_list)
                for e in rd.shuffle(sorted_label_e_list):
                    # print('edge#order:'+str(dict_labels_e[e]))
                    if e[0]==q:
                        if not push_down[e[1]]:
                            T.add_edge(*e)
                            new_queue.append(e[1])
                            push_down[e[1]]=True

                    else:
                        if not push_down[e[0]]:
                            T.add_edge(*e)
                            new_queue.append(e[1])
                            push_down[e[0]]=True
            root_queue=new_queue[:]
        return T

    def path_list(self,G,root):

        leaves=[]
        paths={}

        for n in G.nodes():
            if G.degree(n) == 1:
                leaves.append(n)
        for n in leaves:
            p=nx.shortest_path(nx.Graph(G),source=root,target=n)
            paths[tuple(p)]=len(p)

        return paths

    def track_branchings(self,G):

        hist_G=nx.degree_histogram(nx.Graph(G))

        return float(np.sum(hist_G[3:]))/nx.number_of_nodes(G)

    def murray_law( self , *args  ):

        y=1.-(np.sum([args[i]**args[-1] for i,x in enumerate(np.array(args)[:-1])]))
        return y

    # calc coefficients
    def murray_coefficients_root( self ,x, *args ):

        sign=list(args)[0]
        pars=list(args)[1]

        # coefficients1=[pars[0][1],pars[1][1]]
        coefficients1=[pars[0][1],pars[1][1],pars[2][1]]
        coefficients2=[pars[0][2],pars[1][2],pars[2][2]]
        # radii=[pars[0][0],pars[1][0]]
        radii=[pars[0][0],pars[1][0],pars[2][0]]

        # y0= np.multiply( radii[0] , np.sqrt(np.divide(np.subtract(np.absolute(x[1]),np.multiply(np.absolute(x[0]),coefficients2[0])),np.add(np.ones(3),np.multiply(np.absolute(x[2]),coefficients1[0])))) )
        # y1= np.multiply( radii[1] , np.sqrt(np.divide(np.subtract(np.absolute(x[1]),np.multiply(np.absolute(x[0]),coefficients2[1])),np.add(np.ones(3),np.multiply(np.absolute(x[2]),coefficients1[1])))) )
        # y2= np.multiply( radii[2] , np.sqrt(np.divide(np.subtract(np.absolute(x[1]),np.multiply(np.absolute(x[0]),coefficients2[2])),np.add(np.ones(3),np.multiply(np.absolute(x[2]),coefficients1[2])))) )
        # Y0=1. + np.sum( np.multiply( y0, sign ) )
        # Y1=1. + np.sum( np.multiply( y1, sign ) )
        # Y2=1. + np.sum( np.multiply( y2, sign ) )
        Y=np.zeros(3)

        for i in range(len(x)):
            A=np.subtract(np.absolute(x[1]),np.multiply(np.absolute(x[0]),coefficients2[i]))
            B=np.add(np.ones(3),np.multiply(np.absolute(x[2]),coefficients1[i]))
            y= np.multiply( radii[i] , np.sqrt(np.divide(A,B)) )
            Y[i]= 1. + np.sum( np.multiply( y, sign ) )

        # return [Y0,Y1,Y2]
        return Y

    def murray_coefficients_root_jac( self ,x, *args ):

        sign=list(args)[0]
        pars=list(args)[1]

        # coefficients1=[pars[0][1],pars[1][1]]
        coefficients1=[pars[0][1],pars[1][1],pars[2][1]]
        coefficients2=[pars[0][2],pars[1][2],pars[2][2]]
        # radii=[pars[0][0],pars[1][0]]
        radii=[pars[0][0],pars[1][0],pars[2][0]]

        Y=np.zeros(3)
        DY=np.zeros((3,3))

        for i in range(len(x)):
            A=np.subtract(np.absolute(x[1]),np.multiply(np.absolute(x[0]),coefficients2[i]))
            B=np.add(np.ones(3),np.multiply(np.absolute(x[2]),coefficients1[i]))
            y= np.multiply( radii[i] , np.sqrt(np.divide(A,B)) )
            Y[i]= 1. + np.sum( np.multiply( y, sign ) )

            y1= np.multiply( np.multiply(radii[i],coefficients2[i]) , np.sqrt(np.reciprocal(np.multiply(A,B))) )*(-0.5)
            y2= np.multiply( radii[i] , np.sqrt(np.reciprocal(np.multiply(A,B))) )*0.5
            y3= np.multiply( np.multiply(radii[i],coefficients1[i]) , np.divide(np.sqrt(np.divide(A,B)),B) )*(-0.5)
            DY[i,0]=np.sum( np.multiply( y1, sign ) )
            DY[i,1]=np.sum( np.multiply( y2, sign ) )
            DY[i,2]=np.sum( np.multiply( y3, sign ) )

        return Y, DY

    def murray_coefficients_tuple_jac( self ,x, *args ):


        sign=list(args)[0]
        pars=list(args)[1]

        coefficients1=[pars[0][1],pars[1][1],pars[2][1]]
        coefficients2=[pars[0][2],pars[1][2],pars[2][2]]
        radii=[pars[0][0],pars[1][0],pars[2][0]]

        Y=np.zeros(3)
        DY=np.zeros((3,3))

        for i in range(3):
            A=np.subtract(np.absolute(x[1]),np.multiply(np.absolute(x[0]),coefficients2[i]))
            B=np.add(np.ones(3),np.multiply(np.absolute(x[2]),coefficients1[i]))
            y= np.multiply( radii[i] , np.sqrt(np.divide(A,B)) )
            Y[i]= 1. + np.sum( np.multiply( y, sign ) )

            y1= np.multiply( np.multiply(radii[i],coefficients2[i]) , np.sqrt(np.reciprocal(np.multiply(A,B))) )*(-0.5)
            y2= np.multiply( radii[i] , np.sqrt(np.reciprocal(np.multiply(A,B))) )*0.5
            y3= np.multiply( np.multiply(radii[i],coefficients1[i]) , np.divide(np.sqrt(np.divide(A,B)),B) )*(-0.5)
            DY[i,0]=np.sum( np.multiply( y1, sign ) )
            DY[i,1]=np.sum( np.multiply( y2, sign ) )
            DY[i,2]=np.sum( np.multiply( y3, sign ) )

        G=np.zeros(3)
        for i in range(3):
            G[i]=2.*np.sum(np.multiply(Y,DY[:,i]))
        return G
        # return np.sum(np.power(Y,2)), G

    def murray_coefficients_tuple_min( self ,x, *args ):

        sign=list(args)[0]
        pars=list(args)[1]

        coefficients1=[pars[0][1],pars[1][1],pars[2][1]]
        coefficients2=[pars[0][2],pars[1][2],pars[2][2]]
        radii=[pars[0][0],pars[1][0],pars[2][0]]
        #try:

        y0= np.multiply( radii[0] , np.sqrt(np.divide(np.subtract(np.absolute(x[1]),np.multiply(np.absolute(x[0]),coefficients2[0])),np.add(np.ones(3),np.multiply(np.absolute(x[2]),coefficients1[0])))) )
        y1= np.multiply( radii[1] , np.sqrt(np.divide(np.subtract(np.absolute(x[1]),np.multiply(np.absolute(x[0]),coefficients2[1])),np.add(np.ones(3),np.multiply(np.absolute(x[2]),coefficients1[1])))) )
        y2= np.multiply( radii[2] , np.sqrt(np.divide(np.subtract(np.absolute(x[1]),np.multiply(np.absolute(x[0]),coefficients2[2])),np.add(np.ones(3),np.multiply(np.absolute(x[2]),coefficients1[2])))) )
        Y0= 1. + np.sum( np.multiply( y0, sign[0] ) )
        Y1= 1. + np.sum( np.multiply( y1, sign[1] ) )
        Y2= 1. + np.sum( np.multiply( y2, sign[2] ) )
        # except:
        #     Y0=np.inf
        #     Y1=np.inf
        #     Y2=np.inf
        return np.sum(np.power([Y0,Y1,Y2],2))

    def murray_coefficients_min( self ,x, *args ):

        sign=list(args)[0]
        pars=list(args)[1]

        # coefficients1=[pars[0][1],pars[1][1],pars[2][1]]
        # coefficients2=[pars[0][2],pars[1][2],pars[2][2]]
        # radii=[pars[0][0],pars[1][0],pars[2][0]]
        radii=pars[0][0]
        coefficients1=pars[0][1]
        coefficients2=pars[0][2]

        try:

            y0= np.multiply( radii , np.sqrt(np.divide(np.subtract(np.absolute(x[1]),np.multiply(np.absolute(x[0]),coefficients2)),np.add(np.ones(3),np.multiply(np.absolute(x[2]),coefficients1)))) )
            # y1= np.multiply( radii[1] , np.sqrt(np.divide(np.subtract(np.absolute(x[1]),np.multiply(np.absolute(x[0]),coefficients2[1])),np.add(np.ones(3),np.multiply(np.absolute(x[2]),coefficients1[1])))) )
            # y2= np.multiply( radii[2] , np.sqrt(np.divide(np.subtract(np.absolute(x[1]),np.multiply(np.absolute(x[0]),coefficients2[2])),np.add(np.ones(3),np.multiply(np.absolute(x[2]),coefficients1[2])))) )
            Y0=1. + np.sum( np.multiply( y0, sign ) )
        # Y1=1. + np.sum( np.multiply( y1, sign ) )
        # Y2=1. + np.sum( np.multiply( y2, sign ) )

        # return np.linalg.norm([Y0,Y1,Y2])
        except:
            Y0=np.inf

        return np.power(Y0,2)

    def murray_coefficients( self ,x, *args ):

        sign=list(args)[0]
        pars=list(args)[1]

        coefficients1=[pars[0][1],pars[1][1]]
        radii=[pars[0][0],pars[1][0]]

        y0= np.multiply( radii[0] , np.sqrt(np.divide(np.absolute(x[0]),np.add(np.ones(3),np.multiply(np.absolute(x[1]),coefficients1[0])))) )
        y1= np.multiply( radii[1] , np.sqrt(np.divide(np.absolute(x[0]),np.add(np.ones(3),np.multiply(np.absolute(x[1]),coefficients1[1])))) )

        Y0=1. + np.sum( np.multiply( y0, sign ) )
        Y1=1. + np.sum( np.multiply( y1, sign ) )

        return [Y0,Y1]

    def fit_murray( self,  *args):

        # check solvability
        if np.sum(args)==len(args):
            print('no fit reasonable check x,y')
            return 0
        # find upper limit and readjust lower limit accordingly
        a_bottom=0.
        a_top=3.
        iterations=10
        below_tolerance=True

        for i in range(iterations):
            if self.murray_law(*(list(args) + [a_top])) < 0:
                a_bottom=1.*a_top
                a_top=2.*a_top
            else:
                break
            if i==iterations-1:
                print('cant find upper limit')
                below_tolerance=False
                a=-1
        while below_tolerance:
            a=(a_top+a_bottom)/2.
            z=self.murray_law( *(list(args) + [a] ))

            if z * z <= self.tol:
                below_tolerance=False
                break
            if z < 0:
                a_bottom=1.*a
                # a=(a_top+a_bottom)/2.
            elif z > 0:
                a_top=1.*a
                # a=(a_top+a_bottom)/2.
            if z==0:
                break
        return a

    # fit multi parameter murray law
    def fit_murray_kirchhoff_flux_3D( self, tup):

        # cons = sc.NonlinearConstraint(self.larger_zero, [0.,0.,0.], [np.inf,np.inf], keep_feasible=True)
        # sol=sc.optimize.minimize(self.murray_coefficients_fluc,x0=[10.],args=tuple(x),constraints=cons)
        # sol=sc.optimize.minimize(self.murray_coefficients_fluc,x0=[10000.,10.],args=tup,bounds=((0.,np.power(10.,8)),(0.,100)))
        # signs=[[1.,1.,-1.],[1.,-1.,-1.],[-1.,-1.,-1.],[-1.,1.,-1.],[-1.,-1.,1.],[-1.,1.,1.],[1.,-1.,1.]]
        # signs=tup[0]
        # pars=tup[1:]
        signs=tup[0]
        pars=tup[1:]

        cons_tuple=[]
        for i in range(3):
            for j in range(3):
                d={'type':'ineq','fun':self.larger_zero,'args':(i,pars[j])}
                cons_tuple.append(d)
        cons=[{'type':'ineq','fun':self.larger_zero,'args':(0,pars)},{'type':'ineq','fun':self.larger_zero,'args':(1,pars)},{'type':'ineq','fun':self.larger_zero,'args':(2,pars)}]
        solutions=[]
        failed=[]

        X=self.smart_start(pars)
        for s in signs:
            # sol=sc.optimize.root(self.murray_coefficients_root,x0=X,args=(s,tup))
            # sol=sc.optimize.least_squares(self.murray_coefficients_root,x0=X,args=(s,tup),bounds=[0.,np.power(10.,20)])
            # sol=sc.optimize.minimize(self.murray_coefficients_tuple_jac,x0=X,method='COBYLA',options={'maxiter':10000},args=(s,pars),constraints=cons_tuple)
            # sol=sc.optimize.minimize(self.murray_coefficients_tuple_min,x0=X,jac=self.murray_coefficients_tuple_jac,args=(s,pars),method='Newton-CG')
            sol=sc.optimize.root(self.murray_coefficients_root_jac,x0=X,args=(s,pars),jac=True)
            # sol=sc.optimize.minimize(self.murray_coefficients_min,x0=X,method='COBYLA',options={'maxiter':10000},args=(s,pars),constraints=cons)
            # res=sc.optimize.basinhopping(self.murray_coefficients_tuple_min,x0=X,niter=100,minimizer_kwargs = {"method":"COBYLA","constraints":cons,'args':(tup[0],tup[1,2,3])})
            # res=sc.optimize.basinhopping(self.murray_coefficients_min,x0=X,niter=100,minimizer_kwargs = {"method":"COBYLA","constraints":cons,'args':(s,pars)})
            # sol=res.lowest_optimization_result
            # minimizer_kwargs = {"method":"L-BFGS-B", "jac":True, "args":(s,pars)}
            # res=sc.optimize.basinhopping(self.murray_coefficients_tuple_jac,x0=X,niter=20,minimizer_kwargs = minimizer_kwargs)
            # sol=res.lowest_optimization_result

            if sol.success:
                solutions.append(sol)

            else:
                failed.append(sol.message)

        if len(solutions)>0:
            idx=np.argmin([ np.linalg.norm( s.fun) for s in solutions ])
            # print(solutisons[idx].fun)
            return solutions[idx].x

        else:
            return[np.nan,np.nan,np.nan]

        return 0

    def fit_murray_kirchhoff_flux_2D( self,  tup):

        signs=[[1.,1.,-1.],[1.,-1.,-1.],[-1.,-1.,-1.],[-1.,1.,-1.],[-1.,-1.,1.],[-1.,1.,1.],[1.,-1.,1.]]
        # signs=[[1.,1.,-1.]]

        solutions=[]
        failed=[]
        for s in signs:
            sol=sc.optimize.root(self.murray_coefficients,x0=[10.,10.],args=(s,tup))
            if sol.success:
                solutions.append(sol)
            else:
                failed.append(sol.message)

        if len(solutions)>0:
            idx=np.argmin([ np.linalg.norm( s.fun) for s in solutions ])
            return solutions[idx].x

        else:
            print(failed)
            return[np.nan,np.nan]

        return 0

    def smart_start(self,tup):

        # x0=[1.,1.,1.]
        c1=[]
        c3=[]
        for t in tup:
            c1.append(np.amax(t[2]))
            c3.append(np.amax(t[1]))
        scale=np.amax(c1)

        if scale ==0:
            scale=1.

        local_state = np.random.RandomState()
        r=local_state.rand(3)
        x0=np.multiply([0.,scale*100,np.amax(c3)],r)

        return x0

    def larger_zero(self,x,*args):

        idx=list(args)[0]
        pars=list(args)[1]
        coefficients2=pars[2]

        # coefficients2=pars[0][2]
        # print(coefficients2)
        return np.subtract(np.absolute(x[1]),np.multiply(np.absolute(x[0]),coefficients2[idx]))

    def calc_scaling_kirchhoff_flux_parameters(self,j, B,exp):

        K=B.layer[j]
        K.set_structure_coefficient_fluctuation()
        B.set_structure_coefficient_coupling(exp)
        G=K.H

        list_deg=[]
        radii_tuple=[]
        murray_pars_0=[]
        murray_pars_1=[]
        murray_pars_2=[]
        list_nodes=list(G.nodes())
        for n in list_nodes:
            if G.degree[n] >=3:
                list_deg.append(n)
        for i,n in enumerate(list_nodes):
            if n in list_deg:
                r={}
                radii=[]
                # sign=[]
                n_edges=list(G.edges(n))
                # aux=nx.Graph(n_edges)
                for e in n_edges:
                    r[e]=( G.edges[e]['conductivity']**0.25 )

                list_sorted=sorted(r,key=r.__getitem__)
                radii = [ r[e] for e in list_sorted ]

                coefficients_l3 = [ G.edges[e]['coefficient_l3']  for e in list_sorted ]
                coefficients_l1 = [ G.edges[e]['coefficient_l1']  for e in list_sorted ]

                radii_tuple.append([np.power(radii,3),coefficients_l3,coefficients_l1])

        signs=[[1.,1.,-1.],[1.,-1.,-1.],[-1.,-1.,-1.],[-1.,1.,-1.],[-1.,-1.,1.],[-1.,1.,1.],[1.,-1.,1.]]
        super_tup=[]
        # multiplicity=20
        # for m in range(multiplicity):
        #     for i,r in enumerate(radii_tuple):
        #         super_tup.append([signs,r])
        check=[]
        for i in range(len(radii_tuple)):
            for j in range(len(radii_tuple)):
                for k in range(len(radii_tuple)):
                    if k>j and j>i:
                        check.append([i,j,k])
        S=[]
        for i in range(7):
            for j in range(7):
                for k in range(7):
                    if k>j and j>i:
                        S.append([signs[i],signs[j],signs[k]])

        for i,c in enumerate(check):

                tup1=radii_tuple[c[0]]
                tup2=radii_tuple[c[1]]
                tup3=radii_tuple[c[2]]
                super_tup.append([S,tup1,tup2,tup3])


        pool = mp.Pool(processes=4)
        pars=pool.map( self.fit_murray_kirchhoff_flux_3D, super_tup )
        for p in pars:
            if not (np.isnan(p[0]) or np.isnan(p[1]) or np.isnan(p[2])):
                murray_pars_0.append(np.absolute(p[0]))
                murray_pars_1.append(np.absolute(p[1]))
                murray_pars_2.append(np.absolute(p[2]))

        # check=[]
        # for i in range(len(radii_tuple)):
        #     for j in range(len(radii_tuple)):
        #         for k in range(len(radii_tuple)):
        #             if k>j and j>i:
        #                 check.append([i,j,k])
        # for i,tup1 in enumerate(radii_tuple):
        #     for j,tup2 in enumerate(radii_tuple):
        #         for k,tup3 in enumerate(radii_tuple):
        #             if [i,j,k] in check:
        #
        #                 pars=self.fit_murray_kirchhoff_flux_parameters([tup1,tup2,tup3])
        #                 if not (np.isnan(pars[0]) or np.isnan(pars[1]) or np.isnan(pars[2])):
        #                     murray_pars_0.append(np.absolute(pars[0]))
        #                     murray_pars_1.append(np.absolute(pars[1]))
        #                     murray_pars_2.append(np.absolute(pars[2]))

        return murray_pars_0,murray_pars_1,murray_pars_2

    def calc_scaling_kirchhoff_fluctuation(self,j, B,exp):

        K=B.layer[j]
        K.set_structure_coefficient_fluctuation()
        B.set_structure_coefficient_repulsion(exp)
        G=K.H

        list_deg=[]
        radii_tuple=[]
        murray_pars_0=[]
        murray_pars_1=[]
        list_nodes=list(G.nodes())

        for n in list_nodes:
            if G.degree[n] >=3 and G.node[n]['source']<0:
                list_deg.append(n)
        for i,n in enumerate(list_nodes):
            if n in list_deg:
                r={}
                radii=[]
                n_edges=list(G.edges(n))
                for e in n_edges:
                    r[e]=( G.edges[e]['conductivity']**0.25 )

                list_sorted=sorted(r,key=r.__getitem__)
                radii = [ r[e] for e in list_sorted ]

                coefficients_l3 = [ G.edges[e]['coefficient_l3']  for e in list_sorted ]
                radii_tuple.append([np.power(radii,3),coefficients_l3])

        check=[]
        for i in range(len(radii_tuple)):
            for j in range(len(radii_tuple)):
                    if j>i:
                        check.append([i,j])
        for i,tup1 in enumerate(radii_tuple):
            for j,tup2 in enumerate(radii_tuple):
                    if [i,j] in check:

                        pars=self.fit_murray_kirchhoff_fluctuation([tup1,tup2])
                        if not (np.isnan(pars[0]) or np.isnan(pars[1])) :

                            murray_pars_0.append(np.absolute(pars[0]))
                            murray_pars_1.append(np.absolute(pars[1]))

        return murray_pars_0,murray_pars_1

    # contraction algorithms
    def relabel_neighbors(self,total_cycle_list,list_c,e0,e1,label,loop_cond):
        for other_c in list_c:
            # contract neighbor loops as well when found in the cycle as well and label is matching

            if total_cycle_list[other_c].has_edge(e0,e1):
                CC_other=nx.contracted_edge(total_cycle_list[other_c],(e0,e1))
                label_e=nx.get_edge_attributes(CC_other,'label')
                if len(self.getKeysByValues(label_e, [label]))!=0:
                    CC_other.remove_edge(*self.getKeysByValues(label_e, [label])[0])
                total_cycle_list[other_c]=CC_other

           # keep track of node relabelling,  True if loops share an edge and it's not a dangeling self-loop
            elif loop_cond:
                aux_list_e=list(total_cycle_list[other_c].edges(keys=True))
                for edge in aux_list_e :
                    if edge[0]==e1:
                        total_cycle_list[other_c].add_edge(edge[1],e0,label=total_cycle_list[other_c].edges[edge]['label'])

                    elif edge[1]==e1:

                        total_cycle_list[other_c].add_edge(edge[0],e0,label=total_cycle_list[other_c].edges[edge]['label'])

                if total_cycle_list[other_c].has_node(e1):
                    total_cycle_list[other_c].remove_node(e1)
            # print(total_cycle_list[other_c].edges.data('label',keys=True))
        return total_cycle_list

    def getKeysByValues(self,dict_e, values):
        keys=[]
        listOfItems = dict_e.items()
        for item in listOfItems:
            if item[1] in values:
               keys.append(item[0])
        return  keys

    def view_pydot(self,CG):
        pydot_graph = nx.nx_pydot.to_pydot(CG)
        plt = Image(pydot_graph.create_png())
        display(plt)

    def calc_betty_number(self,SG,cycle_subset,total_cycle_list):

        minimum_basis=[]
        EC=nx.MultiGraph()
        counter=0
        for c in cycle_subset:

            cycle_edges_in_basis=True

            for e in total_cycle_list[c].edges(keys=True):
                if not EC.has_edge(*e):
                    EC.add_edge(*e,label=counter)
                    counter+=1
                    cycle_edges_in_basis=False
            #if cycle edges where not part of the supergraph yet then it becomes automatically part of the basis
            if not cycle_edges_in_basis:
                minimum_basis.append(total_cycle_list[c])
            #if cycle edges are already included we check for linear dependece
            else:
                linear_independent=False
                rows=len(list(EC.edges()))
                columns=len(minimum_basis)+1
                E=np.zeros((rows,columns))
                # translate the existent basis vectors into z2 representation
                for idx_c,cycle in enumerate(minimum_basis+[total_cycle_list[c]]):
                    for m in cycle.edges(keys=True):
                        if EC.has_edge(*m):
                            E[EC.edges[m]['label'],idx_c]=1

                # calc echelon form
                a_columns=np.arange(columns-1)
                zwo=np.ones(columns)*2
                for column in a_columns:
                    idx_nz=np.nonzero(E[column:,column])[0]
                    if idx_nz.size:
                        if len(idx_nz)==1:
                            E[column,:],E[idx_nz[0]+column,:]=E[idx_nz[0]+column,:].copy(),E[column,:].copy()
                        else:
                            for r in idx_nz[1:]:
                                aux_E=np.add(E[r+column],E[idx_nz[0]+column])
                                E[r+column]=np.mod(aux_E,zwo)
                            E[column,:],E[idx_nz[0]+column,:]=E[idx_nz[0]+column,:].copy(),E[column,:].copy()
                    else:
                        sys.exit('Error: minimum_weight_basis containing inconsistencies ...')
                # test echelon form for inconsistencies
                for r in range(rows):
                    line_check=np.nonzero(E[r])[0]
                    if len(line_check)==1 and line_check[0]==(columns-1):
                        linear_independent=True
                        break
                if linear_independent:
                    minimum_basis.append(total_cycle_list[c])

        b2=len(cycle_subset)-len(minimum_basis)
        b1=nx.number_of_edges(SG) - nx.number_of_nodes(SG) + 1 -len(minimum_basis)
        b0=nx.number_connected_components(SG)
        return b0,b1,b2

    def generate_cycle_subset(self,SG,root):
        total_cycle_dict={}
        total_cycle_list=[]
        super_list=[]

        counter=0

        TR=nx.MultiGraph(self.breadth_first_tree(SG,root))

        D=nx.difference(SG,TR)
        for e in D.edges(keys=True):
            p=nx.shortest_path(TR,source=e[1],target=e[0])

            # label pathways
            simple_cycle=nx.MultiGraph(cycle_weight=0.)
            nx.add_path(simple_cycle,p)
            simple_cycle.add_edge(*e)

            if nx.is_eulerian(simple_cycle):
                # relabeling and weighting graph
                for m in simple_cycle.edges(keys=True):
                    simple_cycle.edges[m]['label']=SG.edges[m]['label']
                simple_cycle.edges[e]['label']=SG.edges[e]['label']
                total_cycle_list.append(simple_cycle)
                total_cycle_dict.update({counter:nx.number_of_edges(simple_cycle)})
                counter+=1

        return total_cycle_dict, total_cycle_list, super_list

    def contract_subset(self,SG,cycle_subset,total_cycle_list):

        CG=nx.MultiGraph(SG)

        for j,c in enumerate(cycle_subset):

            list_e=list(total_cycle_list[c].edges(keys=True))
            list_n=list(total_cycle_list[c].nodes())
            list_c=[c_other for c_other in cycle_subset if c_other!=c]
            CC=total_cycle_list[c]
            cond=True
            if len(list_e)==0:
                cond=False
            while cond:
                e1=list_e[0][1]
                e0=list_e[0][0]
                e=list_e[0]
                # contract all loop edges

                edge_label=CC.edges[e]['label']
                CC=nx.contracted_edge(total_cycle_list[c],(e0,e1))
                c_label_e=nx.get_edge_attributes(CC,'label')
                CC.remove_edge(*self.getKeysByValues(c_label_e, [edge_label])[0])

                CG=nx.contracted_edge(CG,(e0,e1))
                label_e=nx.get_edge_attributes(CG,'label')
                CG.remove_edge(*self.getKeysByValues(label_e, [edge_label])[0])

                total_cycle_list[c]=CC
                list_e=list(CC.edges(keys=True))

                #relabel neighbors
                loop_cond=True
                if e0==e1:
                    loop_cond=False

                total_cycle_list=self.relabel_neighbors(total_cycle_list,list_c,e0,e1,edge_label,loop_cond)
                # view_pydot(CG)
                if len(list_e)==0:
                    cond=False

            # view_pydot(CG)
        SG=nx.MultiGraph()
        dict_nodes={}
        for idx_n,n in enumerate(CG.nodes()):
            SG.add_node(idx_n)
            dict_nodes.update({n:idx_n})
        for idx_e,e in enumerate(CG.edges(keys=True)):
            SG.add_edge(dict_nodes[e[0]],dict_nodes[e[1]],weight=1.,label=CG.edges[e]['label'])

        return SG
