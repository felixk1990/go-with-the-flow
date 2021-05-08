import networkx as nx
import numpy as np
import sys
import os
import os.path as op
import shutil
import datetime
import random as rd
import pandas as pd
import performance
import analyze_graph
# from scipy.optimize import curve_fit
import scipy.optimize as so
import scipy.spatial as scs
import scipy.linalg as lina
import multiprocessing as mp
# np.set_printoptions(threshold=sys.maxsize)
#new test
class tool_liver(analyze_graph.tool_box,object):

    def __init__(self):
        super(tool_liver,self).__init__()
        self.num_chi_intveralls = 1
        self.time_stamp=''
        self.branching_mode=''
        self.chi_hist=10
        self.counter_n=0
        self.counter_e=0
        self.reference_graph=nx.Graph()

    def analyze_graph(self,G):
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
            M=self.make_it_simple(G)
            cycle_basis=self.construct_minimum_basis(M)
            CT,A=self.calc_cycle_coalescence(M,cycle_basis)

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
        if self.SCALING:
            print('...Caculating branch scaling...')
            if self.branching_mode=='chi':
                n,d=self.calc_scaling_chi(G)
            else:
                n,d=self.calc_scaling(G)
            self.list_scaling.append(n)
            self.list_delta.append(d)

            print('...Done...')

        if self.CYCLE_NEMATIC:

            print('...Caculating cycle nematic order...')
            M=self.make_it_simple(G)
            cycle_basis=self.construct_minimum_basis(M)
            p1,p2,d,centers = self.cycle_nematic_order(M,cycle_basis)

            self.aux_list1.append(p1)
            self.aux_list2.append(p2)
            self.aux_list3.append(d)
            self.aux_list4.append(centers)

            print('...Done...')

    def liver_extract_data(self,file_nodes,file_edges,mode):

        node_data=[]
        edge_data=[]
        # try:
        # data_table_nodes=pd.read_csv(op.join(self.input_dir,file_nodes),delimiter=',',na_values='-1.#IND00', index_col=False).values
        data_table_nodes=pd.read_csv(op.join(self.input_dir,file_nodes),delimiter=';',na_values='-1.#IND00', index_col=False).values
        data_table_edges=pd.read_csv(op.join(self.input_dir,file_edges),delimiter=',',na_values='-1.#IND00', index_col=False).values

        node_index=data_table_nodes[:,:2]
        node_points=data_table_nodes[:,2:5]
        node_radius=data_table_nodes[:,6]
        # node_radius=data_table_nodes[:,5]
        node_chi=data_table_nodes[:,5]
        # print(node_chi)
        # node_degree=data_table_nodes[:,6]

        # handle segmentation artifacts
        # for i,boolean in enumerate(np.isnan(node_radius[:])):
        #     if boolean:
        #
        #         node_radius[i]=2.
        #     elif node_radius[i]==0.:
        #
        #         node_radius[i]=2.

        edge_index=data_table_edges[:,0:3]
        # edge_length=data_table_edges[:,3]
        #find all components
        if 'all' in mode:
            edge_index_aux=[]
            # edge_length_aux=[]

            # node_data=[node_index,node_points,node_radius,node_degree,node_chi]
            # node_data=[node_index,node_points]
            # node_data=[node_index,node_points,node_radius]
            node_data=[node_index,node_points,node_chi,node_radius]

            dict_nodes={}

            for i,idx_n in enumerate(node_index[:,0]):

                if idx_n not in dict_nodes.keys():
                    list_n=[node_index[i,1]]
                    dict_nodes.update({idx_n:list_n})
                else:
                    dict_nodes[idx_n].append(node_index[i,1])

            for i,e_idx in enumerate(edge_index):

                if e_idx[0] in dict_nodes.keys():

                    if np.any( dict_nodes[e_idx[0]] == e_idx[1]) and np.any( dict_nodes[e_idx[0]] == e_idx[2]):
                        edge_index_aux.append(e_idx)
                    # edge_length_aux.append(edge_length[i])

            edge_data=[edge_index_aux]
            # edge_data=[edge_index_aux,edge_length_aux]

        # find largest component
        elif 'max' in mode:

            # find largest component
            counter=0
            idx=0
            idx_list=[]

            for i in node_index[:,0]:
                if i > idx:
                    idx_list.append(counter)
                    idx+=1
                    counter=1
                else:
                    counter+=1
            idx_max=np.argmax(np.asarray(idx_list))

            # isolate largest component
            # node_degree_branch_max=[]
            node_index_max=[]
            node_points_max=[]
            node_radius_max=[]
            # node_degree_max=[]
            node_chi_max=[]
            edge_index_max=[]
            # edge_length_max=[]

            for i, n_idx in enumerate(node_index):

                if node_index[i,0] == idx_max:

                    node_index_max.append(n_idx)
                    node_points_max.append(node_points[i])
                    node_radius_max.append(node_radius[i])
                    # node_degree_max.append(node_degree[i])
                    if '-chi' in mode:

                        node_chi_max.append(node_chi[i])

            nodes = np.asarray(node_index_max)[:,1]

            for i,e_idx in enumerate(edge_index):
                if edge_index[i,0] == idx_max:

                    # if np.any(nodes == edge_index[i,1]) and np.any(nodes == edge_index[i,2]):
                    edge_index_max.append(e_idx)
                    # edge_length_max.append(edge_length[i])

            # node_data=[node_index_max,node_points_max,node_radius_max,node_degree_max,node_chi_max]
            # node_data=[node_index_max,node_points_max]
            node_data=[node_index_max,node_points_max,node_chi_max,node_radius_max]
            # node_data=[node_index_max,node_points_max,node_chi_max]
            edge_data=[edge_index_max]
            # edge_data=[edge_index_max,edge_length_max]
            # print(edge_data)
        else:
            print('Error: No suitable import mode provided!')

        return node_data, edge_data

    def count_n(self):
        self.counter_n+=1
        return self.counter_n

    def count_e(self):
        self.counter_e+=1
        return self.counter_e

    def liver_construct_network(self,file_nodes,file_edges,mode):
        # load experimental data and construct network from it
        cond=True
        node_data,edge_data=self.liver_extract_data(file_nodes,file_edges,mode)


        node_index=np.asarray(node_data[0])
        node_points=np.asarray(node_data[1])
        node_chi=np.asarray(node_data[2])
        node_radius=np.asarray(node_data[3])

        edge_index=np.asarray(edge_data[0])
        # edge_length=np.asarray(edge_data[1])

        C=node_chi
        # print(self.num_chi_intveralls)
        # if '-chi' in mode:
        #     C=[ int((chi*self.num_chi_intveralls)/1000.)  for chi in node_chi
        G=nx.Graph()
        for n,pos_n in enumerate(node_points):

            if '-chi' in mode:
                G.add_node(tuple(node_index[n]),pos=pos_n,chi=C[n],label=self.count_n(),radius=node_radius[n])
            else:
                G.add_node(tuple(node_index[n]),pos=pos_n,radius=node_radius[n],label=self.count_n())
            # G.add_node(tuple(node_index[n]),pos=pos_n,radius=node_radius[n],chi=C[n],label=self.count_n())
        for e,idx_e in enumerate(edge_index):
            n1=tuple((idx_e[0],idx_e[1]))
            n2=tuple((idx_e[0],idx_e[2]))
            cond=True
            # if '-chi' in mode:
            #     if G.nodes[n1]['chi']==G.nodes[n2]['chi']:
            #         cond=True
            #     else:
            #         cond=False
            if cond:
                G.add_edge(n1,n2,slope=(G.nodes[n1]['pos'],G.nodes[n2]['pos']),label=self.count_e(),length=np.linalg.norm(np.subtract(G.nodes[n1]['pos'],G.nodes[n2]['pos'])))
        # relabel
        # dict_nodes={}
        # G_fin=nx.Graph()
        # for idx_n,n in enumerate(G.nodes()):
        #     # G_fin.add_node(idx_n,pos=G.nodes[n]['pos'],radius=G.nodes[n]['radius'],chi=G.nodes[n]['chi'],label=G.nodes[n]['label'])
        #     if '-chi' in mode:
        #         G_fin.add_node(idx_n,pos=G.nodes[n]['pos'],label=G.nodes[n]['label'],chi=G.nodes[n]['chi'],radius=G.nodes[n]['radius'])
        #     else:
        #         G_fin.add_node(idx_n,pos=G.nodes[n]['pos'],label=G.nodes[n]['label'],radius=G.nodes[n]['radius'])
        #     # G_fin.add_node(idx_n,pos=G.nodes[n]['pos'],label=G.nodes[n]['label'])
        #     dict_nodes.update({n:idx_n})
        #
        # for idx_e,e in enumerate(G.edges()):
        #     G_fin.add_edge(dict_nodes[e[0]],dict_nodes[e[1]],slope=(G.nodes[e[0]]['pos'],G.nodes[e[1]]['pos']),label=G.edges[e]['label'],length=G.edges[e]['length'])

        G_fin=nx.convert_node_labels_to_integers(G, first_label=0, ordering='default')
        nx.set_node_attributes(G_fin,0,'source')

        return G_fin

    def liver_randomize_network(self,G):

        RG=nx.MultiGraph(G)
        weights=list(nx.get_edge_attributes(G,'weight').values())
        w_max=np.amax(weights)
        w_min=np.amin(weights)

        for e in RG.edges(keys=True):
            RG.edges[e]['weight']=rd.uniform(w_min,w_max)

        return RG

    def liver_coarsegrain_network(self,G,mode):

        # take given network and smooth it, so only branching nodes do survive, cut all 'loose' edges
        list_nodes=[]
        SG=nx.MultiGraph(G)

        # weight umerged edges, count mutiples
        if '-geometric' in mode:


            for e in SG.edges(keys=True):
                SG.edges[e]['weight']=0.
                SG.edges[e]['origin']=[]

            # merge pathways which include nodes with degree 2, edges get designated a length as weight
            while True:
                labels_n = nx.get_node_attributes(SG,'label')
                sorted_label_n_list=sorted(labels_n ,key=labels_n.__getitem__)
                cond1= self.get_loop_condition(SG)

                while cond1:
                    SG=self.liver_merge_series(SG,sorted_label_n_list,mode)

                    if '-noloop' in mode:
                        list_sl=list(nx.selfloop_edges(SG))
                        if list_sl:
                            for sl in list_sl:
                                SG.remove_edge(*sl)
                    labels_n = nx.get_node_attributes(SG,'label')
                    sorted_label_n_list=sorted(labels_n ,key=labels_n.__getitem__)
                    cond1= self.get_loop_condition(SG)

                if '-noleaves' in mode:
                    labels_n = nx.get_node_attributes(SG,'label')
                    sorted_label_n_list=sorted(labels_n ,key=labels_n.__getitem__)
                    cond2=True
                    while cond2:

                        for n in sorted_label_n_list:
                            # print(n)
                            if SG.degree(n) == 0:

                                SG.remove_node(n)

                            elif SG.degree(n) == 1:
                                e=list(SG.edges(n,keys=True))
                                SG.remove_edge(*e[0])
                                SG.remove_node(n)

                        labels_n = nx.get_node_attributes(SG,'label')
                        sorted_label_n_list=sorted(labels_n ,key=labels_n.__getitem__)
                        list_degrees=np.array(SG.degree(sorted_label_n_list))
                        cond2=np.any( list_degrees[:,1] <=1 )

                cond1= self.get_loop_condition(SG)
                if not cond1:
                    break
        if '-topological' in mode:

            nx.set_node_attributes(SG,'true','identity')
            for e in SG.edges(keys=True):
                SG.edges[e]['weight']=1.
                SG.edges[e]['origin']=[]

            # merge pathways which include nodes with degree 2, edges get designated a length as weight
            while True:
                labels_n = nx.get_node_attributes(SG,'label')
                sorted_label_n_list=sorted(labels_n ,key=labels_n.__getitem__)
                cond1= self.get_loop_condition(SG)

                while cond1:
                    SG=self.liver_merge_series(SG,sorted_label_n_list,mode)

                    if '-noloop' in mode:
                        list_sl=list(nx.selfloop_edges(SG))
                        if list_sl:
                            for sl in list_sl:
                                SG.remove_edge(*sl)
                    labels_n = nx.get_node_attributes(SG,'label')
                    sorted_label_n_list=sorted(labels_n ,key=labels_n.__getitem__)
                    cond1= self.get_loop_condition(SG)

                if '-noleaves' in mode:
                    labels_n = nx.get_node_attributes(SG,'label')
                    sorted_label_n_list=sorted(labels_n ,key=labels_n.__getitem__)
                    cond2=True
                    while cond2:

                        for n in sorted_label_n_list:
                            # print(n)
                            if SG.degree(n) == 0:

                                SG.remove_node(n)

                            elif SG.degree(n) == 1:
                                e=list(SG.edges(n,keys=True))
                                SG.remove_edge(*e[0])
                                SG.remove_node(n)

                        labels_n = nx.get_node_attributes(SG,'label')
                        sorted_label_n_list=sorted(labels_n ,key=labels_n.__getitem__)
                        list_degrees=np.array(SG.degree(sorted_label_n_list))
                        cond2=np.any( list_degrees[:,1] <=1 )

                cond1= self.get_loop_condition(SG)
                if not cond1:
                    break
        counter=0
        if '-conductance' in mode:

            for e in SG.edges(keys=True):

                r=(SG.nodes[e[0]]['radius']+SG.nodes[e[1]]['radius'])/2.
                l=SG.edges[e]['length']
                # l=1.
                C=(r**4)/l
                SG.edges[e]['radius']=r
                SG.edges[e]['weight']=C
                SG.edges[e]['origin']=[]

            # merge pathways which include nodes with degree 2, edges get designated a mean conductivity
            while True:
                labels_n = nx.get_node_attributes(SG,'label')
                sorted_label_n_list=sorted(labels_n ,key=labels_n.__getitem__)
                cond1= self.get_loop_condition(SG)

                while cond1:
                    SG=self.liver_merge_series(SG,sorted_label_n_list,mode)

                    if '-noloop' in mode:
                        list_sl=list(nx.selfloop_edges(SG))
                        if list_sl:
                            for sl in list_sl:
                                SG.remove_edge(*sl)
                    labels_n = nx.get_node_attributes(SG,'label')
                    sorted_label_n_list=sorted(labels_n ,key=labels_n.__getitem__)
                    cond1= self.get_loop_condition(SG)

                if '-noleaves' in mode:
                    labels_n = nx.get_node_attributes(SG,'label')
                    sorted_label_n_list=sorted(labels_n ,key=labels_n.__getitem__)
                    cond2=True
                    while cond2:

                        for n in sorted_label_n_list:
                            # print(n)
                            if SG.degree(n) == 0:

                                SG.remove_node(n)

                            elif SG.degree(n) == 1:
                                e=list(SG.edges(n,keys=True))
                                SG.remove_edge(*e[0])
                                SG.remove_node(n)

                        labels_n = nx.get_node_attributes(SG,'label')
                        sorted_label_n_list=sorted(labels_n ,key=labels_n.__getitem__)
                        list_degrees=np.array(SG.degree(sorted_label_n_list))
                        cond2=np.any( list_degrees[:,1] <=1 )

                if '-noparallel' in mode:

                    list_edges=np.array(list(SG.edges(keys=True)))
                    cond3=np.any( list_edges[:,2]  > 0 )

                    while cond3:
                        seen={}
                        # list_edges_nokey=list(SG.edges())
                        dict_labels_e = nx.get_edge_attributes(SG,'label')
                        sorted_label_e_list=sorted(dict_labels_e,key=dict_labels_e.__getitem__)

                        for edge in sorted_label_e_list:
                            e=(edge[0],edge[1])
                            num_conncetions=SG.number_of_edges(*e)
                            if num_conncetions >1 and e not in seen.keys():
                                seen[e]=0
                                par_weight=0.
                                origins=[]
                                for i in range(num_conncetions):

                                    par_weight+=SG.edges[(e[0],e[1],i)]['weight']

                                    origins+=SG.edges[(e[0],e[1],i)]['origin']
                                    SG.remove_edge(e[0],e[1],i)


                                SG.add_edge(e[0],e[1],slope=(SG.nodes[e[0]]['pos'],SG.nodes[e[1]]['pos']),weight=par_weight,label=self.count_e(),origin=origins,length=np.linalg.norm(np.subtract(SG.nodes[e[0]]['pos'],SG.nodes[e[1]]['pos'] )))
                        list_edges=np.array(list(SG.edges(keys=True)))

                        cond3=np.any( list_edges[:,2]  > 0 )

                cond1= self.get_loop_condition(SG)
                if not cond1:
                    break

        # relabel/count the nodes and edges
        SG_fin=nx.convert_node_labels_to_integers(SG, first_label=0, ordering='default')

        return SG_fin

    def liver_merge_series(self,SG,list_nodes,mode):
        # adding further merge modii here
        LS=0
        CS=0

        for n in list_nodes:
            e_merged=[]
            e=list(SG.edges(n,keys=True))

            if SG.degree(n) >= 3:
                continue

            elif SG.degree(n) == 2 and len(e)==2:

                if SG.nodes[n]['source'] <=0 :
                    if e[0][0] != n:
                        e_merged.append(e[0][0])
                    elif e[0][1] != n:
                        e_merged.append(e[0][1])
                    if e[1][0] != n:
                        e_merged.append(e[1][0])
                    elif e[1][1] != n:
                        e_merged.append(e[1][1])

                    if '-geometric' in mode:

                        l=SG.edges[e[0]]['length']+SG.edges[e[1]]['length']
                        SG.add_edge(e_merged[0],e_merged[1],slope=(SG.nodes[e_merged[0]]['pos'],SG.nodes[e_merged[1]]['pos']),weight=LS,origin=[n]+SG.edges[e[0]]['origin']+SG.edges[e[1]]['origin'],label=self.count_e(),length=l)
                    if '-topological' in mode:
                        l=SG.edges[e[0]]['length']+SG.edges[e[1]]['length']
                        SG.add_edge(e_merged[0],e_merged[1],slope=(SG.nodes[e_merged[0]]['pos'],SG.nodes[e_merged[1]]['pos']),weight=1.,origin=[n]+SG.edges[e[0]]['origin']+SG.edges[e[1]]['origin'],label=self.count_e(),length=l)
                    if '-conductance' in mode:
                        CS=(1./SG.edges[e[0]]['weight'])+(1./SG.edges[e[1]]['weight'])
                        l=SG.edges[e[0]]['length']+SG.edges[e[1]]['length']
                        SG.add_edge(e_merged[0],e_merged[1],slope=(SG.nodes[e_merged[0]]['pos'],SG.nodes[e_merged[1]]['pos']),weight=(1./CS),label=self.count_e(),origin=[n]+SG.edges[e[0]]['origin']+SG.edges[e[1]]['origin'],length=l)

                    SG.remove_edge(*e[0])
                    SG.remove_edge(*e[1])
                    SG.remove_node(n)

        return SG

    def get_loop_condition(self,G):

        list_nodes=list(G.nodes())
        self_loops=list(nx.selfloop_edges(G))
        # print(self_loops)
        for e in self_loops:
            if G.degree(e[0]) == 2 :
                list_nodes.remove(e[0])
        list_degrees=np.array(G.degree(list_nodes))

        deg_idx=np.where( list_degrees[:,1]==2)[0]
        deg_sinks=np.array([list_degrees[i,1] for i in deg_idx if G.nodes[list_degrees[i,0]]['source']<=0])

        # cond=np.any( list_degrees[:,1] == 2 )
        cond=np.any( deg_sinks == 2 )

        return cond

    def liver_carved(self,G,ratio,reference):

        H=nx.Graph(G)
        list_nodes=list(G.nodes())
        #prepare distance_sphere
        pos=np.array( [ G.nodes[n]['pos'] for n in list_nodes ] )
        dist=scs.distance.pdist(pos,metric='euclidean')
        max_dist=np.amax(dist)
        # print(max_dist*ratio)
        # adjust reference point
        s=1
        if reference>500:
            s=-1
        # get hold of networks chi distribution
        dict_chi=nx.get_node_attributes(G,'chi')
        chi_sorted=sorted(dict_chi,key=dict_chi.__getitem__)
        chi_val=np.array([dict_chi[c] for c in chi_sorted])

        # calc carved subset of the graph
        cond=True
        while cond:
            if s==1:
                core_nodes=np.where(chi_val <= reference)[0]
            else:
                core_nodes=np.where(chi_val >= reference)[0]

            if len(core_nodes)==0:
                reference+=s
            else:
                cond=False

        for cn in core_nodes:
            core_n=chi_sorted[cn]
            H.nodes[core_n]['source']=1
            list_nodes_H=list(H.nodes())
            cn_pos=np.array([G.nodes[core_n]['pos'] for n in list_nodes_H])
            pos_H=np.array( [ H.nodes[n]['pos'] for n in list_nodes_H ] )
            dv=np.subtract(cn_pos,pos_H)

            dist=np.linalg.norm(dv,axis=1)

            idx_bool=np.where(dist >= max_dist*ratio)[0]
            for i in idx_bool:

                H.remove_node(list_nodes_H[int(i)])

        # clean up debris and relabel
        CC=sorted(nx.connected_components(H), key=len, reverse=True)
        if len(CC)>1:
            list_CC=CC[1:]
            for c in list_CC:
                for n in c:
                    H.remove_node(n)

        # H_fin=nx.Graph()
        # dict_nodes={}
        # for idx_n,n in enumerate(H.nodes()):
        #     H_fin.add_node(idx_n,pos=H.nodes[n]['pos'],chi=H.nodes[n]['chi'],label=H.nodes[n]['label'],source=H.nodes[n]['source'],radius=H.nodes[n]['radius'])
        #     dict_nodes.update({n:idx_n})
        # for idx_e,e in enumerate(H.edges()):
        #     H_fin.add_edge(dict_nodes[e[0]],dict_nodes[e[1]],slope=(H.nodes[e[0]]['pos'],H.nodes[e[1]]['pos']),label=H.edges[e]['label'],length=H.edges[e]['length'])
        H_fin=nx.convert_node_labels_to_integers(H, first_label=0, ordering='default')

        return H_fin

    def liver_set_coefficients( self,graph_sets,ref_graph_sets,threshold,exp ):

        dict_paths,branching=self.find_overlap(graph_sets,ref_graph_sets)
        updated_graphs,dict_pairs, dict_parameters, ref_length=self.find_affiliation(graph_sets,ref_graph_sets,dict_paths,threshold)
        E=exp
        G=updated_graphs[0]

        list_n=list(G.nodes())
        list_e=list(G.edges())
        B=nx.incidence_matrix(G,nodelist=list_n,edgelist=list_e,oriented=True).toarray()
        BT=np.transpose(B)

        J=np.array([ G.nodes[n]['source'] for n in list_n ])
        C=np.array([ G.edges[e]['weight'] for e in list_e ])
        # C=np.array([ G.edges[e]['radius']**4/G.edges[e]['length'] for e in list_e ])
        # for e in G.edges():
        #     G.edges[e]['radius'] = np.power( np.multiply(C,G.edges[e]['length']),0.25)
        # R=np.array([ G.edges[e]['radius'] for e in list_e ])
        # R=np.power(C,0.25)

        # scale it into an arbitrary system and set repulsion coefficients
        nx.set_edge_attributes(G,0.,'coefficient_l1')

        counter=0.
        countertotal=0.
        hist_delta=[]
        hist_deltatotal=[]
        for e in dict_pairs:
            rep_sum=0.
            for tup in dict_parameters[e]:
                R1,R2,dist=tup

                # test skeleton approach
                R1=0.
                R2=0.

                delta=dist-(R1+R2)
                countertotal+=1
                # print(dist)
                # print(R1)
                # print(R2)
                # print(delta)
                hist_deltatotal.append(delta)
                if delta < 0:
                #     print(R1)
                #     print(R2)
                #     print(dist)
                    delta=dist
                    hist_delta.append(delta)
                    counter+=1
                else:
                    hist_delta.append(delta)
                rep_sum+= np.power(np.divide((delta),ref_length),E)

            G.edges[e]['coefficient_l1']=(E/np.absolute(E))*rep_sum*G.edges[e]['length']/G.edges[e]['radius']

        # set fluctuation coefficient
        num_n=nx.number_of_nodes(G)
        x=np.where(J > 0)[0]
        idx=np.where(J < 0)[0]
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

        # C=np.divide(C,ref_length**4)
        D=np.matmul(B,np.matmul(np.diag(C),BT))
        ID=lina.pinv(D)
        BID=np.matmul(BT,ID)
        BIDT=np.transpose(BID)
        VA=np.matmul(BID,np.matmul(V,BIDT))
        UA=np.matmul(BID,np.matmul(U,BIDT))

        for j,e in enumerate(list_e):
            G.edges[e]['coefficient_l3']=VA[j,j]/UA[j,j]

        return G,branching,ref_length,dict_pairs,hist_delta, hist_deltatotal
        # return branching,ref_length

    def find_overlap(self,graph_sets,ref_graph_sets):
        rad=[]
        center=[]
        pos=[]
        list_nodes=[]
        branching=[[],[]]
        paths=[[],[]]
        for G in graph_sets:
            list_n_G=list(G.nodes())
            list_nodes.append(list_n_G)
            pos_G=np.array( [ G.nodes[n]['pos'] for n in list_n_G ] )
            pos.append(pos_G)
            dist_G=scs.distance.pdist( pos_G,metric='euclidean' )
            rad.append(np.amax(dist_G))
            center.append(np.sum(pos_G,axis=0)/len(pos_G))

        for i,c in enumerate(center):
            for j,p in enumerate(pos[i-1]):
                d=np.linalg.norm( np.subtract(p,c) )
                if d <rad[i]:
                    n=list_nodes[i-1][j]
                    if graph_sets[i-1].degree(n)>=3:
                        branching[i-1].append(n)

        dict_paths=[{},{}]
        for i,G in enumerate(graph_sets):
            for e in G.edges():
                if (e[0] in branching[i]) or (e[1] in branching[i]):
                    path=[ G.nodes[e[0]]['pos'], G.nodes[e[1]]['pos'] ]
                    for n in G.edges[e][ 'origin' ]:
                        path.append(ref_graph_sets[i].nodes[n]['pos'])
                    dict_paths[i][e]=path

        return dict_paths,branching[0]

    def find_affiliation(self,graph_sets,ref_graph_sets,dict_paths,threshold):

        dict_center=[{},{}]
        dict_pairs_idx={}
        dict_pairs_dist={}
        dict_repulsion_parameters={}

        for i,G in enumerate(graph_sets):
            paths=dict_paths[i]
            for e in G.edges():
                G.edges[e]['radius']=(G.nodes[e[0]]['radius']+G.nodes[e[1]]['radius'])/2.
            for e in paths:
                dict_center[i][e]=np.sum(paths[e],axis=0)/len(paths[e])

        for e1 in dict_center[0]:
            for e2 in dict_center[1]:
                d=np.linalg.norm(np.subtract(dict_center[0][e1],dict_center[1][e2]))

                if d <=threshold:
                    if e1 not in dict_pairs_idx.keys():
                        dict_pairs_idx[e1]=[e2]
                        dict_pairs_dist[e1]=[d]
                    else:
                        dict_pairs_idx[e1].append(e2)
                        dict_pairs_dist[e1].append(d)

        global_mean_dist=[]
        for i,e1 in enumerate(dict_pairs_idx):
            path_1=dict_paths[0][e1]

            for j,e2 in enumerate(dict_pairs_idx[e1]):
                # if i==0 and j==0:
                    path_2=dict_paths[1][e2]
                    min_dist=[]
                    # print(path_2)
                    for p1 in path_1:
                        # print(p1)
                        D=np.subtract(p1,path_2)
                        # print(D)
                        dist=np.linalg.norm( D ,axis=1)
                        # print(dist)
                        min_dist.append(np.amin(dist))

                    # mean_dist=np.amax(min_dist)
                    mean_dist=np.mean(min_dist)
                    # mean_dist=np.amin(min_dist)

                    global_mean_dist.append(mean_dist)
                    R1=[graph_sets[0].nodes[e1[0]]['radius'],graph_sets[0].nodes[e1[1]]['radius']]
                    R2=[graph_sets[1].nodes[e2[0]]['radius'],graph_sets[1].nodes[e2[1]]['radius']]
                    # print(min_dist)
                    for m in graph_sets[0].edges[e1]['origin']:
                        R1.append( ref_graph_sets[0].nodes[m]['radius'])
                    for m in graph_sets[1].edges[e2]['origin']:
                        R2.append( ref_graph_sets[1].nodes[m]['radius'])
                    graph_sets[0].edges[e1]['radius']=np.mean(R1)
                    graph_sets[1].edges[e2]['radius']=np.mean(R2)

                    # R1=np.power(graph_sets[0].edges[e1]['weight']*graph_sets[0].edges[e1]['length'],0.25)
                    # R2=np.power(graph_sets[1].edges[e2]['weight']*graph_sets[1].edges[e2]['length'],0.25)

                    if e1 not in dict_repulsion_parameters.keys():
                        # dict_repulsion_parameters[e1]=[[graph_sets[0].edges[e1]['weight'],graph_sets[1].edges[e2]['weight'],mean_dist]]
                        # dict_repulsion_parameters[e1]=[[R1,R2,mean_dist]]
                        dict_repulsion_parameters[e1]=[[graph_sets[0].edges[e1]['radius'],graph_sets[1].edges[e2]['radius'],mean_dist]]
                    else:
                        dict_repulsion_parameters[e1].append([graph_sets[0].edges[e1]['radius'],graph_sets[1].edges[e2]['radius'],mean_dist])
                        # dict_repulsion_parameters[e1].append([graph_sets[0].edges[e1]['weight'],graph_sets[1].edges[e2]['weight'],mean_dist])
                        # dict_repulsion_parameters[e1].append([R1,R2,mean_dist])
        global_mean=np.amax(global_mean_dist)
        # global_mean=np.mean(global_mean_dist)

        return graph_sets,dict_pairs_idx,dict_repulsion_parameters,global_mean

    def calc_histogram_chi_cycles(self,G,cycle_basis):

        chi_list=[]
        for c in cycle_basis:
            cycle_chi=0.
            for n in c.nodes():
                cycle_chi+=G.nodes[n]['chi']
            cycle_chi/=nx.number_of_nodes(c)
            c.graph['chi']=cycle_chi
            chi_list.append(cycle_chi)
        hist,bins=np.histogram(chi_list,range(1001))

        return hist,bins[:-1]

    def make_it_simple(self,M):
        # check whether multigraph has parallels, if yes then plant a auxilarry node in one parellel to enable networkx the identification of euclidean cycles

        G=nx.Graph(M)
        nx.set_edge_attributes(G,'true',name='identity')
        graphtype=nx.info(M).split('\n')[1].split(' ')[1]
        if graphtype=='MultiGraph':

            D=nx.MultiGraph(M)
            nx.set_edge_attributes(D,'true',name='identity')
            nx.set_node_attributes(D,'true',name='identity')
            N=nx.number_of_nodes(M)
            seen={}
            labels_e = M.edges('label')
            dict_labels_e={}
            for le in labels_e:
                dict_labels_e[(le[0],le[1])]=le[2]
            sorted_label_e_list=sorted(dict_labels_e,key=dict_labels_e.__getitem__)
            for e in M.edges():
                num_conncetions=M.number_of_edges(*e)
                if num_conncetions >1 and e not in seen.keys():

                    seen[e]=0
                    for i in range(1,num_conncetions):
                        N+=1
                        D.add_edge( e[0],N,key=i, weight=2.*M.edges[(e[0],e[1],i)]['weight'],identity='dummy',label=self.count_e(),origin=M.edges[(e[0],e[1],i)]['origin'],length=M.edges[(e[0],e[1],i)]['length']/2. )
                        D.add_edge( N,e[1],key=i, weight=2.*M.edges[(e[0],e[1],i)]['weight'],identity='dummy',label=self.count_e(),origin=[],length=M.edges[(e[0],e[1],i)]['length']/2. )

                        # D.nodes[N]['chi']=(D.nodes[e[0]]['chi']+D.nodes[e[1]]['chi'])/2.
                        D.nodes[N]['label']=self.count_n()
                        D.nodes[N]['source']=-1.
                        D.nodes[N]['identity']='dummy'
                        D.nodes[N]['radius']=( D.nodes[e[0]]['radius'] + D.nodes[e[1]]['radius'] )/2.
                        D.nodes[N]['pos']=np.add( D.nodes[e[0]]['pos'] , D.nodes[e[1]]['pos'] )/2.
                        D.remove_edge(e[0],e[1],key=i)

            G=nx.Graph(D)
        G.graph['identity']=0
        return G

    def calc_cycle_coalescence(self,G,cycle_basis):

        if 'identity' in G.graph.keys():
            # print('checking out dummies...')
            for e in G.edges():
                if G.edges[e]['identity']=='dummy':
                    G.edges[e]['weight']*=2.
        # for cb in cycle_basis:
        #     print(cb.edges())
        #create cycle_map_tree with cycles' edges as tree nodes
        Cycle_Tree, list_asymmetry=super(tool_liver,self).calc_cycle_coalescence(G,cycle_basis)

        return Cycle_Tree, list_asymmetry

    def calc_scaling(self, G):

        list_deg=[]
        radii_tuple=[]
        radii_delta=[]
        murray_exp=[]

        for n in G.nodes():
            if G.degree[n] ==3:
                list_deg.append(n)
        for n in list_deg:
            # r=[G.nodes[n]['radius']]
            r=[]
            list_edges=list(G.edges(n))
            # r_aux=[]
            for e in list_edges:
                # r_aux.append(G.nodes[e[1]]['radius'])
                r.append((G.nodes[e[0]]['radius']+G.nodes[e[1]]['radius'])/2.)

            # r_aux.pop(np.argmax(r_aux))
            radii_tuple.append(sorted(r))

        for r_tup in radii_tuple:
            x=r_tup[0]/r_tup[-1]
            y=r_tup[1]/r_tup[-1]
            exp=self.fit_murray(x,y)
            murray_exp.append(exp)
            # delta=np.sqrt( np.power(r_tup[0]-r_tup[-1],2.) + np.power(r_tup[1]-r_tup[-1],2.) )
            # radii_delta.append(delta)
            dr1=np.power(r_tup[0]-r_tup[-1],2)
            dr2=np.power(r_tup[1]-r_tup[-1],2)
            d=[dr1,dr2]
            radii_delta.append(d)


        return murray_exp,np.array(radii_delta)

    def calc_scaling_chi(self, G):

        list_deg=[]
        radii_tuple=[]
        radii_delta=[]
        murray_exp=[]

        chi_list_aux=[ [] for i in range(self.chi_hist) ]
        chi_list_exp=[ [] for i in range(self.chi_hist) ]
        for n in G.nodes():
            if G.degree[n] ==3:
                list_deg.append(n)
        for n in list_deg:
            chi=G.nodes[n]['chi']
            r=[]
            list_edges=list(G.edges(n))
            for e in list_edges:
                r.append((G.nodes[e[0]]['radius']+G.nodes[e[1]]['radius'])/2.)

            idx=int(chi*self.chi_hist/1000.)
            chi_list_aux[idx].append(sorted(r))
        for i,c in enumerate(chi_list_aux):
            for r_tup in c:
                x=r_tup[0]/r_tup[-1]
                y=r_tup[1]/r_tup[-1]
                exp=self.fit_murray(x,y)
                chi_list_exp[i].append(exp)

        return chi_list_exp,radii_delta

    def measure_nematic_order(self,directors):

        N=len(directors)
        Q=np.zeros((3,3))
        for d in directors:
            director_dyad=np.subtract(np.multiply(3.,np.outer(d,d)),np.identity(3))
            Q=np.add(Q,director_dyad)
        Q=np.divide(Q,2.* N)
        w,v=np.linalg.eigh(Q)
        i=np.argmax(w)
        p1=[np.dot(v[:,i],d) for d in directors]

        return np.sum(p1)/float(len(p1)),w[i],v[:,i]

    def plane_hessian_form(self,points):
        p1=np.subtract(points[0],points[1])
        p2=np.subtract(points[0],points[2])
        p0=points[0]
        n=self.calc_normal(p1,p2)
        return n, p0

    def calc_normal(self,p1,p2):
        normal=np.cross(p1,p2)
        n=np.divide(normal,np.linalg.norm(normal))
        return n

    def delta(self,stack_np,*args):
        points=args[0]
        d=np.array([np.power(np.dot(stack_np[0:3],np.subtract(p,stack_np[3:])),2) for p in points])
        delta=np.sum(d)
        return delta

    def norm(self,stack_np):
        n=stack_np[0:3]
        return np.dot(n,n) - 1

    def cycle_nematic_order(self,M,cycle_base):

        cons={'type':'eq','fun':self.norm}
        directors=[]
        centers=[]
        for c in cycle_base:

            #find all original circle nodes before graphs was refined
            points=[M.nodes[n]['pos'] for n in c.nodes() if M.nodes[n]['identity']=='true']
            for e in c.edges():
                for n in M.edges[e]['origin']:
                    points.append(self.reference_graph.nodes[n]['pos'])

            n,p0=self.plane_hessian_form([points[0],points[1],points[2]])
            stack_np=np.concatenate((n,p0))
            res=so.minimize(self.delta,stack_np,points,constraints=cons)
            directors.append(res.x[0:3])
            center=np.zeros(3)
            for p in points:
                center=np.add(center,p)

            centers.append(center/float(len(points)))
        p1,p2,d=self.measure_nematic_order(directors)

        return p1,p2,d,[np.array(directors),np.array(centers)]

    #testing
    def calc_torsion_vectors(self, G):

        list_links=[]
        branching_dict={}

        # find links with 2 y junctions
        for e in G.edges(keys=True):

            if G.degree[e[0]] == 3:
                    if G.degree[e[1]] == 3:
                        list_links.append(e)
        # iterate through that lis, identifying the non-shared vessel arms and calc their plane normal
        for e in list_links:
            u=G.nodes[e[0]]['pos']-G.nodes[e[1]]['pos']
            branching_dict[e]=[]
            for n in [e[0],e[1]]:
                vector_list=[]
                list_edges=list(G.edges(n,keys=True))
                for i,m in enumerate(list_edges):
                    if G.edges[m]['label']!=G.edges[e]['label']:
                        v=G.nodes[m[0]]['pos']-G.nodes[m[1]]['pos']
                        vector_list.append(v)
                branching_dict[e].append([self.calc_normal(u,v)for v in vector_list])

        return branching_dict

    def calc_torsion_vectors_coarse(self, G, SG):

        list_links=[]
        branching_dict={}

        # find links with 2 y junctions
        for e in SG.edges(keys=True):

            if SG.degree[e[0]] == 3:
                    if SG.degree[e[1]] == 3:
                        list_links.append(e)
        # iterate through that lis, identifying the non-shared vessel arms and calc their plane normal
        for e in list_links:
            u=np.subtract(SG.nodes[e[0]]['pos'],SG.nodes[e[1]]['pos'])

            branching_dict[e]=[]
            for n in [e[0],e[1]]:
                vector_list=[]
                d1=[]
                for neighbor in SG.edges[e]['origin']:
                    d1.append(np.linalg.norm(np.subtract(SG.nodes[n]['pos'],G.nodes[neighbor]['pos'])))
                if len(d1)>0:
                    idx=np.argmin(d1)
                    u=np.subtract(SG.nodes[n]['pos'],G.nodes[neighbor]['pos'])

                list_edges=list(SG.edges(n,keys=True))
                for i,m in enumerate(list_edges):
                    if SG.edges[m]['label']!=SG.edges[e]['label']:

                        v=np.subtract(SG.nodes[m[0]]['pos'],SG.nodes[m[1]]['pos'])
                        d1=[]
                        for neighbor in SG.edges[m]['origin']:
                            d1.append(np.linalg.norm(np.subtract(SG.nodes[n]['pos'],G.nodes[neighbor]['pos'])))
                        if len(d1)>0:
                            idx=np.argmin(d1)
                            v=np.subtract(SG.nodes[n]['pos'],G.nodes[neighbor]['pos'])

                        vector_list.append(v)
                branching_dict[e].append([self.calc_normal(u,v)for v in vector_list])

        return branching_dict

    def calc_torsion_angle(self, branching_dict):

        torsion_angle=[]

        for e in branching_dict.keys():
            for v in branching_dict[e][0]:
                for u in branching_dict[e][1]:

                    if (np.linalg.norm(v)*np.linalg.norm(u))==0.:
                        print('heeeee?')
                    angle=np.arccos(np.round(np.dot(v,u)/(np.linalg.norm(v)*np.linalg.norm(u)),6)) *180./np.pi
                    if angle > 90.:
                        angle=180-angle
                    torsion_angle.append(angle)

        return torsion_angle

    def calc_angle(self, G):

        list_deg=[]
        branch_angles=[]
        branch_angles_min=[]
        branch_angles_max=[]
        branch_triple=[]
        for n in G.nodes():
            if G.degree[n] == 3:
                list_deg.append(n)
        for n in list_deg:
            vector_list=[]
            list_edges=list(G.edges(n))
            for i,e in enumerate(list_edges):
                v=G.nodes[e[0]]['pos']-G.nodes[e[1]]['pos']
                vector_list.append(v)

            aux_angles=[]

            for i,v1 in enumerate(vector_list[:-1]):
                for j,v2 in enumerate(vector_list[i+1:]):
                    angle=np.arccos(np.round(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)),6)) *180./np.pi
                    # print(angle*180./np.pi)
                    aux_angles.append(angle)

            branch_angles+=aux_angles
            idx_max=np.argmax(aux_angles)
            idx_min=np.argmin(aux_angles)
            idx_misc=0
            for i in range(3):
                if i!=idx_min and i!=idx_max:
                    idx_misc=i
            branch_angles_max.append(aux_angles[idx_max])
            branch_angles_min.append(aux_angles[idx_min])
            # branch_triple.append([aux_angles[idx_min],aux_angles[idx_max],aux_angles[idx_misc]])
            branch_triple.append([aux_angles[0],aux_angles[1],aux_angles[2]])

        return branch_angles, branch_angles_max, branch_angles_min, np.array(branch_triple)

    def calc_angle_coarsegrained(self, G, SG):

        list_deg=[]
        branch_angles=[]
        branch_angles_min=[]
        branch_angles_max=[]
        branch_triple=[]
        for n in SG.nodes():
            if SG.degree[n] == 3:
                list_deg.append(n)


        for n in list_deg:

            vector_list=[]
            list_edges=list(SG.edges(n,keys=True))
            for i,e in enumerate(list_edges):
                d=[]
                for neighbor in SG.edges[e]['origin']:
                    d.append(np.linalg.norm(np.subtract(SG.nodes[n]['pos'],G.nodes[neighbor]['pos'])))
                if len(d)>0:

                    idx=np.argmin(d)
                    v=np.subtract(SG.nodes[n]['pos'],G.nodes[neighbor]['pos'])
                    vector_list.append(v)

                else:

                    v=SG.nodes[e[0]]['pos']-SG.nodes[e[1]]['pos']
                    vector_list.append(v)

            aux_angles=[]

            for i,v1 in enumerate(vector_list[:-1]):
                for j,v2 in enumerate(vector_list[i+1:]):
                    angle=np.arccos(np.round(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)),6)) *180./np.pi
                    # print(angle*180./np.pi)
                    aux_angles.append(angle)

            branch_angles+=aux_angles
            idx_max=np.argmax(aux_angles)
            idx_min=np.argmin(aux_angles)
            idx_misc=0
            for i in range(3):
                if i!=idx_min and i!=idx_max:
                    idx_misc=i
            branch_angles_max.append(aux_angles[idx_max])
            branch_angles_min.append(aux_angles[idx_min])
            # branch_triple.append([aux_angles[idx_min],aux_angles[idx_max],aux_angles[idx_misc]])
            branch_triple.append([aux_angles[0],aux_angles[1],aux_angles[2]])

        return branch_angles, branch_angles_max, branch_angles_min, np.array(branch_triple)

    def calc_intersection_convexhull(self,points_polyhedron,points_curve):

        intersections=[]

        hull=scs.ConvexHull(points_polyhedron)
        simplices_planes=[ self.calc_planes( points_polyhedron[simplex] )  for simplex in hull.simplices ]
        line_segments=[ [ points_curve[i],np.subtract(points_curve[i+1] , points_curve[i]) ] for i,pc in enumerate( points_curve[:-1] ) ]

        for sp in simplices_planes:
            for ls in line_segments:

                t=np.dot( np.subtract(sp[3],ls[0]),sp[0] )/np.dot( ls[1],sp[0] )

                if 0 <= t <= 1:

                    p=np.add(ls[1]*t,ls[0])
                    b=np.subtract(p,sp[3])
                    A=np.column_stack((sp[1],sp[2]))
                    solution=np.linalg.lstsq(A,b,rcond=True)
                    if (0 <= solution[0][0]+solution[0][1] <= 1) and (0 <= solution[0][0] <= 1) and (0 <= solution[0][1] <= 1) :
                        intersections.append(1)

        return intersections

    def calc_planes(self,points):
        p1=np.subtract(points[1],points[0])
        p2=np.subtract(points[2],points[0])
        p0=points[0]
        normal=np.cross(p1,p2)
        n=np.divide(normal,np.linalg.norm(normal))
        return n, p2, p1, p0

    def calc_planes_mean(self,points):
        # p1=np.subtract(points[1],points[0])
        # p2=np.subtract(points[2],points[0])
        # p0=points[0]
        # normal=np.cross(p1,p2)
        # n=np.divide(normal,np.linalg.norm(normal))
        n, p2, p1, p0 = self.calc_planes(points)

        B=np.column_stack((n,p0))
        cons={'type':'eq','fun':self.norm}
        res=so.minimize(self.delta,B,points,constraints=cons)

        n=res.x[0:3]
        p0=np.zeros(3)
        for p in points:
            p0=np.add(p0,p)
        p1=np.dot([[1,0,0],[0,0,-1],[0,1,0]],n)
        p2=np.dot([[0,0,1],[0,1,0],[-1,0,0]],n)

        return n, p2, p1, p0/float(len(points))

    def murray_scaling(self,graph_sets,ref_graph_sets,threshold,exp):

        # containers for output
        radii_tuple=[]
        murray_pars_0=[]
        murray_pars_1=[]
        murray_pars_2=[]
        # set coefficients and fetch relevant branchings

        G,branching,ref_length,dict_pairs,hist_delta, hist_deltatotal=self.liver_set_coefficients(graph_sets,ref_graph_sets,threshold,exp)
        for n in branching:
            if G.degree(n)==3:
                r={}
                radii=[]
                n_edges=list(G.edges(n))

                for e in n_edges:
                    # r[e]=np.divide(np.power(G.edges[e]['weight'],0.25 ),ref_length)
                    r[e]=np.divide(G.edges[e]['radius'],ref_length)

                list_sorted=sorted(r,key=r.__getitem__)
                radii = [ r[e] for e in list_sorted ]

                coefficients_l3 = [ G.edges[e]['coefficient_l3']  for e in list_sorted ]
                coefficients_l1 = [ G.edges[e]['coefficient_l1']  for e in list_sorted ]
                radii_tuple.append([np.power(radii,3),coefficients_l3,coefficients_l1])

        check=[]
        for i in range(len(radii_tuple)):
            for j in range(len(radii_tuple)):
                for k in range(len(radii_tuple)):
                    if k>j and j>i :
                        check.append([i,j,k])


        signs=[[1.,1.,-1.],[1.,-1.,-1.],[-1.,-1.,-1.],[-1.,1.,-1.],[-1.,-1.,1.],[-1.,1.,1.],[1.,-1.,1.]]

        S=[]
        for i in range(7):
            for j in range(7):
                for k in range(7):
                    if k>j and j>i:
                        S.append([signs[i],signs[j],signs[k]])

        # for i,c in enumerate(check):
        #     tup1=radii_tuple[c[0]]
        #     tup2=radii_tuple[c[1]]
        #     tup3=radii_tuple[c[2]]
        #     # if i== 0:
        #     pars=self.fit_murray_kirchhoff_flux_3D([S,tup1,tup2,tup3])
        #     if not (np.isnan(pars[0]) or np.isnan(pars[1]) or np.isnan(pars[2])):
        #         murray_pars_0.append(np.absolute(pars[0]))
        #         murray_pars_1.append(np.absolute(pars[1]))
        #         murray_pars_2.append(np.absolute(pars[2]))
        super_tup=[]
        # check_rand=rd.sample(check,10000)
        check_rand=rd.sample(check,10000)
        for i,c in enumerate(check_rand):
            # if i==0:
                tup1=radii_tuple[c[0]]
                tup2=radii_tuple[c[1]]
                tup3=radii_tuple[c[2]]
                super_tup.append([S,tup1,tup2,tup3])

        # multiplicity=4
        # for m in range(multiplicity):
        #     for i,r in enumerate(radii_tuple):
        #         super_tup.append([signs,r])
        #
        # pool = mp.Pool(processes=4)
        # pars=pool.map( self.fit_murray_kirchhoff_flux_3D, super_tup )
        # for p in pars:
        #     if not (np.isnan(p[0]) or np.isnan(p[1]) or np.isnan(p[2])):
        #         murray_pars_0.append(np.absolute(p[0]))
        #         murray_pars_1.append(np.absolute(p[1]))
        #         murray_pars_2.append(np.absolute(p[2]))

        return murray_pars_0,murray_pars_1,murray_pars_2

    def murray_scaling_subpopulations(self,graph_sets,ref_graph_sets,threshold,exp):

        # containers for output
        radii_tuple=[]
        murray_pars_0=[]
        murray_pars_1=[]
        murray_pars_2=[]
        # set coefficients and fetch relevant branchings

        G,branching,ref_length,dict_pairs,hist_delta, hist_deltatotal=self.liver_set_coefficients(graph_sets,ref_graph_sets,threshold,exp)
        R_branch=[]
        dict_branching={}
        counter=0
        for n in branching:
            if G.degree(n)==3:
                r={}
                radii=[]
                n_edges=list(G.edges(n))

                for e in n_edges:
                    # r[e]=np.divide(np.power(G.edges[e]['weight'],0.25 ),ref_length)
                    r[e]=np.divide(G.edges[e]['radius'],ref_length)

                list_sorted=sorted(r,key=r.__getitem__)
                radii = [ r[e] for e in list_sorted ]

                coefficients_l3 = [ G.edges[e]['coefficient_l3']  for e in list_sorted ]
                coefficients_l1 = [ G.edges[e]['coefficient_l1']  for e in list_sorted ]
                radii_tuple.append([np.power(radii,3),coefficients_l3,coefficients_l1])
                R_branch.append(radii)

                dict_branching[n]=counter
                counter+=1
        # check=[]
        # for i in range(len(radii_tuple)):
        #     for j in range(len(radii_tuple)):
        #         for k in range(len(radii_tuple)):
        #             if k>j and j>i:
        #                 check.append([i,j,k])
        signs=[[1.,1.,-1.],[1.,-1.,-1.],[-1.,-1.,-1.],[-1.,1.,-1.],[-1.,-1.,1.],[-1.,1.,1.],[1.,-1.,1.]]

        # S=[]
        # for i in range(7):
        #     for j in range(7):
        #         for k in range(7):
        #             if k>j and j>i:
        #                 S.append([signs[i],signs[j],signs[k]])

        # for i,c in enumerate(check):
        #     tup1=radii_tuple[c[0]]
        #     tup2=radii_tuple[c[1]]
        #     tup3=radii_tuple[c[2]]
        #     if i==10:
        #         pars=self.fit_murray_kirchhoff_flux_parameters([tup1,tup2,tup3],self.smart_start([tup1,tup2,tup3]))
        #         if not (np.isnan(pars[0]) or np.isnan(pars[1]) or np.isnan(pars[2])):
        #             murray_pars_0.append(np.absolute(pars[0]))
        #             murray_pars_1.append(np.absolute(pars[1]))
        #             murray_pars_2.append(np.absolute(pars[2]))
        super_tup=[]
        # for i,c in enumerate(check):
        #     tup1=radii_tuple[c[0]]
        #     tup2=radii_tuple[c[1]]
        #     tup3=radii_tuple[c[2]]
        #     super_tup.append([S,tup1,tup2,tup3])

        multiplicity=10
        for m in range(multiplicity):
            for i,r in enumerate(radii_tuple):
                super_tup.append([signs,r])

        pool = mp.Pool(processes=4)
        pars=pool.map( self.fit_murray_kirchhoff_flux_3D, super_tup )
        branch_idx=[]
        branch_idx_regulars=[]
        for i,p in enumerate(pars):
            if not (np.isnan(p[0]) or np.isnan(p[1]) or np.isnan(p[2])):
                murray_pars_0.append(np.absolute(p[0]))
                murray_pars_1.append(np.absolute(p[1]))
                murray_pars_2.append(np.absolute(p[2]))
                if murray_pars_2[-1] <= 0.01 :
                    branch_idx.append(int(i/multiplicity))
                else:
                    branch_idx_regulars.append(int(i/multiplicity))

        list_deg=[]
        murray_exp=[]
        murray_exp_regulars=[]
        branch_idx=list(dict.fromkeys(branch_idx))
        branch_idx_regulars=list(dict.fromkeys(branch_idx_regulars))

        print(branch_idx)
        print(branch_idx_regulars)
        R_TUPLE=np.array(R_branch)[branch_idx]
        for r_tup in R_TUPLE:

            x=r_tup[0]/r_tup[-1]
            y=r_tup[1]/r_tup[-1]
            exp=self.fit_murray(x,y)
            murray_exp.append(exp)

        R_TUPLE=np.array(R_branch)[branch_idx_regulars]
        for r_tup in R_TUPLE:

            x=r_tup[0]/r_tup[-1]
            y=r_tup[1]/r_tup[-1]
            exp=self.fit_murray(x,y)
            murray_exp_regulars.append(exp)

        return [murray_pars_0,murray_pars_1,murray_pars_2],[murray_exp,murray_exp_regulars,dict_branching,branch_idx]

    def randomize_weight(self,G):
        RAD_G=nx.Graph(G)
        # maximum=np.amax(list(nx.get_edge_attributes(G,'weight').values()))
        minimum=np.amin(list(nx.get_edge_attributes(G,'weight').values()))
        for e in RAD_G.edges():
            # RAD_G.edges[e]['weight']=np.random.rand(1)[0]*maximum
            # RAD_G.edges[e]['weight']=np.random.rand(1)[0]*minimum
            RAD_G.edges[e]['weight']=np.random.rand(1)[0]*nx.number_of_nodes(G)

        return RAD_G
