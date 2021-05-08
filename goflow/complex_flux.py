import sys
import networkx as nx
import numpy as np
import scipy as sy
import scipy.integrate as si
import scipy.spatial as sp
import scipy.optimize as sc
import os.path as op
import os
import pickle
import scipy.linalg as lina
import random as rd
import init_integration
import line_profiler
import atexit
profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats)

def find_root(G):
    for n in G.nodes():
        if G.nodes[n]['source']>0:
            return n
    return 0

def find_sink(G):
    for n in G.nodes():
        if G.nodes[n]['source']<0:
            return n
    return 0

def calc_peclet_PE(Q,R_sq,L,D):
    V=np.divide(Q,np.pi*R_sq)
    return np.absolute(np.multiply(V,L)/D)

def calc_surface_transport_S(Q,R_sq,L,D,gamma):

    # V=np.absolute(np.divide(Q,np.pi*R_sq))
    R=np.sqrt(R_sq)
    alpha=gamma*L
    # S=gamma*D*np.divide(L,np.multiply(R,V))
    S=D*np.pi*np.divide(np.multiply(alpha,R),np.absolute(Q))

    return alpha,S

def calc_uptake_rate_beta(alpha,PE,S):

    ones=np.ones(len(PE))
    A1=48.*ones
    A2=np.power(np.divide(alpha,S),2)
    A=np.divide(24*PE,np.add(A1,A2))

    B1=np.divide(S,PE)*8.
    B2=np.divide(np.power(alpha,2),np.multiply(PE,S)*6.)
    B=np.sqrt(np.add(ones,np.add(B1,B2)))

    beta=np.multiply(A,np.subtract(B,ones))
    return beta

def calc_flux_orientations(G,B,Q):

    root=find_root(G)
    dict_incoming,dict_outcoming,dict_mem_n,dict_mem_nodes,dict_mem_e,dict_incoming_noroot,dict_incoming_noroot_e={},{},{},{},{},{},{}
    BQ=np.zeros((len(B[:,0]),len(B[0,:])))
    list_e=list(G.edges())
    idx_e=[G.edges[e]['label'] for e in list_e]
    for n in G.nodes():

        idx_n=G.nodes[n]['label']
        BQ[idx_n,:]=np.multiply(B[idx_n,:],Q)
        b=BQ[idx_n,:]
        subhit1_list=[]
        subhit2_list=[]
        subhit3_list=[]

        dict_outcoming[n]=np.where( b > 0)[0]
        hit_list=np.where( b < 0)[0]
        dict_incoming[n]=hit_list

        for idx in hit_list:
            e=list_e[idx]

            if e[0]!=n:
                subhit1_list.append(e[0])
                dict_mem_e[e]=G.nodes[e[0]]['label']
                if e[0]!=root:
                    subhit2_list.append(e[0])
                    subhit3_list.append(idx)
            else:
                subhit1_list.append(e[1])
                dict_mem_e[e]=G.nodes[e[1]]['label']
                if e[1]!=root:
                    subhit2_list.append(e[1])
                    subhit3_list.append(idx)

        dict_mem_nodes[n]=subhit1_list
        dict_incoming_noroot[n]=subhit2_list
        dict_incoming_noroot_e[n]=subhit3_list
        dict_mem_n[n]=[(G.nodes[m]['label']) for m in subhit1_list]

    return dict_incoming,dict_outcoming,dict_mem_n,dict_mem_nodes,dict_mem_e,dict_incoming_noroot,dict_incoming_noroot_e,BQ

def concentrate_love_repeat(c,F,*args):

    dict_fluxes,push_list_nodes,master_list,G,dict_incoming,dict_outcoming,AQ,Q,beta,PE=args
    nodes_left_undetermined=True
    N=len(G.nodes())
    dict_E,dict_idx_E,dict_incoming_noroot={},{},{}
    for n in G.nodes():
        list_e=list(G.edges(n))
        dict_E[n]=list_e
        dict_idx_E[n]=[G.edges[e]['label'] for e in list_e ]
        dict_incoming_noroot[n]=[]
    while(nodes_left_undetermined):

        push_list_cache=[]
        push_list_cache_idx=[]
        for n in push_list_nodes:

            if sorted(dict_fluxes[n]) == sorted(dict_incoming[n]):
                idx_n=G.nodes[n]['label']
                if len(dict_outcoming[n])!=0:

                    c[idx_n]=np.divide(np.sum(F[dict_incoming[n]]),np.sum(AQ[dict_outcoming[n]]))
                    master_list.append(idx_n)
                    for idx_e in dict_outcoming[n]:
                        dict_fluxes[n].append(idx_e)
                        F[idx_e]=F[idx_e]*c[idx_n]
                    for i,e in enumerate(dict_E[n]):
                        for m in e:
                            idx_n=G.nodes[m]['label']
                            if (idx_n not in master_list) :
                                dict_fluxes[m].append(dict_idx_E[n][i])
                                if (idx_n not in push_list_cache_idx) :
                                    push_list_cache.append(m)
                                    push_list_cache_idx.append(idx_n)
                else:
                    master_list.append(idx_n)

            else:
                push_list_cache.append(n)

        push_list_nodes=push_list_cache

        if len(master_list)==N:
            nodes_left_undetermined=False

    return c,F

def calc_nodal_concentrations(G,B,Q,PE,beta,c0):

    N=len(G.nodes)
    M=len(G.edges)
    c=np.zeros(N)
    AQ=np.absolute(np.multiply(np.add(np.ones(M),np.divide(beta,PE)),Q))
    # print(np.divide(beta,PE))
    # print(np.divide(beta,PE))

    F=np.multiply(AQ,np.exp(-beta))
    n=find_root(G)
    n_idx=G.nodes[n]['label']
    c[n_idx]=c0

    master_list=[n_idx]
    E=G.edges(n)
    push_list_nodes=[]
    dict_incoming,dict_outcoming,dict_mem_n,dict_mem_nodes,dict_mem_e,dict_incoming_noroot,dict_incoming_noroot_e,BQ=calc_flux_orientations(G,B,Q)
    dict_fluxes={}

    for n in G.nodes():
        dict_fluxes[n]=[]
    for e in E:
        idx_e=G.edges[e]['label']
        if BQ[n_idx,idx_e]>0:
            idx_e=G.edges[e]['label']
            F[idx_e]=F[idx_e]*c0

            for n in e:

                idx_n=G.nodes[n]['label']
                if idx_n not in master_list:
                    push_list_nodes.append(n)
                    dict_fluxes[n].append(idx_e)

    c,F=concentrate_love_repeat(c,F,dict_fluxes,push_list_nodes,master_list,G,dict_incoming,dict_outcoming,AQ,Q,beta,PE)
    return c,F,dict_incoming,dict_outcoming,dict_mem_n,dict_mem_nodes,dict_mem_e,dict_incoming_noroot,dict_incoming_noroot_e

def recursive_topo(topo,topo_nodes,topo_edges,dict_incoming_noroot,dict_incoming_noroot_e,i,G):

    for n in topo[i]:
        topo_edges[i+1]=[idx for idx in dict_incoming_noroot_e[n]]
        topo_nodes[i+1]=[G.nodes[v]['label'] for v in dict_incoming_noroot[n]]
        topo[i+1]=[v for v in dict_incoming_noroot[n]]

    if len(topo[i+1])!=0:
        topo,topo_nodes,topo_edges=recursive_topo(topo,topo_nodes,topo_edges,dict_incoming_noroot,dict_incoming_noroot_e,i+1,G)

    return topo,topo_nodes,topo_edges

def calc_absorption(J,beta,S,PE,alpha):

    ones=np.ones(len(beta))

    A=np.divide(J,np.add(ones,np.divide(beta,PE)))
    b=np.divide(np.power(alpha,2),np.multiply(S,PE))
    B=np.divide(np.subtract(2.*np.divide(S,beta),b/12.),np.add(ones,b/4.))
    C=np.subtract(ones,np.exp(-beta))

    phi=np.multiply(np.multiply(A,B),C)

    return phi

def calc_flux_parameters(flux_par,abs_par,K):

    B,BT,L,Q=flux_par
    diff_R,diff_S,c0=abs_par

    R_sq=np.sqrt(np.multiply(K.C,L)/K.k)
    PE=calc_peclet_PE(Q,R_sq,L,diff_R)
    alpha,S=calc_surface_transport_S(Q,R_sq,L,diff_R,diff_S)
    beta=calc_uptake_rate_beta(alpha,PE,S)

    c,F,dict_incoming,dict_outcoming,dict_mem_n,dict_mem_nodes,dict_mem_e,dict_incoming_noroot,dict_incoming_noroot_e=calc_nodal_concentrations(nx.Graph(K.G),B,Q,PE,beta,c0)
    # print(F)
    phi=calc_absorption(np.multiply(F,np.exp(beta)),beta,S,PE,alpha)

    return phi,c,F,beta,PE,S,alpha,R_sq,dict_incoming,dict_outcoming,dict_mem_n,dict_mem_nodes,dict_mem_e,dict_incoming_noroot,dict_incoming_noroot_e

def calc_PE_S_jacobian(flux_par,abs_par,G):

    B,BT,L,Q,F,C,R_sq,R,c=flux_par
    PE,S,alpha,beta,diff_R=abs_par
    M=len(C)
    ones=np.ones(M)
    INV=lina.pinv(np.dot(B,np.dot(np.diag(C),BT)))
    D=np.dot(np.dot(BT,INV),B)
    I=np.identity(M)

    SGN=np.ones(len(Q))
    SGN[np.where(Q<0.)[0]]=-1.
    Q=np.absolute(Q)
    f1= 2.*np.multiply(np.divide(PE,R),SGN)
    f2= np.multiply( np.divide(S,R)* (-1),SGN)
    f3= 4.* np.multiply(np.divide(Q,R),SGN)

    J_PE, J_S, J_Q= np.zeros((M,M)),np.zeros((M,M)),np.zeros((M,M))
    for i,c in enumerate(C):
        J_PE[i,:]= f1[i] * np.multiply(np.subtract( I[i,:], 2.* c * np.multiply( D[:,i], R_sq/R_sq[i] ) ),SGN)
        J_S[i,:]= f2[i] * np.multiply(np.subtract( 3.*I[i,:], 4.* c * np.multiply( D[:,i] , np.multiply( np.power(np.divide(Q[i],Q),2), np.power( np.divide( R,R[i]), 5 ) ) ) ),SGN)

        J_Q[i,:]= f3[i] * np.multiply(np.subtract( I[i,:], c*np.multiply( D[:,i], np.multiply( np.divide(L[i],L) , np.power( R_sq/R_sq[i] , 2 ) ) ) ),SGN)

    return J_PE,J_S,J_Q

def calc_PE_beta_jacobian(flux_par,abs_par,G,J_PE,J_S):

    B,BT,L,Q,F,C,R_sq,R,c=flux_par
    PE,S,alpha,beta,diff_R=abs_par
    M=len(C)
    J_beta=np.zeros((M,M))
    ones=np.ones(M)
    I=np.identity(M)

    f1=np.divide( np.power(alpha,2) ,6.*np.multiply(S,PE) )
    f2=np.power(np.divide(alpha,S),2)

    x=np.sqrt( np.add( np.add(ones,8.*np.divide(S,PE)),f1) )
    y=np.add( ones*48. , f2)

    B1= np.multiply(np.reciprocal(PE),np.subtract(beta,96.*np.divide(S,np.multiply(x,y))))
    B2= np.divide(np.add( np.multiply(np.divide(beta,S),f2) , np.reciprocal(x)*48. ),y)*2.
    B3= np.divide(np.multiply(f2,S),np.multiply(x,y))*2.

    for i,r in enumerate(R):

        J_beta[i,:] = np.add(J_beta[i,:],np.multiply( J_PE[i,:] , B1 ))
        J_beta[i,:] = np.add(J_beta[i,:],np.multiply( J_S[i,:] , B2 ))
        J_beta[i,:] = np.add(J_beta[i,:],np.multiply( I[i,:], B3/r))

    return J_beta

def calc_flux_matrices(Q,beta,PE,J_Q,J_beta,J_PE,dict_incoming,dict_outcoming,G):

    list_n=list(G.nodes())
    list_e=list(G.edges())
    M=len(list_e)
    N=len(list_n)
    idx_n_list=np.zeros(N,dtype=np.int8)
    root=find_root(G)
    sink=find_sink(G)
    idx_sink=G.nodes[sink]['label']
    idx_root=G.nodes[root]['label']

    # define tensor templates
    f1=np.add(np.ones(M),np.divide(beta,PE))
    f2=np.exp(-beta)
    flux_out= -np.ones(N)
    flux_in=np.multiply(f1,Q)
    for i,n in enumerate(list_n):
        idx_n_list[i]=G.nodes[n]['label']
        if idx_n_list[i] != idx_sink:
            flux_out[idx_n_list[i]]=np.sum(flux_in[dict_outcoming[n]])

    flux_in=np.multiply(f2,flux_in)
    A=np.outer(np.reciprocal(flux_out),flux_in)
    J_A=np.zeros((N,M,M))

    # define auxiliary varibles & lists, fix and sort signs
    X,Y,Z=J_A[:,:,:],J_A[:,:,:],J_A[:,:,:]
    # calculate auxiliary derivates
    DF_IN=np.zeros((M,M))
    DF_OUT=np.zeros((M,N))
    for j,e in enumerate(list_e):
        A1=np.multiply(J_Q[j,:],f1)
        A2=np.multiply(J_beta[j,:],np.multiply(Q,np.subtract(np.reciprocal(PE),f1)))
        A3=np.multiply(J_PE[j,:],np.divide(np.multiply(beta,Q),np.power(PE,2)))
        DF_IN[j,:]=np.multiply(f2,np.subtract(np.add(A1,A2),A3))

    for i,n in enumerate(list_n):
        idx_n=G.nodes[n]['label']
        idx_out=dict_outcoming[n]

        f3= np.divide(Q[idx_out],np.power(PE[idx_out],2))
        if idx_n != idx_sink and idx_n !=idx_root:
            for j,e in enumerate(list_e):
                DF_OUT[j,idx_n]=np.sum( np.add( np.multiply( J_Q[j,idx_out],f1[idx_out] ), np.multiply( f3, np.subtract( np.multiply( J_beta[j,idx_out],PE[idx_out] ), np.multiply( beta[idx_out],J_PE[j,idx_out] ) ) ) ) )

        elif idx_n==idx_sink:
            A[idx_n,:]=0.
    # calculate jacobian
    for i,e1 in enumerate(list_e):
        idx_e1=G.edges[e1]['label']
        for j,e2 in enumerate(list_e):
            idx_e2=G.edges[e2]['label']
            J_A[:,idx_e1,idx_e2]= np.subtract(  DF_IN[idx_e1,idx_e2]* np.reciprocal(flux_out) , flux_in[idx_e2]* np.divide(DF_OUT[idx_e1,:],np.power(flux_out,2)))
    J_A[idx_root,:,:]=np.zeros((M,M))
    J_A[idx_sink,:,:]=np.zeros((M,M))

    return A,J_A

def calc_flux_jacobian(flux_par,abs_par,dicts_par,G,J_PE,J_S,J_Q,J_beta):

    B,BT,L,Q,F,C,R_sq,R,c=flux_par
    PE,S,alpha,beta,diff_R=abs_par
    dict_incoming,dict_outcoming,dict_mem_n,dict_mem_nodes,dict_mem_e,dict_incoming_noroot,dict_incoming_noroot_e=dicts_par

    root=find_root(G)
    sink=find_sink(G)
    idx_root=G.nodes[root]['label']
    idx_sink=G.nodes[sink]['label']

    M=len(C)
    N=len(c)
    # calc concentration jacobian
    J_C=np.zeros((M,N))
    list_n=list(G.nodes())
    list_e=list(G.edges())
    idx_e_list=np.zeros(M,dtype=np.int8)
    idx_n_list=np.zeros(N,dtype=np.int8)
    Q=np.absolute(Q)
    A,J_A=calc_flux_matrices(Q,beta,PE,J_Q,J_beta,J_PE,dict_incoming,dict_outcoming,G)
    SUM_J_AC=np.zeros((N,M))
    for i,n in enumerate(list_n):
        idx_n_list[i]=G.nodes[n]['label']
        if idx_n_list[i] != idx_root and idx_n_list[i]!=idx_sink:
            for j,e in enumerate(list_e):
                idx_e_list[j]=( G.edges[e]['label'] )
                SUM_J_AC[idx_n_list[i],idx_e_list[j]]=np.sum( np.multiply( J_A[ idx_n_list[i] , idx_e_list[j], dict_incoming[n]], c[dict_mem_n[n]]))

    for i,n in enumerate(list_n):
        idx_n=idx_n_list[i]
        if idx_n != idx_root and idx_n != idx_sink:
            idx_in=dict_incoming_noroot_e[n]
            J_C[:,idx_n]=SUM_J_AC[idx_n,:]

            if len(dict_incoming_noroot[n])!=0:
                for j,e in enumerate(list_e):
                    SUM_JC=SUM_J_AC[:,idx_e_list[j]]
                    for k,m in enumerate(dict_incoming_noroot[n]):
                        J_C[idx_e_list[j],idx_n]=np.add( J_C[idx_e_list[j],idx_n],calc_iterative_increment(root,idx_root,G,c,dict_incoming_noroot_e,dict_incoming_noroot,A[idx_n,idx_in[k]],A,SUM_JC,m,n,G.nodes[m]['label']) )

    # calc flux jacobian
    J_F=np.zeros((M,M))
    identity=np.identity(M)
    A1=np.zeros(M)
    C1=np.zeros(M)
    for i,e1 in enumerate(list_e):
        for j,e2 in enumerate(list_e):
            idx_e=G.edges[e2]['label']
            A1[idx_e]=np.multiply(J_C[i,dict_mem_e[e2]],Q[idx_e])
            C1[idx_e]=c[dict_mem_e[e2]]
        J_F[i,:]=np.add( J_F[i,:], A1 )
        J_F[i,:]=np.add( J_F[i,:], np.multiply(J_Q[i,:], C1) )

    return J_F

def calc_iterative_increment(root,idx_root,G,c,dict_incoming_noroot_e,dict_incoming_noroot,A_aux,A,SUM_JC,m_in,n,idx_in):

    increment=A_aux*SUM_JC[idx_in]
    for k,m in enumerate(dict_incoming_noroot[m_in]):
        increment=np.add(increment,calc_iterative_increment(root,idx_root,G,c,dict_incoming_noroot_e,dict_incoming_noroot,A_aux*A[idx_in,dict_incoming_noroot_e[m_in][k]],A,SUM_JC,m,m_in,G.nodes[m]['label']))

    return increment

def calc_coefficient_jacobian(flux_par,abs_par,G,J_PE,J_S,J_beta):

    B,BT,L,Q,F,C,R_sq,R,c=flux_par
    PE,S,alpha,beta,diff_R=abs_par

    M=len(C)
    J_COEFF=np.zeros((M,M))
    I=np.identity(M)
    ones=np.ones(M)

    f1=np.divide(np.power(alpha,2),np.multiply(PE,S))
    f2_a=np.add(ones,f1/4.)
    f2=np.reciprocal(f2_a)
    f3=np.divide(S,beta)
    B1=2.*np.multiply(np.reciprocal(beta),f2)
    B2=np.multiply(B1,f3)
    B3=np.multiply(np.multiply(f1,np.power(f2,2)),np.add(ones,6.*f3))/12.
    for i,r in enumerate(R):

        J_COEFF[i,:]=np.add(J_COEFF[i,:],np.multiply(J_S[i,:],B1))
        J_COEFF[i,:]=np.subtract(J_COEFF[i,:],np.multiply(J_beta[i,:],B2))
        J_COEFF[i,:]=np.subtract(J_COEFF[i,:],np.multiply(I[i,:],B3/r))

    return J_COEFF
# @profile
def calc_absorption_jacobian(flux_par,abs_par,dicts_par,G):

    B,BT,L,Q,F,C,R_sq,R,c=flux_par
    PE,S,alpha,beta,diff_R=abs_par

    J_PE,J_S,J_Q=calc_PE_S_jacobian(flux_par,abs_par,G)
    J_beta=calc_PE_beta_jacobian(flux_par,abs_par,G,J_PE,J_S)
    J_c=calc_flux_jacobian(flux_par,abs_par,dicts_par,G,J_PE,J_S,J_Q,J_beta)
    # J_c=calc_flux_jacobian_topological(flux_par,abs_par,dicts_par,G,J_PE,J_S,J_Q,J_beta)
    f1=np.divide(np.power(alpha,2),np.multiply(PE,S))
    J_coeff=calc_coefficient_jacobian(flux_par,abs_par,G,J_PE,J_S,J_beta)
    ones=np.ones(len(C))

    exp=np.exp(-beta)
    F=np.divide(F,np.multiply(exp,np.add(ones,np.divide(beta,PE))))
    coeff1=np.subtract(ones,exp)
    coeff2=np.divide(np.subtract(2.*np.divide(S,beta),f1/12.),np.add(ones,f1/4.))
    phi_jacobian=np.zeros((len(C),len(C)))

    proxy1=np.zeros((len(C),len(C)))
    proxy2=np.zeros((len(C),len(C)))

    B1=np.multiply(coeff2,coeff1)
    B2=np.multiply(F,coeff1)
    B3=np.multiply(np.multiply(coeff2,F),exp)

    for i,R in enumerate(R):

        phi_jacobian[i,:]=np.add(phi_jacobian[i,:],np.multiply(J_c[i,:],B1))
        phi_jacobian[i,:]=np.add(phi_jacobian[i,:],np.multiply(J_coeff[i,:],B2))
        phi_jacobian[i,:]=np.add(phi_jacobian[i,:],np.multiply(J_beta[i,:],B3))

    return phi_jacobian

def supply_pattern(K,mode):

    phi0=np.zeros(len(K.C))
    if 'random' in mode:
        phi0=np.random.random(len(K.C))
    if 'constant' in mode:
        phi0=np.ones(len(K.C))
    if 'gradient' in mode:
        dist={}
        for j,e in enumerate(K.G.edges()):
            d=np.linalg.norm(np.add(K.G.nodes[e[0]]['pos'],K.G.nodes[e[1]]['pos']))*0.5
            phi0[j]=1./d

    return phi0

# stationary concentration profiles
class MySteps(object):
    def __init__(self, stepsize ):
        self.stepsize = stepsize
    def __call__(self, x):
        # print(x)
        rx=np.add(x,np.random.rand(len(x))*self.stepsize)
        return rx

def alpha_omega(G,j):
    for e in G.edges():

        if j == G.edges[e]['label']:
            # print('edge'+str(e))
            alpha=e[1]
            omega=e[0]

    return alpha,omega

def calc_flux_mixed_boundary_conditions(C,K):

    idx_sources=list(K.G.graph['sources'])
    idx_potential=list(K.G.graph['potentials'])
    D=np.dot(K.B,np.dot(np.diag(C),K.BT))

    # initial V,S are zero vectors with exception of given bopundary coordinates
    b=np.subtract(K.J,np.dot(D,K.V))
    L=D[:,:]
    n=len(L[0,:])
    for j in idx_potential:
            L[:,j]=np.zeros(n)
            L[j,j]=-1.
    X,RES,RG,si=np.linalg.lstsq(L,b,rcond=None)

    P=np.array(K.V[:])
    P[idx_sources]=X[idx_sources]
    S=np.array(K.J[:])
    S[idx_potential]=X[idx_potential]
    K.J=S[:]
    dP=np.dot(K.BT,P)
    Q=np.dot(np.diag(C),dP)

    return Q,dP,P

def calc_flows_pressures(K):

    OP=np.dot(K.B,np.dot(np.diag(K.C),K.BT))
    P,RES,RG,si=np.linalg.lstsq(OP,K.J,rcond=None)
    dP=np.dot(K.BT,P)
    Q=np.dot(np.diag(K.C),dP)
    K.dV=dP
    return Q, dP, P

def calc_PE(Q,K):

    R_sq=np.power(K.R,2)
    V=np.divide(Q,R_sq*np.pi)
    return np.multiply(V,K.l/K.D)

def generate_coherent_closure(H,list_e,x):

    idx=rd.sample(range(len(list_e)),x)
    for e in idx:
        H.remove_edge(*list_e[e])
    cond=nx.is_connected(H)

    for e in idx:
        H.add_edge(*list_e[e])
    return cond,idx

def calc_sq_flow_broken_link(K):

    # block p percent of the edges per realization
    X=100
    dV_sq,dF_sq=[],[]
    # AUX=nx.Graph(K.G)
    for i in range(X):
        idx=rd.choice(K.broken_sets)
        C_aux=np.array(K.C[:])
        C_aux[idx]=np.power(10.,-20)
        OP=np.dot(K.B,np.dot(np.diag(C_aux),K.BT))
        P,RES,RG,si=np.linalg.lstsq(OP,K.J,rcond=None)
        dP=np.dot(K.BT,P)
        Q=np.multiply(C_aux,dP)
        # Q,dP,P=calc_flux_mixed_boundary_conditions(C_aux,K)
        dV_sq.append(np.power(dP,2))
        dF_sq.append(np.power(Q,2))

    return np.sum(dV_sq,axis=0)/float(X), np.sum(dF_sq,axis=0)/float(X)

def calc_concentration_coefficients(PE,K):

    ones=np.ones(len(PE))
    x=np.sqrt(np.add(ones,np.divide(K.beta,np.power(PE,2))))

    a=np.multiply(0.5*PE, np.add( ones, x ) )
    b=np.multiply(0.5*PE, np.subtract( ones, x ) )

    return a,b

def update_stationary_operator(K):

    Q,dP,P=calc_flows_pressures(K)
    PE=calc_PE(Q,K)

    ones=np.ones(len(PE))
    N=nx.number_of_nodes(K.G)

    A=np.pi*np.power(K.R,2)*(K.D/K.l)
    x,z,sinh_x,cosh_x,coth_x,e_up,e_down=compute_flux_pars(K,PE)

    f1= np.multiply(z,A)
    f2= np.multiply(A,np.multiply(x,coth_x))*0.5
    f3= np.divide(np.multiply(A,x),sinh_x)*0.5

    B_eff=np.zeros((N,N))
    dict_in={}
    dict_out={}
    dict_edges={}
    dict_node_out={}
    dict_node_in={}

    for i,n in  enumerate(K.G.nodes()):
        dict_in[n]=[]
        dict_out[n]=[]
        dict_node_out[n]=np.where(K.B[i,:]>0)[0]
        dict_node_in[n]=np.where(K.B[i,:]<0)[0]

    for j,e in enumerate(K.G.edges()):

        alpha=e[1]
        omega=e[0]
        if K.B[alpha,j] > 0.:
            dict_edges[e]=[alpha,omega]
            dict_in[omega].append(alpha)
            dict_out[alpha].append(omega)

        elif K.B[alpha,j] < 0.:
            dict_edges[e]=[omega,alpha]
            dict_in[alpha].append(omega)
            dict_out[omega].append(alpha)

        else:
            print('whats going on')

    for i,n in enumerate(K.G.nodes()):
        B_eff[i,i]= np.sum(  np.add( np.multiply(K.B[i,:],f1),np.multiply(np.absolute(K.B[i,:]),f2))  )
        B_eff[i,dict_in[n]]= -np.multiply( e_up[dict_node_in[n]],f3[dict_node_in[n]] )
        B_eff[i,dict_out[n]]= -np.multiply( e_down[dict_node_out[n]],f3[dict_node_out[n]] )

    return B_eff,Q,[dict_edges,dict_in,dict_out]

def solve_inlet_peak(B_eff,K):

    # set_yourself_some_boundaries=[]
    idx_source=0
    idx_sink=0
    for i,n in enumerate(K.G.nodes()):
        if K.G.nodes[n]['source'] >  0:
            idx_source=i
        # else:
        #     set_yourself_some_boundaries.append((0.,None))
        if K.G.nodes[n]['source'] <  0:
            idx_sink=i

    idx_sinks=[i for i in range(len(K.J_C)) if i!=idx_sink]
    B_new=np.delete(np.delete(B_eff,idx_sink,axis=0),idx_source,axis=1)
    b=np.delete(B_eff,idx_sink,axis=0)
    S=np.subtract( K.J_C[idx_sinks]/K.C0, b[:,idx_source] )

    # fun = lambda x: np.linalg.norm(np.dot(B_new,x)-S)
    # dfun = lambda x: np.dot(np.subtract(np.dot(B_new,x),S),B_new)/np.linalg.norm(np.dot(B_new,x)-S)
    # n = len(K.J_C)-1
    # sol = sc.minimize( fun, 1.5*np.ones(n), method='L-BFGS-B', bounds=set_yourself_some_boundaries,jac=dfun)
    # sol = sc.basinhopping( fun, 0.1*np.ones(n),niter=100,T=10, minimizer_kwargs={'method':'L-BFGS-B', 'bounds':[(0.,None) for x in range(n)],'jac':dfun})
    # return sol

    A=np.linalg.inv(B_new)
    c=np.dot(A,S)
    # idx_sources=[i for i in range(len(K.J_C)) if i!=idx_source]
    # print( np.sum(np.multiply(B_eff[idx_sink,idx_sources],c))+B_eff[idx_sink,idx_source]*1.)
    idx=0
    for i,n in enumerate(K.G.nodes()):
        if K.G.nodes[n]['source'] >  0:
            K.G.nodes[n]['concentrations']=K.C0
        else:
            K.G.nodes[n]['concentrations']=c[idx]
            idx+=1

    return c,B_new,K

def solve_absorbing_boundary(B_eff,K):

    set_yourself_some_boundaries=[]
    idx=0
    for i,n in enumerate(K.G.nodes()):
        if K.G.nodes[n]['source'] <  0:
            idx=i
        else:
            set_yourself_some_boundaries.append((0.,None))

    n = len(K.J_C)-1
    idx_eff=[i for i in range(len(K.J_C)) if i!=idx]
    B_new=np.delete(np.delete(B_eff,idx,axis=0),idx,axis=1)
    S=K.J_C[idx_eff]

    # fun = lambda x: np.linalg.norm(np.dot(B_new,x)-S)
    # dfun = lambda x: np.dot(np.subtract(np.dot(B_new,x),S),B_new)/np.linalg.norm(np.dot(B_new,x)-S)
    # n = len(K.J_C)-1
    # sol = sc.minimize( fun, .5*np.ones(n), method='L-BFGS-B', bounds=set_yourself_some_boundaries,jac=dfun)
    # c,RES,RG,si=np.linalg.lstsq(B_new,S,rcond=None)
    A=np.linalg.inv(B_new)
    c=np.dot(A,S)
    # c=np.ones(n)
    # sol = sc.basinhopping( fun, 0.1*np.ones(n),niter=100,T=10, minimizer_kwargs={'method':'L-BFGS-B', 'bounds':[(0.,None) for x in range(n)],'jac':dfun})
    idx=0
    for i,n in enumerate(K.G.nodes()):
        # if K.G.nodes[n]['source'] >  0:
            # K.G.nodes[n]['concentrations']=0.99
            # K.G.nodes[n]['concentrations']=K.C0
        if K.G.nodes[n]['source'] <  0:
            K.G.nodes[n]['concentrations']=0.
        else:
            K.G.nodes[n]['concentrations']=c[idx]
            idx+=1
    # return sol
    return c,B_new,K

def calc_profile_concentration(K,mode):

    B_eff,Q,dicts=update_stationary_operator(K)

    # use absorbing boundaries + reduced equation system
    if mode=='absorbing_boundary':
        c,B_new,K=solve_absorbing_boundary(B_eff,K)

    # use inlet delta peak + reduced equation system
    if mode=='mixed_boundary':
        c,B_new,K=solve_inlet_peak(B_eff,K)

    return c,Q,dicts,B_new,K

def calc_stationary_concentration(K,mode):

    c,Q,dicts,B_new,K=calc_profile_concentration(K,mode)

    # set containers
    A=np.multiply(K.R,K.R)*np.pi*(K.D/K.l)
    m=nx.number_of_edges(K.G)
    J_a,J_b=np.zeros(m),np.zeros(m)
    phi=np.zeros(m)
    ones=np.ones(m)
    c_a,c_b=np.ones(m),np.ones(m)

    # calc coefficients
    for j,e in enumerate(K.G.edges()):
        a,b=dicts[0][e]
        c_a[j]=K.G.nodes[a]['concentrations']
        c_b[j]=K.G.nodes[b]['concentrations']

    PE=calc_PE(Q,K)
    x,z,sinh_x,cosh_x,coth_x,e_up,e_down=compute_flux_pars(K,PE)

    f1= np.divide(x,sinh_x)*0.5
    f1_up=np.multiply( f1,e_up )
    f1_down=np.multiply( f1,e_down )

    F1=np.add( np.subtract( np.multiply(x,coth_x)*0.5 , f1_up), z)
    F2=np.subtract( np.subtract( np.multiply(x,coth_x)*0.5 , f1_down), z)

    f2= np.add( z, np.multiply(x,coth_x)*0.5 )
    f3= np.subtract( z ,np.multiply(x,coth_x)*0.5 )

    # calc edgewise absorption
    phi=np.add(np.multiply(c_a,F1) ,np.multiply( c_b,F2 ))
    phi=np.multiply( phi, A)

    J_a=np.multiply(A, np.subtract( np.multiply(f2,c_a) , np.multiply(f1_down,c_b )) )
    J_b=np.multiply(A, np.add( np.multiply(f3,c_b), np.multiply(f1_up,c_a )) )

    return c,J_a,J_b,phi

def calc_absorption(R, *args):

    # unzip
    Q,dict_edges,K= args

    # set containers
    m=nx.number_of_edges(K.G)
    phi=np.zeros(m)
    ones=np.ones(m)
    c_a,c_b=np.ones(m),np.ones(m)
    # calc coefficients
    for j,e in enumerate(K.G.edges()):
        a,b=dict_edges[e]
        c_a[j]=K.G.nodes[a]['concentrations']
        c_b[j]=K.G.nodes[b]['concentrations']

    PE=calc_PE(Q,K)
    x,z,sinh_x,cosh_x,coth_x,e_up,e_down=compute_flux_pars(K,PE)

    f1= np.divide(x,sinh_x)*0.5
    F1=np.add( np.subtract( np.multiply(x,coth_x)*0.5 , np.multiply( f1,e_up )), z)
    F2=np.add( np.subtract( np.multiply(x,coth_x)*0.5 , np.multiply( f1,e_down )), -z)

    # calc edgewise absorption
    phi=np.add(np.multiply(c_a,F1) ,np.multiply( c_b,F2 ))
    A=np.pi*np.multiply(R,R)*(K.D/K.l)

    return np.multiply( A, phi )

def calc_flux_jacobian(R,*args):

    # unzip parameters
    Q,PE,L,K= args

    # init containers
    M=len(Q)
    I=np.identity(M)
    J_PE, J_Q= np.zeros((M,M)),np.zeros((M,M))

    # set coefficients
    f1= 2.*np.divide(PE,R)
    f2= 4.* np.divide(Q,R)
    R_sq=np.power(R,2)
    C=np.power(R,4)*K.k
    INV=lina.pinv(np.dot(K.B,np.dot(np.diag(C),K.BT)))
    D=np.dot(np.dot(K.BT,INV),K.B)

    # calc jacobian
    for i,c in enumerate(C):
        J_PE[i,:]= f1[i] * np.subtract( I[i,:], 2.* c * np.multiply( D[:,i], R_sq/R_sq[i] ) )
        J_Q[i,:]= f2[i] * np.subtract( I[i,:], c*np.multiply( D[:,i], np.multiply( np.divide(L[i],L) , np.power( R_sq/R_sq[i] , 2 ) ) ) )

    return J_PE,J_Q

def calc_concentration_jacobian( R,*args ):

    # unzip
    PE,J_PE,c,dicts,K,mode=args
    dict_edges,dict_in,dict_out=dicts
    # set containers

    m=nx.number_of_edges(K.G)
    N=nx.number_of_nodes(K.G)
    ones=np.ones(m)
    J_C=np.zeros((m,N))
    dict_node_in={}
    dict_node_out={}

    # set coefficients
    A=np.pi*np.multiply(R,R)*(K.D/K.l)
    x,z,sinh_x,cosh_x,coth_x,e_up,e_down=compute_flux_pars(K,PE)
    f1= np.multiply(z,A)
    f2= np.multiply(np.multiply(x,coth_x),A)*0.5
    f3= np.divide(np.multiply(A,x),sinh_x)*0.5
    B_eff=np.zeros((N,N))
    # root=0
    idx_source=0
    idx_sink=0

    for i,n in enumerate(K.G.nodes()):
        dict_node_out[n]=np.where(K.B[i,:]>0)[0]
        dict_node_in[n]=np.where(K.B[i,:]<0)[0]
        b=K.B[i,:]
        B_eff[i,i]= np.sum(  np.add( np.multiply(b,f1), np.multiply(np.absolute(b),f2))  )
        B_eff[i,dict_in[n]]= - np.multiply( e_up[dict_node_in[n]],f3[dict_node_in[n]] )
        B_eff[i,dict_out[n]]= - np.multiply( e_down[dict_node_out[n]],f3[dict_node_out[n]] )
        # absorbing boundary
        if mode=='absorbing_boundary':
            if K.G.nodes[n]['source'] <  0:
                root=i

        # inlet peak
        if mode=='mixed_boundary':
            if K.G.nodes[n]['source'] >  0:
                idx_source=i
            if K.G.nodes[n]['source'] <  0:
                idx_sink=i

    j_coth_x=np.power(np.divide(coth_x,cosh_x),2)
    f2= np.multiply(x,coth_x)*0.5
    f4=np.subtract( np.multiply( np.divide(z,x), coth_x) ,  np.multiply( z,j_coth_x )*0.5 )

    f_up=np.divide(np.multiply( e_up ,x ), sinh_x )*0.5
    f_down=np.divide( np.multiply( e_down ,x ), sinh_x )*0.5
    f5=np.divide( np.multiply(A, e_up), sinh_x )
    f6=np.divide( np.multiply(A, e_down), sinh_x )

    J_f_up=-np.multiply( f5, np.subtract( np.add( np.divide(z,x), x*0.25 ), np.multiply(z,coth_x)*0.5 ))
    J_f_down=-np.multiply( f6, np.subtract( np.subtract( np.divide(z,x), x*0.25 ), np.multiply(z,coth_x)*0.5 ))

    # inlet peak
    if mode=='mixed_boundary':
        idx_sinks=[i for i in range(len(K.J_C)) if i!=idx_sink]
        B_new=np.delete(np.delete(B_eff,idx_sink,axis=0),idx_source,axis=1)
        c=c[idx_sinks]
    #
    # absorbing boundary
    if mode=='absorbing_boundary':
        idx_eff=[i for i in range(N) if i!=root]
        B_new=np.delete(np.delete(B_eff,root,axis=0),root,axis=1)
        c=c[idx_eff]

    inv_B=np.linalg.inv(B_new)
    for j,e in enumerate(K.G.edges()):
        JB_eff=np.zeros((N,N))
        J_A=np.zeros((m,m))
        J_A[j,j]=2.*np.pi*R[j]*(K.D/K.l)
        for i,n in enumerate(K.G.nodes()):

            b=K.B[i,:]
            JB_eff[i,i]=  np.sum( np.multiply( J_A, np.add( np.multiply(b,z), np.multiply(np.absolute(b),f2))  ) )+np.sum( np.multiply( J_PE[j,:], np.multiply(A, np.add( b*0.5, np.multiply(np.absolute(b),f4) ) ) ))
            JB_eff[i,dict_out[n]]= np.subtract( np.multiply( J_PE[j,dict_node_out[n]] , J_f_down[dict_node_out[n]] ) , np.multiply( J_A[j,dict_node_out[n]], f_down[dict_node_out[n]] ))
            JB_eff[i,dict_in[n]]=  np.subtract( np.multiply( J_PE[j,dict_node_in[n]] , J_f_up[dict_node_in[n]] ) , np.multiply( J_A[j,dict_node_in[n]], f_up[dict_node_in[n]] ))

        # absorbing boundary
        if mode=='absorbing_boundary':
            JB_new=np.delete(np.delete(JB_eff,root,axis=0),root,axis=1)
            J_C[j,idx_eff]=-np.dot(inv_B, np.dot( JB_new, c ))

        # inlet peak
        if mode=='mixed_boundary':
            JB_new=np.delete(np.delete(JB_eff,idx_sink,axis=0),idx_source,axis=1)
            J_C[j,idx_sinks]=-np.dot(inv_B, np.dot( JB_new, c ))

    return J_C

def calc_absorption_jacobian(R, *args):

    # unzip
    B_eff,Q,dicts,K,mode= args

    # set containers
    m=nx.number_of_edges(K.G)
    n=nx.number_of_nodes(K.G)
    ones=np.ones(m)
    L=ones*K.l
    J_phi= np.zeros((m,m))
    phi=np.zeros(m)
    c_a,c_b,c_n=np.zeros(m),np.zeros(m),np.zeros(n)
    alphas,omegas=[],[]
    dict_edges=dicts[0]
    # calc coefficients
    for j,e in enumerate(K.G.edges()):
        a,b=dict_edges[e]
        c_a[j]=K.G.nodes[a]['concentrations']
        c_b[j]=K.G.nodes[b]['concentrations']
        alphas.append(a)
        omegas.append(b)
    for i,n in enumerate(K.G.nodes()):
        c_n[i]=K.G.nodes[n]['concentrations']

    PE=calc_PE(Q,K)
    x,z,sinh_x,cosh_x,coth_x,e_up,e_down=compute_flux_pars(K,PE)

    f1= 0.5*np.divide(x,sinh_x)
    F1=np.add( np.subtract( 0.5*np.multiply(x,coth_x) , np.multiply( f1,e_up )), z )
    F2=np.subtract( np.subtract( 0.5*np.multiply(x,coth_x) , np.multiply( f1,e_down )), z)

    f2_up=np.subtract( np.multiply( np.divide(PE,x), np.subtract( cosh_x, e_up )), np.divide(z,sinh_x))
    f3_up=np.add( np.multiply( e_up, np.subtract( np.multiply(coth_x,z), 0.5*x ) ) , sinh_x)
    f2_down=np.subtract(np.multiply( np.divide(PE,x), np.subtract( cosh_x, e_down )), np.divide(z,sinh_x))
    f3_down=np.subtract( np.multiply( e_down, np.add( np.multiply(coth_x,z), 0.5*x  ) ), sinh_x )

    F3= 0.5*np.divide( np.add(f2_up, f3_up) , sinh_x)
    F4= 0.5*np.divide( np.add(f2_down, f3_down) , sinh_x)
    phi=np.add( np.multiply(c_a,F1) ,np.multiply(c_b,F2 ) )

    # calc jacobian
    J_PE,J_Q= calc_flux_jacobian(R,Q,PE,L,K)
    A=np.pi*np.multiply(R,R)*(K.D/K.l)
    J_A=2.*np.pi*np.diag(R)*(K.D/K.l)
    J_C=calc_concentration_jacobian( R,PE,J_PE,c_n,dicts,K,mode )

    qa=np.multiply(A,c_a)
    qb=np.multiply(A,c_b)
    q1=np.multiply( A, F1 )
    q2=np.multiply( A, F2 )

    for j,e in enumerate(K.G.edges()):
        J_phi[j,:]=np.add(J_phi[j,:],np.multiply( J_A[j,:], phi))

        J_phi[j,:]=np.add(J_phi[j,:], np.multiply( J_C[j,alphas], q1 ))
        J_phi[j,:]=np.add(J_phi[j,:], np.multiply( J_C[j,omegas], q2 ))

        J_phi[j,:]=np.add(J_phi[j,:],np.multiply( J_PE[j,:], np.multiply(qa,F3)))
        J_phi[j,:]=np.add(J_phi[j,:],np.multiply( J_PE[j,:], np.multiply(qb,F4)))

    return J_phi

def compute_flux_pars(K,PE):

    x=np.sqrt( np.add( np.power(PE,2),K.beta ) )
    z=PE*0.5
    sinh_x=np.sinh(x*0.5)
    cosh_x=np.cosh(x*0.5)
    coth_x=np.reciprocal(np.tanh(x*0.5))
    e_up=np.exp(z)
    e_down=np.exp(-z)

    return x,z,sinh_x,cosh_x,coth_x,e_up,e_down

# optimize networks
def calc_absorption_cost(R,*args):

    K=args[0]
    m=len(K.R)
    K.C=np.power(R[:],4)*K.k
    K.R=R[:]
    c,Q,dicts,B_new,K=calc_profile_concentration(K)

    phi=calc_absorption(R, Q, dicts[0],K)
    J_phi=calc_absorption_jacobian(R, B_new,Q,dicts,K)

    phi0=np.ones(m)*K.phi0
    F=np.sum(np.power(np.subtract(phi,phi0),2))
    DF=np.array( [ 2.*np.sum( np.multiply( np.subtract(phi,phi0), J_phi[j,:] ) ) for j in range(m) ] )
    # print(F)
    return F,DF
    # return F

def calc_absorption_dissipation_cost(R,*args):

    K=args[0]
    m=len(K.R)
    K.C=np.power(R[:],4)*K.k
    c,Q,dicts,B_new,S,K=calc_profile_concentration(K)
    B,BT=K.get_incidence_matrices()
    Q,dP,P=calc_flows_pressures(B,BT,K.C,K.J)

    phi=calc_absorption(R, Q, dicts[0],K)
    J_phi=calc_absorption_jacobian(R, B_new,Q,dicts,K)

    phi0=np.ones(m)*K.phi0
    F=np.sum(np.power(np.subtract(phi,phi0),2))+ K.alpha_0*np.sum(np.multiply(np.power(dP,2),np.power(R,4)))
    DF=np.array( [ 2.*np.sum( np.multiply( np.subtract(phi,phi0), J_phi[j,:] ) ) for j in range(m) ] )
    DF=np.subtract(DF,4.*K.alpha_0*np.multiply(np.power(dP,2),np.power(R,3)))
    # print(F)
    return F,DF

def calc_absorption_volume_cost(R,*args):

    K=args[0]
    m=len(K.R)
    K.C=np.power(R[:],4)*K.k
    c,Q,dicts,B_new,S,K=calc_profile_concentration(K)

    phi=calc_absorption(R, Q, dicts[0],K)
    J_phi=calc_absorption_jacobian(R, B_new,Q,dicts,K)

    phi0=np.ones(m)*K.phi0
    F=np.sum(np.power(np.subtract(phi,phi0),2))+K.alpha*np.sum(np.power(R,2))
    DF=np.array( [ 2.*np.sum( np.multiply( np.subtract(phi,phi0), J_phi[j,:] ) ) for j in range(m) ] )
    DF=np.add(DF,K.alpha*R)

    return F,DF

def calc_absorption_dissipation_volume_cost(R,*args):

    K=args[0]
    m=len(K.R)
    K.C=np.power(R[:],4)*K.k
    K.R=R[:]
    c,Q,dicts,B_new,S,K=calc_profile_concentration(K)

    phi=calc_absorption(R, Q, dicts[0],K)
    J_phi=calc_absorption_jacobian(R, B_new,Q,dicts,K)
    phi0=np.ones(m)*K.phi0

    B,BT=K.get_incidence_matrices()
    Q,dP,P=calc_flows_pressures(B,BT,K.C,K.J)
    sq_R=np.power(R,2)
    F=np.sum(np.power(np.subtract(phi,phi0),2)) + K.alpha_1*np.sum( np.add( np.multiply(np.power(dP,2),np.power(R,4)), K.alpha_0*sq_R ) )
    DF1=np.array( [ 2.*np.sum( np.multiply( np.subtract(phi,phi0), J_phi[j,:] ) ) for j in range(m) ] )
    DF2=2.*K.alpha_1*np.multiply( np.subtract( np.ones(m)*K.alpha_0 ,2.*np.multiply(np.power(dP,2),sq_R) ),R )

    return F,np.add(DF1,DF2)
    # return F

def calc_dissipation_volume_cost(R,*args):

    K=args[0]
    m=len(K.R)
    K.C=np.power(R[:],4)*K.k
    B,BT=K.get_incidence_matrices()
    Q,dP,P=calc_flows_pressures(B,BT,K.C,K.J)

    sq_R=np.power(R,2)
    F = np.sum( np.add( K.alpha_1*np.multiply( np.power(dP,2),np.power(R,4)), sq_R*K.alpha_0 ))
    DF=2.*np.subtract( np.ones(m)*K.alpha_0, 2.*K.alpha_1*np.multiply( np.power(dP,2),sq_R ) )
    DF=np.multiply(DF,R)

    return F,DF

def optimize_network_targets(K,mode):

    m=nx.number_of_edges(K.G)
    b0=1e-25
    mysteps=MySteps(1.)

    # sol=sc.basinhopping(calc_absorption_cost,np.ones(m),niter=100,T=1.,take_step=mysteps,minimizer_kwargs={'method':'L-BFGS-B', 'bounds':[(b0,None) for x in range(m)],'args':(K),'jac':True})
    # c,Q,dicts,B_new,S,K=calc_profile_concentration(K)
    if mode=='uptake' :
        sol=sc.basinhopping(calc_absorption_cost,K.R,niter=10,T=10.,take_step=mysteps,minimizer_kwargs={'method':'L-BFGS-B', 'bounds':[(b0,None) for x in range(m)],'args':(K),'jac':True,'tol':1e-10})
    if mode=='uptake+volume' :
        sol=sc.minimize(calc_absorption_volume_cost,K.R, method='L-BFGS-B', bounds=[(b0,None) for x in range(m)],args=(K),jac=True)
    if mode=='uptake+dissipation' :
        sol=sc.minimize(calc_absorption_dissipation_cost,K.R, method='L-BFGS-B', bounds=[(b0,None) for x in range(m)],args=(K),jac=True,tol=1e-10)
        # sol=sc.basinhopping(calc_absorption_dissipation_cost,K.R,niter=100,T=1.,take_step=mysteps,minimizer_kwargs={'method':'L-BFGS-B', 'bounds':[(b0,None) for x in range(m)],'args':(K),'jac':True})
    if mode=='uptake+dissipation+volume' :
        sol=sc.basinhopping(calc_absorption_dissipation_volume_cost,K.R,niter=100,T=10.,take_step=mysteps,minimizer_kwargs={'method':'L-BFGS-B', 'bounds':[(b0,None) for x in range(m)],'args':(K),'jac':True,'tol':1e-10})
        # sol=sc.minimize(calc_absorption_dissipation_volume_cost,K.R, method='L-BFGS-B', bounds=[(b0,None) for x in range(m)],args=(K),jac=True,tol=1e-10)
        # sol=sc.minimize(calc_absorption_dissipation_volume_cost,K.R, method='L-BFGS-B', bounds=[(b0,None) for x in range(m)],args=(K))
    if mode=='dissipation+volume' :
        sol=sc.basinhopping(calc_dissipation_volume_cost,K.R,niter=10,T=10.,take_step=mysteps,minimizer_kwargs={'method':'L-BFGS-B', 'bounds':[(b0,None) for x in range(m)],'args':(K),'jac':True,'tol':1e-10})
        # sol=sc.minimize(calc_dissipation_volume_cost,np.ones(m), method='L-BFGS-B', bounds=[(b0,None) for x in range(m)],args=(K),jac=True,tol=1e-10)
    # sol=sc.minimize(calc_absorption_cost,np.ones(m),method='L-BFGS-B', bounds=[(b0,None) for x in range(m)],args=(K))

    return sol

# optimize networks with gradient descent
def flatlining(t,R,K,mode):
    B,BT=K.get_incidence_matrices()
    mode_boundary='absorbing_boundary'
    K.R=R
    sq_R=np.power(R,2)
    K.C=np.power(sq_R,2)*K.k
    c,Q,dicts,B_new,K=calc_profile_concentration(K,mode_boundary)
    phi=calc_absorption(K.R, Q, dicts[0],K)
    phi0=np.ones(len(R))*K.phi0
    x=16
    if 'absorption+shear' ==  mode:

        Q,dP,X=calc_flows_pressures(K)
        F=np.sum(np.power(np.subtract(phi,phi0),2))+np.sum( np.add( K.alpha_1*np.multiply(np.power(dP,2),np.power(sq_R,2)), K.alpha_0*sq_R ))
    elif 'shear+absorption' ==  mode:
        Q,dP,X=calc_flows_pressures(K)
        F=K.alpha_0*np.sum(np.power(np.subtract(phi,phi0),2))+np.sum( np.add( K.k*np.multiply(np.power(dP,2),np.power(sq_R,2)), np.sqrt(K.k)*K.alpha_1*sq_R ))

    elif 'absorption' ==  mode:
        F=np.sum(np.power(np.subtract(phi,phi0),2))

    elif 'shear' ==  mode:
        x=20
        F=np.sum( np.add( K.alpha_1*np.multiply(np.power(K.dV,2)*K.k,np.power(sq_R,2)), K.alpha_0*sq_R*np.sqrt(K.k) ))

    elif 'shear+fluctuation' ==  mode:
        x=30
        # F=np.multiply(2.*np.subtract(K.alpha_1*np.multiply(K.dV_sq,np.power(sq_R,2)),K.alpha_0*np.ones(len(Q))), R)
        F=R[:]
    elif 'absorption+volumetric' ==  mode:
        ones=np.ones(len(K.dict_volumes.values()))
        phi0=ones*K.phi0
        dphi=ones
        sum_phi=0.
        for i,v in enumerate(K.dict_volumes.keys()):
            dphi[v]=np.sum(np.multiply(K.weights[K.dict_volumes[v]],phi[K.dict_volumes[v]]))-phi0[v]
            sum_phi+=np.sum(np.multiply(K.weights[K.dict_volumes[v]],phi[K.dict_volumes[v]]))
        F=np.sum(np.power(dphi,2))

    dF=np.subtract(F,K.F)
    K.F=F
    # dR=update_stimuli(t,R,K,mode)
    # z=np.round( np.sum(np.power(dR,2)) ,16)
    z=np.round( np.sum(np.power(np.divide(dF,F),2)) ,x  )

    return z

flatlining.terminal=True
flatlining.direction = -1

def update_stimuli(t,R,K,mode):

    mode_boundary='absorbing_boundary'
    K.C=np.power(R,4)*K.k
    K.R=R
    m=len(R)
    dr=np.zeros(m)

    if 'absorption+shear' ==  mode:

        c,Q,dicts,B_new,K=calc_profile_concentration(K,mode_boundary)
        phi=calc_absorption(R, Q, dicts[0],K)
        J_phi=calc_absorption_jacobian( R, B_new,Q,dicts,K ,mode_boundary)

        Q,dP,X=calc_flows_pressures(K)
        phi0=np.ones(m)*K.phi0

        for i in range(m):
            dr[i]=-2.*np.sum( np.multiply(np.subtract(phi,phi0),J_phi[i,:]))

        DR=np.multiply(2.*np.subtract(K.alpha_1*np.power(np.multiply(dP,R),2),K.alpha_0*np.ones(len(phi))), R)
        dr=np.add(dr,2.*DR)

    elif 'absorption' ==  mode:

        c,Q,dicts,B_new,K=calc_profile_concentration(K,mode_boundary)

        phi=calc_absorption(R, Q, dicts[0],K)
        J_phi=calc_absorption_jacobian(R, B_new,Q,dicts,K,mode_boundary)
        # ones=np.ones(m)

        phi0=np.ones(m)*K.phi0
        # for i in range(len(R)):
            # dr[i]=-2.*np.sum( np.multiply(np.subtract(phi/K.phi0,ones),J_phi[i,:]/K.phi0))
        for i in range(len(R)):
            dr[i]=-2.*np.sum( np.multiply(np.subtract(phi,phi0),J_phi[i,:]))

    # elif 'absorption+volumetric' ==  mode:
    #     ones=np.ones(len(K.dict_volumes.values()))
    #     c,Q,dicts,B_new,K=calc_profile_concentration(K,mode_boundary)
    #
    #     phi=calc_absorption(R, Q, dicts[0],K)
    #     J_phi=calc_absorption_jacobian(R, B_new,Q,dicts,K)
    #     phi0=ones*K.phi0
    #     dphi=ones
    #
    #     for i,v in enumerate(K.dict_volumes.keys()):
    #         dphi[v]=np.sum(np.multiply(K.weights[K.dict_volumes[v]],phi[K.dict_volumes[v]]))-phi0[v]
    #
    #     for j,e in enumerate(K.G.edges()):
    #         for i,v in enumerate(K.dict_volumes.keys()):
    #             dr[j]-=2.*dphi[v]*np.sum(np.multiply(K.weights[K.dict_volumes[v]],J_phi[j,K.dict_volumes[v]]))
    # elif 'absorption+volumetric+volume' ==  mode:
    #     ones=np.ones(len(K.dict_volumes.values()))
    #     c,Q,dicts,B_new,K=calc_profile_concentration(K)
    #
    #     phi=calc_absorption(R, Q, dicts[0],K)
    #     J_phi=calc_absorption_jacobian(R, B_new,Q,dicts,K)
    #     phi0=ones*K.phi0
    #     dphi=ones
    #
    #     for i,v in enumerate(K.dict_volumes.keys()):
    #         dphi[v]=np.sum(np.multiply(K.weights[K.dict_volumes[v]],phi[K.dict_volumes[v]]))-phi0[v]
    #
    #     for j,e in enumerate(K.G.edges()):
    #         for i,v in enumerate(K.dict_volumes.keys()):
    #             dr[j]-=2.*dphi[v]*np.sum(np.multiply(K.weights[K.dict_volumes[v]],J_phi[j,K.dict_volumes[v]]))
    #     DR=K.alpha_0* R
    #     dr=np.subtract(np.multiply(dr,R),DR)
    elif 'shear' ==  mode:

        B,BT=K.get_incidence_matrices()
        Q,dP,X=calc_flows_pressures(K)

        DR=np.multiply(2.*np.subtract(K.alpha_1*np.power(np.multiply(dP,R),2),K.alpha_0*np.ones(len(Q))), R)
        dr=np.add(dr,DR)
    elif 'shear+fluctuation' ==  mode:

        B,BT=K.get_incidence_matrices()

        I=init_integration.integrate_stochastic()
        I.M=nx.number_of_edges(K.G)
        I.N=nx.number_of_nodes(K.G)
        I.noise=K.noise
        # I.fraction=K.fraction
        # I.setup_random_fluctuations_terminals(K)
        I.setup_random_fluctuations_effective(K)
        dV_sq, F_sq=I.calc_sq_flow(K.C,B,BT)

        DR=np.multiply(np.subtract(2.*K.alpha_1*np.multiply(dV_sq,np.power(R,2)),K.alpha_0*np.ones(len(R))), R)
        dr=np.add(dr,2.*DR)

    else:
        sys.abort('no legitimate mode')

    return dr

def propagate_system_standard(K,t_span,t_eval,mode):

    nsol=si.solve_ivp(update_stimuli,t_span,K.R,args=( K,mode ),t_eval=t_eval,method='LSODA',event=flatlining)

    sol=np.transpose(np.array(nsol.y))
    K.R=sol[-1,:]
    K.C=np.power(K.R,4)*K.k
    K.set_network_attributes()

    return sol,K

def evaluate_timeline(nsol,K,mode):

    mode_boundary='absorbing_boundary'
    dict_output={}
    m=len(K.R)
    phi0=np.ones(m)*K.phi0
    F=[]
    PHI=[]
    DP=[]
    C=[]
    PE=[]
    SUM=[]

    for i in range(len(nsol[:,0])):

        K.R=nsol[i,:]
        sq_R=np.power(K.R,2)
        K.C=np.power(sq_R,2)*K.k
        c,Q,dicts,B_new,K=calc_profile_concentration(K,mode_boundary)
        phi=calc_absorption( K.R, Q, dicts[0],K )
        PHI.append(phi)
        C.append(c)
        PE.append(calc_PE(Q,K))

        if 'absorption+shear' ==  mode:

            Q,dP,X=calc_flows_pressures(K)
            F.append(np.sum(np.power(np.subtract(phi,phi0),2))+np.sum( np.add( K.alpha_1*np.multiply(np.power(dP,2),np.power(sq_R,2)), K.alpha_0*sq_R )))
            SUM.append(np.sum(phi))
        elif 'shear+absorption' ==  mode:
            Q,dP,X=calc_flows_pressures(K)
            F.append(K.alpha_0*np.sum(np.power(np.subtract(phi,phi0),2))+np.sum( np.add( K.k*np.multiply(np.power(dP,2),np.power(sq_R,2)), np.sqrt(K.k)*K.alpha_1*sq_R )))
            SUM.append(np.sum(phi))

        elif 'absorption' ==  mode:
            F.append(np.sum(np.power(np.subtract(phi,phi0),2)))
            SUM.append(np.sum(phi)/(K.num_sources))
            # F.append(np.sum(np.power(np.subtract(phi/K.phi0,ones),2)))
        elif 'shear' ==  mode:

            Q,dP,X=calc_flows_pressures(K)
            F.append(np.sum( np.add( K.alpha_1*np.multiply(np.power(dP,2),np.power(sq_R,2)), K.alpha_0*sq_R )))
            SUM.append(np.sum(phi))
        elif 'shear+fluctuation' ==  mode:
            # I=init_integration.integrate_stochastic()
            # I.M=nx.number_of_edges(K.G)
            # I.N=nx.number_of_nodes(K.G)
            # I.noise=K.noise
            # I.fraction=K.fraction
            # I.setup_random_fluctuations_terminals(K)
            # I.setup_random_fluctuations_effective(K)
            # dV_sq, F_sq=calc_sq_flow_broken_link(K)
            # F.append(np.sum( np.add( K.alpha_1*np.multiply(dV_sq,np.power(sq_R,2)), K.alpha_0*sq_R )))
            SUM.append(np.sum(phi))
            F.append(0.)
        elif 'absorption+volumetric' ==  mode:
            ones=np.ones(len(K.dict_volumes.values()))
            phi0=ones*K.phi0
            dphi=ones
            sum_phi=0.
            for i,v in enumerate(K.dict_volumes.keys()):
                dphi[v]=np.sum(np.multiply(K.weights[K.dict_volumes[v]],phi[K.dict_volumes[v]]))-phi0[v]
                sum_phi+=np.sum(np.multiply(K.weights[K.dict_volumes[v]],phi[K.dict_volumes[v]]))
            F.append(np.sum(np.power(dphi,2)))
            SUM.append(sum_phi)
        elif 'absorption+volumetric+volume' ==  mode:
            ones=np.ones(len(K.dict_volumes.values()))
            phi0=ones*K.phi0
            dphi=ones
            sum_phi=0.
            for i,v in enumerate(K.dict_volumes.keys()):
                dphi[v]=np.sum(np.multiply(K.weights[K.dict_volumes[v]],phi[K.dict_volumes[v]]))-phi0[v]
                sum_phi+=np.sum(np.multiply(K.weights[K.dict_volumes[v]],phi[K.dict_volumes[v]]))
            F.append(np.sum(np.power(dphi,2)))
            SUM.append(sum_phi)

    dict_output['sum']=SUM
    dict_output['radii_temporal']=nsol
    dict_output['cost']=F
    dict_output['uptake']=PHI
    dict_output['concentration']=C
    dict_output['PE']=PE

    return dict_output

# have a routine which constantly produces output/reports from which new simulations may be started

def update_stimuli_report(t,R,K,mode):

    mode_boundary='absorbing_boundary'
    K.C=np.power(R,4)*K.k
    K.R=R
    m=len(R)
    dr=np.zeros(m)

    if 'absorption+shear' ==  mode:

        c,Q,dicts,B_new,K=calc_profile_concentration(K,mode_boundary)
        phi=calc_absorption(R, Q, dicts[0],K)
        J_phi=calc_absorption_jacobian( R, B_new,Q,dicts,K ,mode_boundary)
        PE=calc_PE(Q,K)

        Q,dP,X=calc_flows_pressures(K)
        phi0=np.ones(m)*K.phi0

        for i in range(m):
            dr[i]=-2.*np.sum( np.multiply(np.subtract(phi,phi0),J_phi[i,:]))

        DR=np.multiply(2.*np.subtract(K.alpha_1*np.power(np.multiply(dP,R),2),K.alpha_0*np.ones(len(phi))), R)
        dr=np.add(dr,2.*DR)

        dict_output={}
        dict_output['radii']=R
        dict_output['PE']=PE

        K.set_network_attributes()
        nx.write_gpickle(K.G,op.join(K.DIR_OUT_DATA,'graph_backup'))
        f = open(op.join(K.DIR_OUT_DATA,'dict_dynamic_report.pkl'),"wb")
        pickle.dump(dict_output,f)
        f.close()

    elif 'shear+absorption' ==  mode:

        c,Q,dicts,B_new,K=calc_profile_concentration(K,mode_boundary)
        phi=calc_absorption(R, Q, dicts[0],K)
        J_phi=calc_absorption_jacobian( R, B_new,Q,dicts,K ,mode_boundary)
        PE=calc_PE(Q,K)

        Q,dP,X=calc_flows_pressures(K)
        phi0=np.ones(m)*K.phi0

        for i in range(m):
            dr[i]=-2.*K.alpha_0*np.sum( np.multiply(np.subtract(phi,phi0),J_phi[i,:]))

        DR=np.multiply(2.*np.subtract(np.power(np.multiply(dP,R),2)*K.k, K.alpha_1*np.ones(len(phi))*np.sqrt(K.k)), R)
        dr=np.add(dr,2.*DR)

        dict_output={}
        dict_output['radii']=R
        dict_output['PE']=PE

        K.set_network_attributes()
        nx.write_gpickle(K.G,op.join(K.DIR_OUT_DATA,'graph_backup'))
        f = open(op.join(K.DIR_OUT_DATA,'dict_dynamic_report.pkl'),"wb")
        pickle.dump(dict_output,f)
        f.close()

    elif 'absorption' ==  mode:

        c,Q,dicts,B_new,K=calc_profile_concentration(K,mode_boundary)
        PE=calc_PE(Q,K)
        phi=calc_absorption(R, Q, dicts[0],K)
        J_phi=calc_absorption_jacobian(R, B_new,Q,dicts,K,mode_boundary )
        phi0=np.ones(m)*K.phi0
        for i in range(len(R)):
            dr[i]=-2.*np.sum( np.multiply(np.subtract(phi,phi0),J_phi[i,:]))

        dict_output={}
        dict_output['radii']=R
        dict_output['PE']=PE

        K.set_network_attributes()
        nx.write_gpickle(K.G,op.join(K.DIR_OUT_DATA,'graph_backup'))
        f = open(op.join(K.DIR_OUT_DATA,'dict_dynamic_report.pkl'),"wb")
        pickle.dump(dict_output,f)
        f.close()

    elif 'absorption+volumetric' ==  mode:
        ones=np.ones(len(K.dict_volumes.values()))
        c,Q,dicts,B_new,K=calc_profile_concentration(K,mode_boundary)

        phi=calc_absorption(R, Q, dicts[0],K)
        J_phi=calc_absorption_jacobian(R, B_new,Q,dicts,K,mode_boundary)
        phi0=ones*K.phi0
        dphi=ones
        PE=calc_PE(Q,K)

        for i,v in enumerate(K.dict_volumes.keys()):
            dphi[v]=np.sum(np.multiply(K.weights[K.dict_volumes[v]],phi[K.dict_volumes[v]]))-phi0[v]

        for j,e in enumerate(K.G.edges()):
            for i,v in enumerate(K.dict_volumes.keys()):
                dr[j]-=2.*dphi[v]*np.sum(np.multiply(K.weights[K.dict_volumes[v]],J_phi[j,K.dict_volumes[v]]))

        dict_output={}
        dict_output['radii']=R
        dict_output['PE']=PE
        K.set_network_attributes()
        nx.write_gpickle(K.G,op.join(K.DIR_OUT_DATA,'graph_backup'))
        f = open(op.join(K.DIR_OUT_DATA,'dict_dynamic_report.pkl'),"wb")
        pickle.dump(dict_output,f)
        f.close()

    elif 'absorption+volumetric+shear' ==  mode:
        ones=np.ones(len(K.dict_volumes.values()))
        c,Q,dicts,B_new,K=calc_profile_concentration(K,mode_boundary)

        phi=calc_absorption(R, Q, dicts[0],K)
        J_phi=calc_absorption_jacobian(R, B_new,Q,dicts,K,mode_boundary)
        phi0=ones*K.phi0
        dphi=ones
        PE=calc_PE(Q,K)

        for i,v in enumerate(K.dict_volumes.keys()):
            dphi[v]=np.sum(np.multiply(K.weights[K.dict_volumes[v]],phi[K.dict_volumes[v]]))-phi0[v]

        for j,e in enumerate(K.G.edges()):
            for i,v in enumerate(K.dict_volumes.keys()):
                dr[j]-=2.*dphi[v]*np.sum(np.multiply(K.weights[K.dict_volumes[v]],J_phi[j,K.dict_volumes[v]]))

        dict_output={}
        dict_output['radii']=R
        dict_output['PE']=PE
        K.set_network_attributes()
        nx.write_gpickle(K.G,op.join(K.DIR_OUT_DATA,'graph_backup'))
        f = open(op.join(K.DIR_OUT_DATA,'dict_dynamic_report.pkl'),"wb")
        pickle.dump(dict_output,f)
        f.close()
    # elif 'absorption+volumetric+volume' ==  mode:
    #     ones=np.ones(len(K.dict_volumes.values()))
    #     c,Q,dicts,B_new,K=calc_profile_concentration(K)
    #
    #     phi=calc_absorption(R, Q, dicts[0],K)
    #     J_phi=calc_absorption_jacobian(R, B_new,Q,dicts,K)
    #     phi0=ones*K.phi0
    #     dphi=ones
    #
    #     for i,v in enumerate(K.dict_volumes.keys()):
    #         dphi[v]=np.sum(np.multiply(K.weights[K.dict_volumes[v]],phi[K.dict_volumes[v]]))-phi0[v]
    #
    #     for j,e in enumerate(K.G.edges()):
    #         for i,v in enumerate(K.dict_volumes.keys()):
    #             dr[j]-=2.*dphi[v]*np.sum(np.multiply(K.weights[K.dict_volumes[v]],J_phi[j,K.dict_volumes[v]]))
    #     DR=K.alpha_0* R
    #     dr=np.subtract(np.multiply(dr,R),DR)
    elif 'shear' ==  mode:

        Q,dP,X=calc_flows_pressures(K)
        K.dV=dP[:]
        DR=np.multiply(np.subtract(2.*K.alpha_1*np.power(np.multiply(dP,R),2),K.alpha_0*np.ones(len(Q))), R)
        dr=np.add(dr,DR)

    elif 'shear+fluctuation' ==  mode:
        # I=init_integration.integrate_stochastic()
        # I.M=nx.number_of_edges(K.G)
        # I.N=nx.number_of_nodes(K.G)
        # I.noise=K.noise
        # I.fraction=K.fraction
        # I.setup_random_fluctuations_terminals(K)
        dV_sq, F_sq=calc_sq_flow_broken_link(K)

        K.dV_sq=dV_sq[:]
        DR=np.multiply(np.subtract(2.*K.alpha_1*np.multiply(dV_sq,np.power(R,2)),K.alpha_0*np.ones(len(R))), R)
        dr=np.add(dr,2.*DR)

    else:
        os.abort('no legitimate mode')

    return dr

def propagate_system_dynamic_report(K,t_span,t_eval,mode):

    nsol=si.solve_ivp(update_stimuli_report,t_span,K.R,args=( K,mode ),t_eval=t_eval,method='LSODA',events=flatlining)
    # nsol=si.solve_ivp(update_stimuli_report,t_span,K.R,args=( K,mode ),t_eval=t_eval,method='RK45')

    sol=np.transpose(np.array(nsol.y))
    K.R=sol[-1,:]
    K.C=np.power(K.R,4)*K.k
    K.set_network_attributes()

    return sol,K

def propagate_system_manually(K,N,mode):

    sol=[]
    for i in range(N):
        try:
            DR=update_stimuli_report(i,K.R,K,mode)
            K.R=np.add(K.R,DR*K.dt)
            sol.append(K.R)
        except:
            break
    K.C=np.power(K.R,4)*K.k
    K.set_network_attributes()

    return np.array(sol),K
