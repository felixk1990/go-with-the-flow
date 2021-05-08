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

crit_pe=50.

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

# stationary concentration profiles
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
    K.Q=Q
    return Q, dP, P

def calc_flows_pressures_mapped(graph_matrices):

    C_aux,B,BT,J=graph_matrices
    OP=np.dot(B,np.dot(np.diag(C_aux),BT))
    P,RES,RG,si=np.linalg.lstsq(OP,J,rcond=None)
    dP=np.dot(BT,P)
    Q=np.multiply(C_aux,dP)

    return [Q,P,dP]

def break_link(K,idx):
    C_aux=np.array(K.C[:])
    C_aux[idx]=np.power(10.,-20)
    return C_aux

def calc_PE(K):

    R_sq=np.power(K.R,2)
    V=np.divide(K.Q,R_sq*np.pi)
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
    X=150
    idx=rd.choices(K.broken_sets,k=X)
    C_broken_ensemble=[break_link(K,i) for i in idx]
    graph_matrices=[[C_broken_ensemble[i],K.B,K.BT,K.J] for i in range(X)]

    # calc the flow landscapes for each realization
    flow_observables=list(map(calc_flows_pressures_mapped,graph_matrices))

    # calc ensemble averages
    F_sq=np.power([fo[0] for fo in flow_observables],2)
    dV_sq=np.power([fo[2] for fo in flow_observables],2)
    R_sq=[np.sqrt(C_broken_ensemble[i]/K.k)  for i in range(X)]

    avg_shear_sq=np.sum(np.multiply(dV_sq,R_sq),axis=0)/float(X)
    avg_dV_sq=np.sum(dV_sq,axis=0)/float(X)
    avg_F_sq= np.sum(F_sq,axis=0)/float(X)

    return avg_shear_sq,avg_dV_sq,avg_F_sq

def calc_concentration_coefficients(PE,K):

    ones=np.ones(len(PE))
    x=np.sqrt(np.add(ones,np.divide(K.beta,np.power(PE,2))))

    a=np.multiply(0.5*PE, np.add( ones, x ) )
    b=np.multiply(0.5*PE, np.subtract( ones, x ) )

    return a,b

def update_stationary_operator(K):

    Q,dP,P=calc_flows_pressures(K)
    PE=calc_PE(K)

    ones=np.ones(len(PE))
    N=nx.number_of_nodes(K.G)

    A=np.pi*np.power(K.R,2)*(K.D/K.l)
    x,z,e_up_sinh_x,e_down_sinh_x,coth_x,idx_pack=compute_flux_pars(K,PE)

    f1= np.multiply(z,A)
    f2= np.multiply(A,np.multiply(x,coth_x))*0.5

    f3= np.multiply(np.multiply(A,x),e_up_sinh_x)*0.5
    f4= np.multiply(np.multiply(A,x),e_down_sinh_x)*0.5

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
        B_eff[i,dict_in[n]]= -f3[dict_node_in[n]]
        B_eff[i,dict_out[n]]= -f4[dict_node_out[n]]

    return B_eff,[dict_edges,dict_in,dict_out]

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

    B_eff,dicts=update_stationary_operator(K)

    # use absorbing boundaries + reduced equation system
    if mode=='absorbing_boundary':
        c,B_new,K=solve_absorbing_boundary(B_eff,K)

    # use inlet delta peak + reduced equation system
    if mode=='mixed_boundary':
        c,B_new,K=solve_inlet_peak(B_eff,K)

    return c,dicts,B_new,K

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
    dict_edges,K= args

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

    PE=calc_PE(K)
    x,z,e_up_sinh_x,e_down_sinh_x,coth_x,idx_pack=compute_flux_pars(K,PE)

    f1_up= x*0.5*e_up_sinh_x
    f1_down= x*0.5*e_down_sinh_x
    F1=np.add( np.subtract( np.multiply(x,coth_x)*0.5 , f1_up), z)
    F2=np.add( np.subtract( np.multiply(x,coth_x)*0.5 , f1_down), -z)
    # calc edgewise absorption
    phi=np.add(np.multiply(c_a,F1) ,np.multiply( c_b,F2 ))
    A=np.pi*np.multiply(R,R)*(K.D/K.l)

    return np.multiply( A, phi )

def calc_flux_jacobian(R,*args):

    # unzip parameters
    PE,L,K= args

    # init containers
    M=len(PE)
    I=np.identity(M)
    J_PE, J_Q= np.zeros((M,M)),np.zeros((M,M))

    # set coefficients
    f1= 2.*np.divide(PE,R)
    f2= 4.* np.divide(K.Q,R)
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
    x,z,e_up_sinh_x,e_down_sinh_x,coth_x,idx_pack=compute_flux_pars(K,PE)
    f1= np.multiply(z,A)
    f2= np.multiply(np.multiply(x,coth_x),A)*0.5

    f3= np.multiply(np.multiply(A,x),e_up_sinh_x)*0.5
    f4= np.multiply(np.multiply(A,x),e_down_sinh_x)*0.5

    B_eff=np.zeros((N,N))
    idx_source=0
    idx_sink=0

    for i,n in enumerate(K.G.nodes()):
        dict_node_out[n]=np.where(K.B[i,:]>0)[0]
        dict_node_in[n]=np.where(K.B[i,:]<0)[0]
        b=K.B[i,:]
        B_eff[i,i]= np.sum(  np.add( np.multiply(b,f1), np.multiply(np.absolute(b),f2))  )
        B_eff[i,dict_in[n]]= - f3[dict_node_in[n]]
        B_eff[i,dict_out[n]]= - f4[dict_node_out[n]]
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

    j_coth_x=np.zeros(len(PE))
    # for i,pe in enumerate(PE):
    #     if np.absolute(pe) < crit_pe:
    #         j_coth_x[i]=np.power(np.divide(coth_x[i],np.cosh(x[i])),2)
    #     else:
    #         j_coth_x[i]=0.

    idx_lower=idx_pack[0]
    # idx_over=idx_pack[1]
    # subcritical
    j_coth_x[idx_lower]=np.power(np.divide(coth_x[idx_lower],np.cosh(x[idx_lower])),2)
    # overcritical
    # j_coth_x[idx_over]=0.

    f2= np.multiply(x,coth_x)*0.5
    f4=np.subtract( np.multiply( np.divide(z,x), coth_x) ,  np.multiply( z,j_coth_x )*0.5 )

    f_up=np.multiply( e_up_sinh_x ,x )*0.5
    f_down=np.multiply( e_down_sinh_x ,x )*0.5
    f5= np.multiply(A, e_up_sinh_x )
    f6= np.multiply(A, e_down_sinh_x)

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
    B_eff,dicts,K,mode= args

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

    PE=calc_PE(K)
    x,z,e_up_sinh_x,e_down_sinh_x,coth_x,idx_pack=compute_flux_pars(K,PE)

    f1_up= 0.5*np.multiply(x,e_up_sinh_x)
    f1_down= 0.5*np.multiply(x,e_down_sinh_x)

    F1=np.add( np.subtract( 0.5*np.multiply(x,coth_x) ,  f1_up ), z )
    F2=np.subtract( np.subtract( 0.5*np.multiply(x,coth_x) , f1_down ), z)
    F3,F4=calc_absorption_jacobian_coefficients(x,z,e_up_sinh_x,e_down_sinh_x,coth_x,idx_pack)

    phi=np.add( np.multiply(c_a,F1) ,np.multiply(c_b,F2 ) )

    # calc jacobian
    J_PE,J_Q= calc_flux_jacobian(R,PE,L,K)
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

def calc_absorption_jacobian_coefficients(x,z,e_up_sinh_x,e_down_sinh_x,coth_x,idx_pack):

    F3=np.zeros(len(x))
    F4=np.zeros(len(x))
    idx_lower=idx_pack[0]
    idx_over=idx_pack[1]

    # subcritical
    sinh_x=np.sinh(x[idx_lower]*0.5)
    cosh_x=np.cosh(x[idx_lower]*0.5)
    e_up=np.exp(z[idx_lower])
    e_down=np.exp(-z[idx_lower])

    f2_up=np.subtract( np.multiply( np.divide(2.*z[idx_lower],x[idx_lower]), np.subtract( cosh_x, e_up )), np.divide(z[idx_lower],sinh_x))
    f3_up=np.add( np.multiply( e_up, np.subtract( np.multiply(coth_x[idx_lower],z[idx_lower]), 0.5*x[idx_lower] ) ) , sinh_x)
    f2_down=np.subtract(np.multiply( np.divide(2.*z[idx_lower],x[idx_lower]), np.subtract( cosh_x, e_down )), np.divide(z[idx_lower],sinh_x))
    f3_down=np.subtract( np.multiply( e_down, np.add( np.multiply(coth_x[idx_lower],z[idx_lower]), 0.5*x[idx_lower]  ) ), sinh_x )

    F3[idx_lower]= 0.5*np.divide( np.add(f2_up, f3_up) , sinh_x)
    F4[idx_lower]= 0.5*np.divide( np.add(f2_down, f3_down) , sinh_x)


    # overcritical
    f2_up= np.multiply( np.divide(2.*z[idx_over],x[idx_over]), np.subtract( coth_x[idx_over], 2. ))
    f3_up=np.add( 2.* np.subtract( np.multiply(coth_x[idx_over],z[idx_over]), 0.5*x[idx_over] )  , 1.)

    f2_down=np.multiply( np.divide(2.*z[idx_over],x[idx_over]), coth_x[idx_over] )
    # f3_down=np.zeros(len(idx_over))

    F3[idx_over]= 0.5* np.add(f2_up, f3_up)
    F4[idx_over]= 0.5* f2_down

    return F3,F4

def compute_flux_pars(K,Peclet):

    x=np.sqrt( np.add( np.power(Peclet,2),K.beta ) )
    z=Peclet*0.5

    e_up_sinh_x=np.zeros(len(Peclet))
    e_down_sinh_x=np.zeros(len(Peclet))
    coth_x=np.zeros(len(Peclet))
    # establish the use of converging limit expressions to prevent overflow error
    idx_lower=np.where(np.absolute(Peclet)<crit_pe)[0]
    idx_over_pos=np.where((np.absolute(Peclet)>=crit_pe) & (Peclet > 0))[0]
    idx_over_neg=np.where((np.absolute(Peclet)>=crit_pe) & (Peclet < 0))[0]
    idx_pack=[list(idx_lower),list(idx_over_pos)+list(idx_over_neg)]
    # for i,PE in enumerate(Peclet):
    #
    #     if np.absolute(PE) < crit_pe:
    #
    #         e_up=np.exp(z[i])
    #         e_down=np.exp(-z[i])
    #         e_up_sinh_x[i]=e_up/np.sinh(x[i]*0.5)
    #         e_down_sinh_x[i]=e_down/np.sinh(x[i]*0.5)
    #         coth_x[i]=np.reciprocal(np.tanh(x[i]*0.5))
    #
    #     else:
    #         if PE > 0:
    #             e_up_sinh_x[i]=2.
    #             e_down_sinh_x[i]=0.
    #         if PE < 0:
    #             e_up_sinh_x[i]=0.
    #             e_down_sinh_x[i]=2.
    #         coth_x[i]=1.

    # subcritical pe
    e_up=np.exp(z[idx_lower])
    e_down=np.exp(-z[idx_lower])
    e_up_sinh_x[idx_lower]=e_up/np.sinh(x[idx_lower]*0.5)
    e_down_sinh_x[idx_lower]=e_down/np.sinh(x[idx_lower]*0.5)
    coth_x[idx_lower]=np.reciprocal(np.tanh(x[idx_lower]*0.5))

    # overcriticial, pe positive
    e_up_sinh_x[idx_over_pos]=2.
    e_down_sinh_x[idx_over_pos]=0.
    coth_x[idx_over_pos]=1.

    # overcriticial, pe negative
    e_up_sinh_x[idx_over_neg]=0.
    e_down_sinh_x[idx_over_neg]=2.
    coth_x[idx_over_neg]=1.

    return x,z,e_up_sinh_x,e_down_sinh_x,coth_x,idx_pack
# calc the incerements for various update_stimuli
def calc_absorption_shear_dr(K):
    dr=np.zeros(len(K.R))
    c,Q,dicts,B_new,K=calc_profile_concentration(K,mode_boundary)
    phi=calc_absorption(K.R, Q, dicts[0],K)

    J_phi=calc_absorption_jacobian( K.R, B_new,Q,dicts,K ,mode_boundary)
    PE=calc_PE(K)

    Q,dP,X=calc_flows_pressures(K)
    phi0=np.ones(m)*K.phi0

    for i in range(m):
        dr[i]=-2.*np.sum( np.multiply(np.subtract(phi,phi0),J_phi[i,:]))

    DR=np.multiply(2.*np.subtract(K.alpha_1*np.power(np.multiply(dP,K.R),2),K.alpha_0*np.ones(len(phi))), K.R)
    dr=np.add(dr,2.*DR)

    return dr,PE

def calc_shear_absorption_dr(K):

    dr=np.zeros(len(K.R))
    c,Q,dicts,B_new,K=calc_profile_concentration(K,mode_boundary)
    phi=calc_absorption(K.R, Q, dicts[0],K)
    J_phi=calc_absorption_jacobian( K.R, B_new,Q,dicts,K ,mode_boundary)
    PE=calc_PE(K)

    Q,dP,X=calc_flows_pressures(K)
    phi0=np.ones(m)*K.phi0

    for i in range(m):
        dr[i]=-2.*K.alpha_0*np.sum( np.multiply(np.subtract(phi,phi0),J_phi[i,:]))

    DR=np.multiply(2.*np.subtract(np.power(np.multiply(dP,K.R),2)*K.k, K.alpha_1*np.ones(len(phi))*np.sqrt(K.k)), K.R)
    dr=np.add(dr,2.*DR)

    return dr, PE

def calc_shear_fluctuation_dr(K):

    dr=np.zeros(len(K.R))
    shear_sq,dV_sq, F_sq=calc_sq_flow_broken_link(K)
    K.dV_sq=dV_sq[:]
    # DR=np.multiply(np.subtract(2.*K.alpha_1*np.multiply(dV_sq,np.power(R,2)),K.alpha_0*np.ones(len(R))), R)
    DR=np.multiply(np.subtract(2.*K.alpha_1*shear_sq,K.alpha_0*np.ones(len(K.R))), K.R)
    dr=np.add(dr,2.*DR)

    return dr,shear_sq,dV_sq,F_sq

def calc_absorption_dr(K,mode_boundary):

    m=len(K.R)
    dr=np.zeros(m)
    c,dicts,B_new,K=calc_profile_concentration(K,mode_boundary)
    PE=calc_PE(K)
    phi=calc_absorption(K.R, dicts[0],K)
    J_phi=calc_absorption_jacobian(K.R, B_new,dicts,K,mode_boundary )
    phi0=np.ones(m)*K.phi0
    for i in range(len(K.R)):
        dr[i]=-2.*np.sum( np.multiply(np.subtract(phi,phi0),J_phi[i,:]))

    return dr,PE

def calc_absorption_volumetric_dr(K):

    dr=np.zeros(len(K.R))
    ones=np.ones(len(K.dict_volumes.values()))
    c,Q,dicts,B_new,K=calc_profile_concentration(K,mode_boundary)

    phi=calc_absorption(R, Q, dicts[0],K)
    J_phi=calc_absorption_jacobian(K.R, B_new,Q,dicts,K,mode_boundary)
    phi0=ones*K.phi0
    dphi=ones
    PE=calc_PE(K)

    for i,v in enumerate(K.dict_volumes.keys()):
        dphi[v]=np.sum(np.multiply(K.weights[K.dict_volumes[v]],phi[K.dict_volumes[v]]))-phi0[v]

    for j,e in enumerate(K.G.edges()):
        for i,v in enumerate(K.dict_volumes.keys()):
            dr[j]-=2.*dphi[v]*np.sum(np.multiply(K.weights[K.dict_volumes[v]],J_phi[j,K.dict_volumes[v]]))

    return dr,PE

def calc_absorption_volumetric_shear_dr(K):

    dr=np.zeros(len(K.R))
    ones=np.ones(len(K.dict_volumes.values()))
    c,Q,dicts,B_new,K=calc_profile_concentration(K,mode_boundary)

    phi=calc_absorption(K.R, Q, dicts[0],K)
    J_phi=calc_absorption_jacobian(R, B_new,Q,dicts,K,mode_boundary)
    phi0=ones*K.phi0
    dphi=ones
    PE=calc_PE(K)

    for i,v in enumerate(K.dict_volumes.keys()):
        dphi[v]=np.sum(np.multiply(K.weights[K.dict_volumes[v]],phi[K.dict_volumes[v]]))-phi0[v]

    for j,e in enumerate(K.G.edges()):
        for i,v in enumerate(K.dict_volumes.keys()):
            dr[j]-=2.*dphi[v]*np.sum(np.multiply(K.weights[K.dict_volumes[v]],J_phi[j,K.dict_volumes[v]]))
    return dr,PE

def calc_shear_dr(K):

    dr=np.zeros(len(K.R))
    Q,dP,X=calc_flows_pressures(K)
    K.dV=dP[:]
    DR=np.multiply(np.subtract(2.*K.alpha_1*np.power(np.multiply(dP,K.R),2),K.alpha_0*np.ones(len(Q))), K.R)
    dr=np.add(dr,DR)
    return dr

def save_dynamic_output(*args):

    items=args[0]
    keys=args[1]
    K=args[2]

    dict_output={}
    for i,key in enumerate(keys):
        dict_output[key]=items[i]

    K.set_network_attributes()
    nx.write_gpickle(K.G,op.join(K.DIR_OUT_DATA,'graph_backup'))
    f = open(op.join(K.DIR_OUT_DATA,'dict_dynamic_report.pkl'),"wb")
    pickle.dump(dict_output,f)
    f.close()

# optimize networks with gradient descent
def flatlining(t,R,K,mode):
    B,BT=K.get_incidence_matrices()
    mode_boundary='absorbing_boundary'
    K.R=R
    sq_R=np.power(R,2)
    K.C=np.power(sq_R,2)*K.k
    c,dicts,B_new,K=calc_profile_concentration(K,mode_boundary)
    phi=calc_absorption(K.R, dicts[0],K)
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
    # print(z)
    return z

flatlining.terminal=True
flatlining.direction = -1

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
        c,dicts,B_new,K=calc_profile_concentration(K,mode_boundary)
        phi=calc_absorption( K.R,  dicts[0],K )
        PHI.append(phi)
        C.append(c)
        PE.append(calc_PE(K))

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

    if 'absorption+shear' ==  mode:

        dr,PE=calc_absorption_shear_dr(K)
        save_dynamic_output([R,PE],['radii','PE'],K)

    elif 'shear+absorption' ==  mode:

        dr,PE=calc_shear_absorption_dr(K)
        save_dynamic_output([R,PE],['radii','PE'],K)

    elif 'absorption' ==  mode:

        dr,PE=calc_absorption_dr(K,mode_boundary)
        save_dynamic_output([R,PE],['radii','PE'],K)

    elif 'absorption+volumetric' ==  mode:

        dr,PE=calc_absorption_volumetric_dr(K)
        save_dynamic_output([R,PE],['radii','PE'],K)

    elif 'absorption+volumetric+shear' ==  mode:

        dr,PE=calc_absorption_volumetric_shear_dr(K)
        save_dynamic_output([R,PE],['radii','PE'],K)

    elif 'shear' ==  mode:

        dr,PE=calc_shear_dr(K)
        save_dynamic_output([R],['radii'],K)

    elif 'shear+fluctuation' ==  mode:

        dr,shear_sq,dV_sq,F_sq=calc_shear_fluctuation_dr(K)
        save_dynamic_output([shear_sq,dV_sq,F_sq],['shear_sq','dV_sq','F_sq'],K)

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
