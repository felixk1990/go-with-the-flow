# @Author: Felix Kramer <kramer>
# @Date:   26-04-2019
# @Email:  kramer@mpi-cbg.de
# @Project: phd_network_remodelling
# @Last modified by:   kramer
# @Last modified time: 25-09-2019

import numpy as np
import scipy.linalg as lina
import sys
import scipy as sy
import scipy.integrate as si
import scipy.optimize as sc
# import init_flow
import random as rd
import networkx as nx
import functions_template as ft
import complex_flux as cf


# these objects are composed in order to enable a direct scipy independent solving of relevant ode systems
class integrate_kirchoff:

    def __init__(self):
        self.gamma=1.
        self.c0=1.
        self.c1=1.
        self.sigma=1.
        self.kappa=1.

        self.alpha=0
        self.sigma=0
        self.scales=0
        self.vol_diss=0
        self.coupling_diss=0
        self.coupling_exp=0
        self.noise=0
        self.local_flow=0.
        self.x=0
        self.mu=0
        self.N=0
        self.M=0

    def set_integration_scale(self,Num_steps,sample):
        #reshape number of integration steps & sample rates for consistency
        sample_rate=int(Num_steps/sample)
        if (sample_rate*sample) < Num_steps:
            Num_steps=sample_rate*sample

        return Num_steps,sample_rate

    def print_step(self,i,Num_steps):
        #print status of integration procedure

        if i == (Num_steps-1):
            print('Status ... 100%\n THE END')
        elif i % (Num_steps/100) == 0:
            x=100*i/Num_steps
            print('Status ... '+str(x)+'%')

    def setup_random_fluctuations(self,N, mean, variance):

        self.mu=mean
        self.var=variance

        self.G=np.identity(N)
        for n in range(N):
            for m in range(N):
                h=0.
                if n==0 and m==0:
                    h+=(N-1)
                elif n==m and n!=0:
                    h+=1.
                elif n==0 and m!=0:
                    h-=1
                elif m==0 and n!=0:
                    h-1

                self.G[n,m]=h
        self.H=np.identity(N)
        for n in range(N):

            for m in range(N):
                h=0.
                if n==0 and m==0:
                    h+=(1-N)*(1-N)
                elif n!=0 and m!=0:
                    h+=1.
                elif n==0 and m!=0:
                    h+=(1-N)
                elif m==0 and n!=0:
                    h+=(1-N)

                self.H[n,m]=h

    def setup_random_fluctuations_reduced(self,K, mean, std):

        x=np.where(K.J > 0)[0][0]

        N=len(K.J)
        idx=np.where(x!=range(N))[0]
        L0=np.ones((N,N))
        L0[idx,:]=0.
        L0[:,idx]=0.

        L1=np.identity(N)
        L1[x,x]=0.

        L2=np.zeros((N,N))
        L2[idx,:]=1.-N
        L2[:,idx]=1.-N
        L2[x,x]=(N-1)**2

        alpha=mean
        beta=std
        f_beta=1+beta/(N-1)

        self.Z = (L0 + beta * L1 + f_beta * L2)*alpha

    def setup_random_fluctuations_multisink(self,K ):

        num_n=nx.number_of_nodes(K.G)
        x=np.where(K.J > 0)[0]
        idx=np.where(K.J < 0)[0]
        N=len(idx)
        M=len(x)

        U=np.zeros((num_n,num_n))
        V=np.zeros((num_n,num_n))

        m_sq=float(M*M)
        NM=num_n*num_n/float(m_sq)
        Nm=(N/m_sq)+2./M

        for i in range(num_n):
            for j in range(num_n)[i:]:
                delta=0.
                sum_delta=0.
                sum_delta_sq=0.

                if i==j:
                    delta=1.

                if (i in x):
                    sum_delta=1.

                if (j in x):
                    sum_delta=1.

                if (i in x) and (j in x):
                    sum_delta_sq=1.
                    sum_delta=2.

                U[i,j]= ( m_sq - num_n*sum_delta + NM*sum_delta_sq )
                V[i,j]= ( ( Nm + delta )*sum_delta_sq - (1.+M*delta)*sum_delta + m_sq*delta)

                U[j,i]=U[i,j]
                V[j,i]=V[i,j]

        self.Z = np.add(U,np.multiply(self.noise,V))

    def setup_random_fluctuations_effective(self,K):

        self.x=np.where(K.J > 0)[0][0]
        L0=np.ones((self.N,self.N))
        L0[self.x,:]=0.
        L0[:,self.x]=0.

        L1=np.identity(self.N)
        L1[self.x,self.x]=0.

        L2=np.zeros((self.N,self.N))
        L2[self.x,:]=1.-self.N
        L2[:,self.x]=1.-self.N
        L2[self.x,self.x]=(self.N-1)**2

        f_noise=1+self.noise/(self.N-1)

        self.Z = np.add(np.add(L0 , self.noise * L1) ,f_noise * L2)

    def setup_random_fluctuations_terminals(self,K):

        x=np.where(K.J > 0)[0][0]
        y=np.where(K.J < 0)[0][0]

        L0=np.ones((self.N,self.N))

        L0[:,y]=(self.fraction+1.)*(self.N-2)
        L0[y,:]=(self.fraction+1.)*(self.N-2)
        L0[x,:]=-(self.fraction+1.)*(self.N-2)
        L0[:,x]=-(self.fraction+1.)*(self.N-2)
        L0[x,y]-=(2.+self.fraction*(self.fraction+1.)*((self.N-2)**2))
        L0[y,x]-=(2.+self.fraction*(self.fraction+1.)*((self.N-2)**2))
        L0[x,x]=((self.fraction+1.)*(self.N-2))**2
        L0[y,y]=self.fraction**2

        L1=np.identity(self.N)
        L1[x,:]=-1.
        L1[:,x]=-1.
        L1[x,x]=self.N
        L1[y,y]=0.

        self.Z = np.add(L0 ,self.noise * L1 )

    def calc_sq_pressure(self,C,B,BT,S):

        OP=np.dot(np.dot(B,np.diag(C)),BT)
        MP=lina.pinv(OP)
        V=MP.dot(S)
        dV=np.dot(BT,V)
        v_sq=np.multiply(dV,dV)

        return v_sq

    def calc_sq_pressure_general(self,C,B,BT,idx,parameters):

        OP=np.dot(np.dot(B,np.diag(C)),BT)
        MP=lina.pinv(OP)
        A=np.dot(BT,MP)
        N=len(idx[0])

        v_sq=np.zeros(len(C))
        for e,idx_e in enumerate(C):
            A_e0=A[e,idx[1]]
            A_00=A_e0*A_e0
            A_01=np.multiply(A_e0,A[e,idx[0]])
            A_12=np.outer(A[e,idx[0]],A[e,idx[0]])
            G=np.add(A_12,A_00)
            G=np.subtract(G,np.outer(A_01,np.ones(N)))
            G=np.subtract(G,np.outer(np.ones(N),A_01))
            # for i,index1 in enumerate(idx[0]):
            #     for j,index2 in enumerate(idx[0]):
                    # A_01=A[e,idx[1]]*A[e,index1]
                    # A_02=A[e,idx[1]]*A[e,index2]
                    # A_12=A[e,index1]*A[e,index2]

                    # G[i,j]=A_00-A_01[i]-A_02[j]+A_12[i,j]
            # if e==0:
                # print(np.sum(G[np.triu_indices(N,1)]))
                # print(np.sum(np.diagonal(G)))


            v_sq[e]=np.sum(G)+2.*parameters[0]* np.sum(G[np.triu_indices(N,1)])+parameters[1]*np.sum(np.diagonal(G))
        # print(v_sq)
        return v_sq

    def calc_sq_pressure_temporal_mean(self,mode,index,C,t,B,BT):

        idx=index[0]
        x=index[1]
        N=len(idx)

        S=np.zeros((N+1,N+1))

        if 'nodc_square' in mode[0]:
            phases=np.add(mode[1]*t,mode[2])
            oscillator_matrix=np.zeros((N+1,N+1))
            for i in range(N+1):
                if i !=x:
                    oscillator_matrix[i,i]=(mode[3][i]**2)*3./8.
            for i in range(N):
                for j in range(i+1,N+1):
                    if i !=x and j !=x:
                        phase_diff=phases[i]-phases[j]
                        oscillator_matrix[i,j]=mode[3][i]*mode[3][j]*(0.5+np.cos(phase_diff)**2)/4.
                        oscillator_matrix[j,i]=oscillator_matrix[i,j]

            beta=np.sum(oscillator_matrix)
            gamma=np.sum(oscillator_matrix,axis=0)

            for i in range(N+1):
                if i==x:
                    S[i,i]+=beta
                else:
                    S[i,i]+=oscillator_matrix[i,i]
            for i in range(N):
                for j in range(i+1,N+1):
                    if i==x and j!=j:
                        S[i,j]-=gamma[j]
                        S[j,i]-=gamma[j]
                    else:
                        S[i,j]+=oscillator_matrix[i,j]
                        S[j,i]+=oscillator_matrix[i,j]

        if 'nodc_square_random_avg' in mode[0]:
            phases=np.random.uniform(size=N+1)
            oscillator_matrix=np.zeros((N+1,N+1))
            for i in range(N+1):
                if i !=x:
                    oscillator_matrix[i,i]=(mode[3][i]**2)*3./8.
            for i in range(N):
                for j in range(i+1,N+1):
                    if i !=x and j !=x:
                        phase_diff=(phases[i]-phases[j])*2.*np.pi
                        oscillator_matrix[i,j]=mode[3][i]*mode[3][j]*(0.5+np.cos(phase_diff)**2)/4.
                        oscillator_matrix[j,i]=oscillator_matrix[i,j]

            beta=np.sum(oscillator_matrix)
            gamma=np.sum(oscillator_matrix,axis=0)

            for i in range(N+1):
                if i==x:
                    S[i,i]+=beta
                else:
                    S[i,i]+=oscillator_matrix[i,i]
            for i in range(N):
                for j in range(i+1,N+1):
                    if i==x and j!=j:
                        S[i,j]-=gamma[j]
                        S[j,i]-=gamma[j]
                    else:
                        S[i,j]+=oscillator_matrix[i,j]
                        S[j,i]+=oscillator_matrix[i,j]

        if 'nodc_square_random' in mode[0]:
            phases=np.random.uniform(size=N+1)
            oscillator_matrix=np.zeros((N+1,N+1))
            for i in range(N+1):
                if i !=x:
                    oscillator_matrix[i,i]=(mode[3][i]**2)*np.sin(phases[i]*2.*np.pi)**4
            for i in range(N):
                for j in range(i+1,N+1):
                    if i !=x and j !=x:

                        oscillator_matrix[i,j]=mode[3][i]*mode[3][j]*(np.sin(phases[i]*2.*np.pi)**2)*np.sin(phases[j]*2.*np.pi)**2
                        oscillator_matrix[j,i]=oscillator_matrix[i,j]

            beta=np.sum(oscillator_matrix)
            gamma=np.sum(oscillator_matrix,axis=0)

            for i in range(N+1):
                if i==x:
                    S[i,i]+=beta
                else:
                    S[i,i]+=oscillator_matrix[i,i]
            for i in range(N):
                for j in range(i+1,N+1):
                    if i==x and j!=j:
                        S[i,j]-=gamma[j]
                        S[j,i]-=gamma[j]
                    else:
                        S[i,j]+=oscillator_matrix[i,j]
                        S[j,i]+=oscillator_matrix[i,j]

        OP=np.dot(np.dot(B,np.diag(C)),BT)
        MP=lina.pinv(OP)
        D=np.dot(MP.dot(S),MP)
        V_SQ=np.dot(BT,np.dot(D,B))

        return np.diagonal(V_SQ)

    def calc_sq_flow(self,C,B,BT):
        # matrix decomposition numpy.linalg
        # C_diag=np.diag(C)
        # A=np.dot(B,np.dot(C_diag,BT))
        # sub_A=A[1:,:]
        # sub_A_T=np.transpose(sub_A)
        # sym_sub_A=np.dot(sub_A,sub_A_T)
        # inv_sub_A=np.linalg.inv(sym_sub_A)
        # D=np.dot(np.dot(A,sub_A_T),inv_sub_A)
        # D_T=np.transpose(D)
        # inv_sym_D=np.linalg.inv(np.dot(D_T,D))
        # inv_BCBT=np.dot(sub_A_T,np.dot(inv_sub_A,np.dot(inv_sym_D,D_T)))
        # E=np.dot(BT,inv_BCBT)
        # ET=np.transpose(E)
        # dV_sq=np.diag(np.dot(np.dot(E,self.Z),ET))
        # F_sq=np.multiply(np.multiply(C,C),dV_sq)
        #version numpy/ scipy.linalg
        OP=np.dot(B,np.dot(np.diag(C),BT))
        inverse=lina.pinv(OP)
        D=np.dot(BT,inverse)
        DT=np.transpose(D)
        A=np.dot(D,self.Z)
        V=np.dot(A,DT)
        dV_sq=np.diag(V)
        F_sq=np.multiply(np.multiply(C,C),dV_sq)

        #version-sparse
        # K=ssp.csc_matrix(np.diag(C))
        # OP=B.dot(K.dot(BT))
        # inverse=ssp.linalg.inv(OP)
        # D=BT.dot(inverse.todense())
        # DT=np.transpose(D)
        #
        # A=D.dot(self.Z)
        # V=A.dot(DT)
        # dV_sq=np.diag(V)
        # F_sq=np.multiply(np.multiply(C,C),dV_sq)

        return dV_sq,F_sq

    def calc_sq_flow_random(self,C,B,BT):

        OP=np.dot(np.dot(B,C),BT)
        MP=lina.pinv(OP)
        D=np.dot(C,np.dot(BT,MP))
        DT=np.transpose(D)
        # print(D)
        var_matrix=np.dot(np.dot(D,self.G),DT)
        mean_matrix=np.dot(np.dot(D,self.H),DT)
        # print(D)
        # print(self.H)
        var_flow=np.diag(var_matrix)
        mean_flow=np.diag(mean_matrix)

        # print(var_flow)
        # print(mean_flow)
        F_sq= np.add(self.var*var_flow , self.mu*self.mu*mean_flow)
        # print(F_sq)
        return F_sq

    def calc_sq_flow_random_reduced(self,C,B,BT):

        OP=np.dot(np.dot(B,C),BT)
        MP=lina.pinv(OP)
        D=np.dot(C,np.dot(BT,MP))
        DT=np.transpose(D)

        F_sq=np.diag(np.dot(D,np.dot(self.Z,DT)))

        return F_sq

    def calc_sq_flow_random_manual(self,C,B,BT,parameters):

        OP=np.dot(np.dot(B,np.diag(C)),BT)
        MP=lina.pinv(OP)
        D=np.dot(np.diag(C),np.dot(BT,MP))
        DT=np.transpose(D)
        S=self.mean_correlation(parameters)
        # print(S)
        Q_sq=np.dot(np.dot(D,S),DT)
        F_sq=np.diag(Q_sq)
        V_sq=np.divide(F_sq,np.multiply(C,C))
        # print(F_sq)
        return F_sq,V_sq

    def mean_correlation(self,parameters):

        mode=parameters[0]
        n=parameters[1]
        num_realizations=parameters[2]
        source=parameters[3]
        list_n=parameters[4]
        S_correlation=np.zeros((n,n))


        if 'moving_mono' in mode:
            for state in range(num_realizations):
                S=np.zeros(n)
                sink=rd.choice(list_n)
                S[sink]=-1
                S[source]=1

                S_correlation=np.add(S_correlation,np.outer(S,S))

        if 'moving_poly' in mode:
            p=parameters[5]
            for state in range(num_realizations):
                S=np.zeros(n)
                for node in list_n:
                    if p >= rd.random():
                        s=np.sqrt(1./p)
                        S[node]=-s
                        S[source]+=s
                S_correlation=np.add(S_correlation,np.outer(S,S))
        if 'moving_correlated' in mode:
            sigma=parameters[4]
            adj_list=parameters[5]
            for state in range(num_realizations):

                S=np.zeros(n)
                sink=rd.choice(list_n)
                for j,d in enumerate(adj_list[sink]):
                    if j!=source:
                        s=np.exp(-d*d/sigma)
                        S[j]=-s
                        S[source]+=s

                S_correlation=np.add(S_correlation,np.outer(S,S))
        if 'lognormal_mono' in mode:

            for state in range(num_realizations):
                S=np.zeros(n)
                for m in list_n:
                    S[m]=-np.random.lognormal(mean=0.,sigma=1.)
                S[source]=-1.*np.sum(S[list_n])
                S_correlation=np.add(S_correlation,np.outer(S,S))

        return S_correlation/float(num_realizations)

    def propagate_sources(self,mode,K,t):

        S=np.zeros(len(K.J))
        idx=np.where(K.J < 0)[0]
        x=np.where(K.J > 0)[0][0]

        if 'linear' in mode[0]:
            S[idx]=np.add(np.ones(len(idx)),np.sin(2.*np.pi*mode[1][idx]*t+mode[2][idx])*mode[3])
            S[x]=-np.sum(S[idx])

        elif 'square' in mode[0]:
            lin=np.sin(2.*np.pi*mode[1][idx]*t+mode[2][idx])
            S[idx]=np.add(np.ones(len(idx)),np.multiply(lin,lin)*mode[3])
            S[x]=-np.sum(S[idx])

        elif 'nodc_square' in mode[0]:
            lin=np.sin(2.*np.pi*mode[1][idx]*t+mode[2][idx])
            S[idx]=np.multiply(lin,lin)*mode[3]
            S[x]=-np.sum(S[idx])

        else:
            sys.exit('invalid source mode')

        return S

class integrate_deterministic(integrate_kirchoff,object):

    def nsolve_euler_hucai(self,scale_data,parameters, K, IO):


        Num_steps=scale_data[0]
        dt=scale_data[1]
        sample=scale_data[2]

        gamma=parameters[0]
        c0=parameters[1]
        c1=parameters[2]

        OUTPUT_C, OUTPUT_S = IO.init_kirchhoff_data(scale_data,parameters,K)

        #scale system
        g1=gamma+1
        alpha=(c0*K.f*K.l**3)/(K.k**g1)
        beta=(c1*gamma*K.k**g1)/(K.f*K.f)

        #calc network dynamic
        B,BT=K.get_inicidence_matrices()
        M=K.G.number_of_edges()
        j=0

        for i in range(Num_steps):

            OP=np.dot(np.dot(B,K.C),BT)
            MP=lina.pinv(OP)
            K.V=MP.dot(K.J)
            K.F=np.dot(K.C,np.dot(BT,K.V))

            for m in range(M):
                f_sq=K.F[m] * K.F[m]
                #calc energy
                if i % sample == 0:
                    K.E[j]+= ( ( f_sq ) / K.C[m,m] + (beta/gamma)*  K.C[m,m] ** gamma)
                #calc median
                cg=K.C[m,m]**g1
                dC= alpha * ( ( f_sq )/cg - beta ) * K.C[m,m] * dt

                K.C[m,m] += dC

            if i % sample == 0:
                OUTPUT_C=np.vstack((OUTPUT_C,np.diag(K.C)) )
                OUTPUT_S=np.vstack((OUTPUT_S,K.J) )
                j+=1
            self.print_step(i,Num_steps)

        IO.save_kirchhoff_data(OUTPUT_C,OUTPUT_S,K)

    def nsolve_heun_hucai_reduced(self,scale_data,parameters , K, IO):
        # calculate steady state equivalent to hucai-model
        Num_steps=scale_data[0]
        dt=scale_data[1]
        sample=scale_data[2]

        gamma=parameters[0]
        c0=parameters[1]
        c1=parameters[2]

        eqv_factor=c1**((2.*gamma+1)/(gamma+1.))
        c0*=eqv_factor
        c1/=eqv_factor

        OUTPUT_C, OUTPUT_S = IO.init_kirchhoff_data(scale_data,parameters,K)

        #scale system
        g1=gamma+1
        reciproc_g1=1./g1
        alpha=(c0*K.f*K.l**3)/(K.k**g1)
        beta=(c1*gamma*K.k**g1)/(K.f*K.f)

        #calc network dynamic
        B,BT=K.get_incidence_matrices()
        M=K.G.number_of_edges()
        dC=np.zeros(M)
        j=0

        for i in range(Num_steps):

            #1) prediction
            OP=np.dot(np.dot(B,np.diag(K.C)),BT)
            MP=lina.pinv(OP)
            K.V=MP.dot(K.J)
            K.F=np.dot(np.diag(K.C),np.dot(BT,K.V))

            c_aux=np.copy(np.diag(K.C))

            for m in range(M):

                f_sq=K.F[m] * K.F[m]
                dC[m]= alpha*( ( f_sq )**reciproc_g1 - beta  * np.diag(K.C)[m,m] )* dt
                c_aux[m,m] += dC[m]

            OP_aux=np.dot(np.dot(B,c_aux),BT)
            MP_aux=lina.pinv(OP_aux)
            V_aux=MP_aux.dot(K.J)
            F_aux=np.dot(c_aux,np.dot(BT,V_aux))

            #2) correction
            for m in range(M):
                f_sq=K.F[m] * K.F[m]
                #calc energy
                if i % sample == 0:
                    K.E[j]+= ( ( f_sq ) / np.diag(K.C)[m,m] + (beta/(eqv_factor*gamma))*  np.diag(K.C)[m,m] ** gamma)
                dC_aux= alpha * ( ( F_aux[m] * F_aux[m] )**reciproc_g1 - beta  * c_aux[m,m]) * dt
                #calc median
                np.diag(K.C)[m,m] += ( ( dC[m]+ dC_aux)/2. )

            if i % sample == 0:
                OUTPUT_C=np.vstack((OUTPUT_C,K.C) )
                OUTPUT_S=np.vstack((OUTPUT_S,K.J) )
                j+=1
            self.print_step(i,Num_steps)

        IO.save_kirchhoff_data(OUTPUT_C,OUTPUT_S,K)
    # @profile
    def nsolve_heun_hucai(self,scale_data,parameters , K, IO):

        Num_steps=scale_data[0]
        dt=scale_data[1]
        sample=scale_data[2]

        gamma=parameters[0]
        c0=parameters[1]
        c1=parameters[2]

        OUTPUT_C, OUTPUT_S = IO.init_kirchhoff_data(scale_data,parameters,K)

        #scale system
        g1=gamma+1
        alpha=(c0*K.f*K.l**3)/(K.k**g1)
        beta=(c1*gamma*K.k**g1)/(K.f*K.f)

        #calc network dynamic
        B,BT=K.get_incidence_matrices()
        M=nx.number_of_edges(K.G)
        dC=np.zeros(M)
        j=0
        # print(K.C)
        for i in range(Num_steps):

            #1) prediction
            OP=np.dot(np.dot(B,np.diag(K.C)),BT)
            MP=lina.pinv(OP)
            K.V=MP.dot(K.J)
            K.F=np.multiply(K.C,np.dot(BT,K.V))

            c_aux=np.diag(np.copy(K.C))

            for m in range(M):

                f_sq=K.F[m] * K.F[m]
                cg=K.C[m]**g1

                dC[m]= alpha*( ( f_sq )/cg - beta ) * K.C[m] * dt
                c_aux[m,m] += dC[m]

            OP_aux=np.dot(np.dot(B,c_aux),BT)
            MP_aux=lina.pinv(OP_aux)
            V_aux=MP_aux.dot(K.J)
            F_aux=np.dot(c_aux,np.dot(BT,V_aux))

            #2) correction
            for m in range(M):
                f_sq=K.F[m] * K.F[m]
                #calc energy
                if i % sample == 0:
                    K.E[j]+= ( ( f_sq ) / K.C[m] + (beta/gamma)*  K.C[m] ** gamma)
                #calc median
                cg_aux=c_aux[m,m]**g1
                dC_aux= alpha * ( ( F_aux[m] * F_aux[m] )/cg_aux - beta ) * c_aux[m,m] * dt

                K.C[m] += ( ( dC[m] + dC_aux)/2. )

            if i % sample == 0:
                OUTPUT_C=np.vstack((OUTPUT_C,K.C[:]))
                OUTPUT_S=np.vstack((OUTPUT_S,K.J) )
                j+=1
            self.print_step(i,Num_steps)

        IO.save_kirchhoff_data(OUTPUT_C,OUTPUT_S,K)

    def nsolve_rungekutta_hucai(self,scale_data,parameters , K, IO):

        Num_steps=scale_data[0]
        dt=scale_data[1]
        dt2=0.5* dt
        sample=scale_data[2]

        gamma=parameters[0]
        c0=parameters[1]
        c1=parameters[2]

        OUTPUT_C, OUTPUT_S = IO.init_kirchhoff_data(scale_data,parameters,K)

        #scale system
        g1=gamma+1
        alpha=(c0*K.f*K.l**3)/(K.k**g1)
        beta=(c1*gamma*K.k**g1)/(K.f*K.f)

        #calc network dynamic
        B,BT=K.get_inicidence_matrices()
        M=K.G.number_of_edges()

        k1=np.zeros(M)
        k2=np.zeros(M)
        k3=np.zeros(M)
        k4=np.zeros(M)
        j=0

        for i in range(Num_steps):

            OP=np.dot(np.dot(B,K.C),BT)
            MP=lina.pinv(OP)
            K.V=MP.dot(K.J)
            K.F=np.dot(K.C,np.dot(BT,K.V))
            c_aux1=np.copy(K.C)

            #1) prediction k1
            for m in range(M):

                f_sq=K.F[m] * K.F[m]
                cg=K.C[m,m]**g1
                k1[m]= alpha*( ( f_sq )/cg - beta ) * K.C[m,m]
                c_aux1[m,m] += k1[m]* dt2
                if i % sample == 0:
                    K.E[j]+= ( ( f_sq ) / K.C[m,m] + (beta/gamma)*  K.C[m,m] ** gamma)

            #2) prediction k2
            OP_aux=np.dot(np.dot(B,c_aux1),BT)
            MP_aux=lina.pinv(OP_aux)
            V_aux=MP.dot(K.J)
            F_aux=np.dot(c_aux1,np.dot(BT,V_aux))
            c_aux2=np.copy(K.C)

            for m in range(M):
                f_sq=F_aux[m] * F_aux[m]
                cg=c_aux1[m,m]**g1
                k2[m]= alpha*( ( f_sq )/cg - beta ) * c_aux1[m,m]
                c_aux2[m,m] += k2[m]*dt2

            #3) prediction k3
            OP_aux=np.dot(np.dot(B,c_aux2),BT)
            MP_aux=lina.pinv(OP_aux)
            V_aux=MP.dot(K.J)
            F_aux=np.dot(c_aux2,np.dot(BT,V_aux))
            c_aux3=np.copy(K.C)

            for m in range(M):
                f_sq=F_aux[m] * F_aux[m]
                cg=c_aux2[m,m]**g1
                k3[m]= alpha*( ( f_sq )/cg - beta ) * c_aux2[m,m]
                c_aux3[m,m] += k3[m]* dt

            #3) prediction k4 and correction
            OP_aux=np.dot(np.dot(B,c_aux3),BT)
            MP_aux=lina.pinv(OP_aux)
            V_aux=MP.dot(K.J)
            F_aux=np.dot(c_aux3,np.dot(BT,V_aux))
            for m in range(M):

                f_sq=F_aux[m] * F_aux[m]
                cg=c_aux3[m,m]**g1
                k4[m]= alpha*( ( f_sq )/cg - beta ) * c_aux3[m,m]

                K.C[m,m] += (k1[m]+2*k2[m]+2*k3[m]+k4[m])*dt/6

            if i % sample == 0:
                OUTPUT_C=np.vstack((OUTPUT_C,K.C))
                OUTPUT_S=np.vstack((OUTPUT_S,K.J) )
                j+=1
            self.print_step(i,Num_steps)

        IO.save_kirchhoff_data(OUTPUT_C,OUTPUT_S,K)

    def nsolve_heun_hucai_R_tracked(self,scale_data,parameters, K, IO):
        K1=K
        nullity =[]
        shear = []

        # rebranding conductivity Ki.C -> K_i
        kappa=1.

        Num_steps=scale_data[0]
        dt=scale_data[1]
        sample=scale_data[2]

        gamma=parameters[0]
        c0=parameters[1]
        c1=parameters[2]

        OUTPUT_C1, OUTPUT_S1 = IO.init_kirchhoff_data(scale_data,parameters,K1)

        #scale system
        g1=gamma-1
        alpha1=(c0*K1.f**g1)/((K1.k**gamma)*(K1.l**(3*g1)))
        beta1=(c1**K1.k*(K1.l**3)/K1.f)**gamma

        #calc network dynamic
        B1,BT1=K1.get_incidence_matrices()
        M1=K1.G.number_of_edges()
        dC1=np.zeros(M1)

        j=0
        # establish threshold for pruning events
        threshold=10.**(-20)
        for i in range(Num_steps):

            # 1) prediction

            K_1=np.diag( kappa* np.diag(np.copy(K1.C))**4 )
            OP1=np.dot(np.dot(B1,K_1),BT1)
            MP1=lina.pinv(OP1)
            K1.V=MP1.dot(K1.J)
            d_V1=np.dot(BT1,K1.V)

            c_aux1=np.copy(K1.C)
            for m in range(M1):
                if K1.C[m,m] > threshold:
                    dC1[m]= (alpha1*( (np.fabs(d_V1[m])* K1.C[m,m])**gamma - beta1 ) * K1.C[m,m] )* dt
                    c_aux1[m,m] += dC1[m]
                else:
                    K1.C[m,m]=10.**(-21)
                    c_aux1[m,m] =10.**(-21)

            #2) correction
            K_1_aux=np.diag( kappa* np.diag(np.copy(c_aux1))**4 )

            OP1_aux=np.dot(np.dot(B1,K_1_aux),BT1)
            MP1_aux=lina.pinv(OP1_aux)
            V1_aux=MP1_aux.dot(K1.J)
            d_V1_aux=np.dot(BT1,V1_aux)

            for m in range(M1):
                #calc median
                if K1.C[m,m] > threshold:
                    dC_aux1= (alpha1 * ( (np.fabs(d_V1_aux)[m] * c_aux1[m,m])**gamma - beta1 ) * c_aux1[m,m]  ) * dt
                    K1.C[m,m] += ( ( dC1[m] + dC_aux1)/2. )

            if i % sample == 0:
                OUTPUT_C1=np.vstack((OUTPUT_C1,np.diag(K1.C)) )
                OUTPUT_S1=np.vstack((OUTPUT_S1,K1.J) )
                s1=0
                K1.clipp_graph()
                n1=(1.+nx.number_of_edges(K1.H)-nx.number_of_nodes(K1.H))/(1.+nx.number_of_edges(K1.G)-nx.number_of_nodes(K1.G))

                for m in range(M1):
                    s1+=np.fabs(d_V1)[m]*OUTPUT_C1[-1,m]

                nullity.append(n1)
                shear.append(s1)

                j+=1

            self.print_step(i,Num_steps)

        IO.save_kirchhoff_data(OUTPUT_C1,OUTPUT_S1,K1)
        IO.save_nparray(nullity,'nullity_time')
        IO.save_nparray(shear,'shear_time')

    def nsolve_heun_chang_C(self,scale_data,parameters, K, IO):

            nullity =[]
            cost = []

            # rebranding conductivity Ki.C -> K_i
            kappa=1.

            Num_steps=scale_data[0]
            dt=scale_data[1]
            sample=scale_data[2]

            OUTPUT_C, OUTPUT_S = IO.init_kirchhoff_data(scale_data,parameters,K)

            B,BT=K.get_incidence_matrices()

            M=K.G.number_of_edges()
            Q_ref=np.ones(M)

            # establish threshold for pruning events

            for i in range(Num_steps):
                # predict
                K_C=np.diag(K.C)
                OP=np.dot(np.dot(B,K_C),BT)
                D=lina.pinv(OP)
                BD=np.dot(BT,D)

                dV=np.dot(BD,K.J)
                stress=np.diag(np.multiply(K.C,dV))
                P=np.subtract(np.dot(K_C,np.dot(BD,B)),np.identity(M))
                Q=np.dot(K_C,dV)
                sign=np.divide(Q,np.absolute(Q))
                dQ=np.subtract(Q_ref,np.multiply(Q,sign))

                dC_aux=np.dot(dQ,np.dot(P,stress))*dt

                # correct
                K_C_aux=np.diag(np.add(K.C,dC_aux))
                OP=np.dot(np.dot(B,K_C_aux),BT)
                D=lina.pinv(OP)
                BD=np.dot(BT,D)

                dV=np.dot(BD,K.J)
                stress=np.diag(np.multiply(np.diag(K_C_aux),dV))
                P=np.subtract(np.dot(K_C_aux,np.dot(BD,B)),np.identity(M))
                Q=np.dot(K_C_aux,dV)
                sign=np.divide(Q,np.absolute(Q))
                dQ=np.subtract(Q_ref,np.multiply(Q,sign))

                dC=np.dot(dQ,np.dot(P,stress))*dt

                #final increment
                K.C=np.add(K.C,np.add(dC,dC_aux)/2.)

                if i % sample == 0:
                    OUTPUT_C=np.vstack((OUTPUT_C,K.C) )
                    OUTPUT_S=np.vstack((OUTPUT_S,K.J) )

                    K.clipp_graph()
                    n=(1.+nx.number_of_edges(K.H)-nx.number_of_nodes(K.H))/(1.+nx.number_of_edges(K.G)-nx.number_of_nodes(K.G))
                    s=np.dot(dQ,dQ)

                    nullity.append(n)
                    cost.append(s)
                self.print_step(i,Num_steps)

            IO.save_kirchhoff_data(OUTPUT_C,OUTPUT_S,K)
            IO.save_nparray(nullity,'nullity_time')
            IO.save_nparray(cost,'cost_time')

    def nsolve_heun_hucai_oscillation(self,scale_data,parameters , K, IO):

        Num_steps=scale_data[0]
        dt=scale_data[1]
        sample=scale_data[2]
        nullity=[]
        c_mean=[]
        gamma=parameters[0]
        scale=parameters[1]
        volume_penalty=parameters[2]
        mode=parameters[3]

        OUTPUT_C, OUTPUT_S = IO.init_kirchhoff_data(scale_data,parameters,K)

        #scale system
        g1_p=gamma+1
        g1_m=1.-gamma

        #calc network dynamic
        B,BT=K.get_incidence_matrices()
        M=nx.number_of_edges(K.G)
        N=nx.number_of_nodes(K.G)
        dC=np.zeros(M)
        j=0
        for i in range(Num_steps):
            S=self.propagate_sources(mode,K,i*dt)

            #1) prediction
            v_sq=self.calc_sq_pressure(K.C,B,BT,S)
            cg_m=np.power(K.C,g1_m)
            stress_volume=np.subtract( np.multiply(v_sq,cg_m) , volume_penalty )
            dC= np.multiply( scale * dt, np.multiply(K.C, stress_volume  ))
            c_aux= np.add(K.C,dC)

            #2) correction

            S_aux=self.propagate_sources(mode,K,(i+1)*dt)

            v_sq_aux=self.calc_sq_pressure(c_aux,B,BT,S_aux)
            cg_aux=np.power(c_aux,g1_m)
            stress_volume=np.subtract( np.multiply(v_sq_aux,cg_aux) , volume_penalty )
            dC_aux= np.multiply( scale * dt, np.multiply(c_aux, stress_volume  ))

            K.C = np.add( K.C, np.divide(np.add( dC , dC_aux),2.) )
            if i % sample == 0:
                # K.E[j] = np.add( np.sum(np.multiply( v_sq_aux, K.C)) ,  * np.sum( np.power(K.C, gamma)))
                OUTPUT_C = np.vstack((OUTPUT_C,K.C[:]))
                OUTPUT_S = np.vstack((OUTPUT_S,S) )
                # j+=1

                K.clipp_graph()
                H=nx.Graph(K.H)
                n=(nx.number_connected_components(H)+nx.number_of_edges(H)-nx.number_of_nodes(H))/(1.+M-N)
                nullity.append(n)
                c_mean.append(np.mean(K.H_C))

            self.print_step(i,Num_steps)

        IO.save_kirchhoff_data(OUTPUT_C,OUTPUT_S,K)
        IO.save_nparray(nullity,'nullity_time')
        IO.save_nparray(c_mean,'conductivity_mean_time')

    def nsolve_heun_hucai_oscillation_temporal_avg(self,scale_data,parameters , K, IO):

        Num_steps=scale_data[0]
        dt=scale_data[1]
        sample=scale_data[2]
        nullity=[]
        c_mean=[]
        gamma=parameters[0]
        scale=parameters[1]
        volume_penalty=parameters[2]
        mode=parameters[3]

        OUTPUT_C, OUTPUT_S = IO.init_kirchhoff_data(scale_data,parameters,K)

        #scale system
        g1_p=gamma+1
        g1_m=1.-gamma

        #calc network dynamic
        B,BT=K.get_incidence_matrices()
        M=nx.number_of_edges(K.G)
        N=nx.number_of_nodes(K.G)
        dC=np.zeros(M)

        idx=np.where(K.J < 0)[0]
        x=np.where(K.J > 0)[0][0]
        # j=0
        for i in range(Num_steps):

            #1) prediction
            v_sq=self.calc_sq_pressure_temporal_mean(mode,[idx,x],K.C,i*dt,B,BT)
            cg_m=np.power(K.C,g1_m)
            stress_volume=np.subtract( np.multiply(v_sq,cg_m) , volume_penalty )
            dC= np.multiply( scale * dt, np.multiply(K.C, stress_volume  ))
            c_aux= np.add(K.C,dC)

            #2) correction
            v_sq_aux=self.calc_sq_pressure_temporal_mean(mode,[idx,x],c_aux,(i+1)*dt,B,BT)
            cg_aux=np.power(c_aux,g1_m)
            stress_volume=np.subtract( np.multiply(v_sq_aux,cg_aux) , volume_penalty )
            dC_aux= np.multiply( scale * dt, np.multiply(c_aux, stress_volume  ))

            K.C = np.add( K.C, np.divide(np.add( dC , dC_aux),2.) )
            if i % sample == 0:
                # K.E[j] = np.add( np.sum(np.multiply( v_sq_aux, K.C)) ,  * np.sum( np.power(K.C, gamma)))
                S=np.zeros(N)
                OUTPUT_C = np.vstack((OUTPUT_C,K.C[:]))
                OUTPUT_S = np.vstack((OUTPUT_S,S) )
                # j+=1

                K.clipp_graph()
                H=nx.Graph(K.H)
                n=(nx.number_connected_components(H)+nx.number_of_edges(H)-nx.number_of_nodes(H))/(1.+M-N)
                nullity.append(n)
                c_mean.append(np.mean(K.H_C))

            self.print_step(i,Num_steps)

        IO.save_kirchhoff_data(OUTPUT_C,OUTPUT_S,K)
        IO.save_nparray(nullity,'nullity_time')
        IO.save_nparray(c_mean,'conductivity_mean_time')

    def nsolve_heun_hucai_fluctuation(self,scale_data,parameters , K, IO):

        Num_steps=scale_data[0]
        dt=scale_data[1]
        sample=scale_data[2]
        nullity=[]
        gamma=parameters[0]
        scale=parameters[1]
        volume_penalty=parameters[2]
        coupling=[parameters[3],parameters[4]]
        OUTPUT_C, OUTPUT_S = IO.init_kirchhoff_data(scale_data,parameters,K)

        #scale system
        g1_p=gamma+1
        g1_m=1.-gamma

        #calc network dynamic
        B,BT=K.get_incidence_matrices()
        M=nx.number_of_edges(K.G)
        N=nx.number_of_nodes(K.G)
        dC=np.zeros(M)
        j=0
        idx=np.where(K.J < 0)[0]
        x=np.where(K.J > 0)[0][0]
        for i in range(Num_steps):

            #1) prediction
            v_sq=self.calc_sq_pressure_general(K.C,B,BT,[idx,x],coupling)
            cg_m=np.power(K.C,g1_m)
            stress_volume=np.subtract( np.multiply(v_sq,cg_m) , volume_penalty )
            dC= np.multiply( scale * dt, np.multiply( K.C, stress_volume  ))
            c_aux= np.add(K.C,dC)

            #2) correction
            v_sq_aux=self.calc_sq_pressure_general(c_aux,B,BT,[idx,x],coupling)
            cg_aux=np.power(c_aux,g1_m)
            stress_volume=np.subtract( np.multiply(v_sq_aux,cg_aux) , volume_penalty )
            dC_aux= np.multiply( scale * dt, np.multiply(c_aux, stress_volume  ))

            K.C = np.add( K.C, np.divide(np.add( dC , dC_aux),2.) )
            if i % sample == 0:
                K.E[j] = np.add( np.sum(np.multiply( v_sq_aux, K.C)) ,  np.sum( np.power(K.C, gamma)))
                OUTPUT_C = np.vstack((OUTPUT_C,K.C[:]))
                S=np.zeros(N)
                OUTPUT_S = np.vstack((OUTPUT_S,S) )
                j+=1

                K.clipp_graph()
                H=nx.Graph(K.H)
                n=(nx.number_connected_components(H)+nx.number_of_edges(H)-nx.number_of_nodes(H))/(1.+M-N)
                nullity.append(n)

            self.print_step(i,Num_steps)

        IO.save_kirchhoff_data(OUTPUT_C,OUTPUT_S,K)
        IO.save_nparray(nullity,'nullity_time')

class integrate_stochastic(integrate_kirchoff,object):

    def nsolve_heun_hucai_manual_fluctuation(self,scale_data,parameters , K, IO):

        Num_steps=scale_data[0]
        dt=scale_data[1]
        sample=scale_data[2]
        nullity=[]
        c_mean=[]
        gamma=parameters[0]
        scale=parameters[1]
        volume_penalty=parameters[2]
        mode=parameters[3]
        num_realizations=parameters[4]
        p=parameters[5]
        source=np.where(K.J> 0)[0][0]
        sinks=np.where(K.J < 0)[0]
        OUTPUT_C, OUTPUT_S = IO.init_kirchhoff_data(scale_data,parameters,K)

        #scale system
        g1_m=1.-gamma

        #calc network dynamic
        B,BT=K.get_incidence_matrices()
        M=nx.number_of_edges(K.G)
        N=nx.number_of_nodes(K.G)
        dC=np.zeros(M)
        dT=(np.ones(M)*dt)
        threshold=10.**(-20)

        for i in range(Num_steps):


            #1) prediction
            f_sq,v_sq=self.calc_sq_flow_random_manual(K.C,B,BT,[mode,N,num_realizations,source,sinks,p])
            cg_m=np.power(K.C,g1_m)
            stress_volume=np.subtract( np.multiply(v_sq,cg_m) , volume_penalty )
            dC= np.multiply( scale * dt, np.multiply(K.C, stress_volume  ))
            c_aux= np.add(K.C,dC)
            control=np.where( c_aux < threshold )
            for m in control:
                dT[m]=0.
                c_aux[m]=10.**(-21)

            #2) correction
            f_sq_aux,v_sq_aux=self.calc_sq_flow_random_manual(c_aux,B,BT,[mode,N,num_realizations,source,sinks,p])
            cg_aux=np.power(c_aux,g1_m)
            stress_volume=np.subtract( np.multiply(v_sq_aux,cg_aux) , volume_penalty )
            dC_aux= np.multiply( scale * dt, np.multiply(c_aux, stress_volume  ))

            K.C = np.add( K.C, np.divide(np.add( dC , dC_aux),2.) )
            control=np.where( K.C < threshold )
            for m in control:
                dT[m]=0.
                K.C[m]=10.**(-21)
            if i % sample == 0:
                # K.E[j] = np.add( np.sum(np.multiply( v_sq_aux, K.C)) ,  * np.sum( np.power(K.C, gamma)))
                OUTPUT_C = np.vstack((OUTPUT_C,K.C[:]))
                # j+=1

                K.clipp_graph()
                H=nx.Graph(K.H)
                n=(nx.number_connected_components(H)+nx.number_of_edges(H)-nx.number_of_nodes(H))/(1.+M-N)
                nullity.append(n)
                c_mean.append(np.mean(K.H_C))

            self.print_step(i,Num_steps)

        IO.save_kirchhoff_data(OUTPUT_C,OUTPUT_S,K)
        IO.save_nparray(nullity,'nullity_time')

    def nsolve_euler_hucai_R_random(self,scale_data,parameters, K, IO):

        Num_steps=scale_data[0]
        dt=scale_data[1]
        sample=scale_data[2]

        gamma=parameters[0]
        g1=gamma+1
        c0=parameters[1]
        c1=parameters[2]
        mu=parameters[3]
        var=parameters[4]
        kappa=1.
        threshold=10.**(-20)
        OUTPUT_C, OUTPUT_S = IO.init_kirchhoff_data(scale_data,parameters,K)
        nullity =[]
        dissipation=[]
        volume=[]
        #calc network dynamic

        B,BT=K.get_inicidence_matrices()
        M=nx.number_of_edges(K.G)
        N=nx.number_of_nodes(K.G)
        dC=np.zeros(M)
        self.setup_random_fluctuations(N,mu,var)

        # unit/scales
        alpha=(c0*K.f**g1)/((K.k**gamma)*(K.l**(3*g1)))
        ALPHA=(np.ones(M)*alpha)
        beta=(c1**K.k*(K.l**3)/K.f)**gamma
        BETA=(np.ones(M)*beta)
        HALF=(np.ones(M)*0.5)
        REC_3=(np.ones(M)*(-3.))
        SIGMA=(np.ones(M)*gamma)
        dT=(np.ones(M)*dt)
        KAPPA=(np.ones(M)*kappa)

        for i in range(Num_steps):

            c_aux=K.C[:]
            C=np.diag( np.multiply(KAPPA,c_aux**4 ) )
            F_sq=self.calc_sq_flow_random(C,B,BT)

            tau=np.multiply(np.power(F_sq,HALF),np.power(c_aux,REC_3))
            shear_sigma=np.power(tau,SIGMA)
            diff_shear=np.multiply(np.subtract(shear_sigma,BETA),c_aux)
            dC=np.multiply(np.multiply(ALPHA,diff_shear),dT)

            K.C[:]=np.add(c_aux,dC)

            if np.any(K.C[:] < threshold):
                for m in range(M):
                    if K.C[m] < threshold:
                        dT[m]=0.
                        K.C[m]=10.**(-21)

            if i % sample == 0:
                OUTPUT_C=np.vstack((OUTPUT_C,K.C) )
                OUTPUT_S=np.vstack((OUTPUT_S,K.J) )
                s=0
                K.clipp_graph()
                n=(1.+nx.number_of_edges(K.H)-nx.number_of_nodes(K.H))/(1.+nx.number_of_edges(K.G)-nx.number_of_nodes(K.G))
                nullity.append(n)
                d=np.divide(F_sq,np.diag(C))
                dissipation.append(np.sum(d))
                volume.append(np.sum(K.C[:]**2))

            self.print_step(i,Num_steps)

        IO.save_kirchhoff_data(OUTPUT_C,OUTPUT_S,K)
        IO.save_nparray(nullity,'nullity_time')
        IO.save_nparray(dissipation,'dissipation_time')
        IO.save_nparray(volume,'volume_time')

    def nsolve_euler_hucai_R_random_manual_sink(self,scale_data,parameters, data_manual_sink, K, IO):

        Num_steps=scale_data[0]
        dt=scale_data[1]
        sample=scale_data[2]

        gamma=parameters[0]
        g1=gamma+1
        c0=parameters[1]
        c1=parameters[2]

        kappa=1.
        threshold=10.**(-20)
        OUTPUT_C, OUTPUT_S = IO.init_kirchhoff_data(scale_data,parameters,K)
        nullity =[]
        dissipation=[]
        volume=[]
        #calc network dynamic

        B,BT=K.get_inicidence_matrices()
        M=nx.number_of_edges(K.G)
        N=nx.number_of_nodes(K.G)
        dC=np.zeros(M)

        # unit/scales
        alpha=(c0*K.f**g1)/((K.k**gamma)*(K.l**(3*g1)))
        ALPHA=(np.ones(M)*alpha)
        beta=(c1**K.k*(K.l**3)/K.f)**gamma
        BETA=(np.ones(M)*beta)
        HALF=(np.ones(M)*0.5)
        REC_3=(np.ones(M)*(-3.))
        SIGMA=(np.ones(M)*gamma)
        dT=(np.ones(M)*dt)
        KAPPA=(np.ones(M)*kappa)

        for i in range(Num_steps):

            c_aux=K.C[:]
            C=np.diag( np.multiply(KAPPA,c_aux**4 ) )
            F_sq=self.calc_sq_flow_random_manual(C,B,BT,data_manual_sink)

            tau=np.multiply(np.power(F_sq,HALF),np.power(c_aux,REC_3))
            shear_sigma=np.power(tau,SIGMA)
            diff_shear=np.multiply(np.subtract(shear_sigma,BETA),c_aux)
            dC=np.multiply(np.multiply(ALPHA,diff_shear),dT)

            K.C[:]=np.add(c_aux,dC)

            if np.any(K.C[:] < threshold):
                for m in range(M):
                    if K.C[m] < threshold:
                        dT[m]=0.
                        K.C[m]=10.**(-21)

            if i % sample == 0:
                OUTPUT_C=np.vstack((OUTPUT_C,K.C) )
                OUTPUT_S=np.vstack((OUTPUT_S,K.J) )
                s=0
                K.clipp_graph()
                n=(1.+nx.number_of_edges(K.H)-nx.number_of_nodes(K.H))/(1.+nx.number_of_edges(K.G)-nx.number_of_nodes(K.G))
                nullity.append(n)
                d=np.divide(F_sq,np.diag(C))
                dissipation.append(np.sum(d))
                volume.append(np.sum(K.C[:]**2))

            self.print_step(i,Num_steps)

        IO.save_kirchhoff_data(OUTPUT_C,OUTPUT_S,K)
        IO.save_nparray(nullity,'nullity_time')
        IO.save_nparray(dissipation,'dissipation_time')
        IO.save_nparray(volume,'volume_time')

    def nsolve_euler_hucai_random_scaling(self,scale_data,parameters,K,IO):

        Num_steps=scale_data[0]
        dt=scale_data[1]
        sample=scale_data[2]

        gamma=parameters[0]
        g1=gamma+1
        c0=parameters[1]
        c1=parameters[2]
        mu=parameters[3]
        var=parameters[4]
        kappa=1.
        threshold=10.**(-20)
        OUTPUT_C, OUTPUT_S = IO.init_kirchhoff_data(scale_data,parameters,K)
        nullity =[]
        dissipation=[]
        volume=[]
        #calc network dynamic

        B,BT=K.get_inicidence_matrices()
        M=nx.number_of_edges(K.G)
        N=nx.number_of_nodes(K.G)
        dC=np.zeros(M)
        self.setup_random_fluctuations(N,mu,var)

        # unit/scales
        alpha=(c0*K.f**g1)/((K.k**gamma)*(K.l**(3*g1)))
        ALPHA=(np.ones(M)*alpha)
        beta=(c1**K.k*(K.l**3)/K.f)**gamma
        BETA=(np.ones(M)*beta)
        dT=(np.ones(M)*dt)
        KAPPA=(np.ones(M)*kappa)

        for i in range(Num_steps):
            self.mu+=(1./tau)*self.mu*dT[0]
            c_aux=K.C[:]
            F_sq=self.calc_sq_flow_random(K.C[:],B,BT)

            dissipation=np.divide(F_sq,np.power(K.C[:],g1))
            diff_shear=np.multiply(np.subtract(shear_sigma,BETA),K.C[:])
            dC=np.multiply(np.multiply(ALPHA,diff_shear),dT)
            K.C[:]=np.add(K.C[:],dC)

            if np.any(K.C[:] < threshold):
                for m in range(M):
                    if K.C[m] < threshold:
                        dT[m]=0.
                        K.C[m]=10.**(-21)

            if i % sample == 0:
                OUTPUT_C=np.vstack((OUTPUT_C,K.C) )
                OUTPUT_S=np.vstack((OUTPUT_S,K.J) )
                s=0
                K.clipp_graph()
                n=(1.+nx.number_of_edges(K.H)-nx.number_of_nodes(K.H))/(1.+nx.number_of_edges(K.G)-nx.number_of_nodes(K.G))
                nullity.append(n)
                d=np.divide(F_sq,np.diag(C))
                dissipation.append(np.sum(d))
                volume.append(np.sum(K.C[:]**2))

            self.print_step(i,Num_steps)

        IO.save_kirchhoff_data(OUTPUT_C,OUTPUT_S,K)
        IO.save_nparray(nullity,'nullity_time')
        IO.save_nparray(dissipation,'dissipation_time')
        IO.save_nparray(volume,'volume_time')

    def nsolve_euler_hucai_random_reduced(self,scale_data,parameters,K,IO):

        Num_steps=scale_data[0]
        dt=scale_data[1]
        sample=scale_data[2]

        gamma=parameters[0]
        g1=gamma+1
        c0=parameters[1]
        c1=parameters[2]
        mu=parameters[3]
        var=parameters[4]
        kappa=1.
        threshold=10.**(-20)
        OUTPUT_C, OUTPUT_S = IO.init_kirchhoff_data(scale_data,parameters,K)
        nullity =[]
        dissipation=[]
        volume=[]
        #calc network dynamic

        B,BT=K.get_inicidence_matrices()
        M=nx.number_of_edges(K.G)
        N=nx.number_of_nodes(K.G)
        dC=np.zeros(M)
        self.setup_random_fluctuations_reduced(N,mu,var)

        # unit/scales
        alpha=(c0*K.f**g1)/((K.k**gamma)*(K.l**(3*g1)))
        ALPHA=(np.ones(M)*alpha)
        beta=(c1**K.k*(K.l**3)/K.f)**gamma
        BETA=(np.ones(M)*beta)
        HALF=(np.ones(M)*0.5)
        REC_3=(np.ones(M)*(-3.))
        SIGMA=(np.ones(M)*gamma)
        dT=(np.ones(M)*dt)
        KAPPA=(np.ones(M)*kappa)


        for i in range(Num_steps):

            c_aux=K.C[:]
            C=np.diag( np.multiply(KAPPA,c_aux**4 ) )
            F_sq=self.calc_sq_flow_random_reduced(C,B,BT)

            tau=np.multiply(np.power(F_sq,HALF),np.power(c_aux,REC_3))
            shear_sigma=np.power(tau,SIGMA)
            diff_shear=np.multiply(np.subtract(shear_sigma,BETA),c_aux)
            dC=np.multiply(np.multiply(ALPHA,diff_shear),dT)

            K.C[:]=np.add(c_aux,dC)

            if np.any(K.C[:] < threshold):
                for m in range(M):
                    if K.C[m] < threshold:
                        dT[m]=0.
                        K.C[m]=10.**(-21)

            if i % sample == 0:
                OUTPUT_C=np.vstack((OUTPUT_C,K.C) )
                OUTPUT_S=np.vstack((OUTPUT_S,K.J) )
                s=0
                K.clipp_graph()
                n=(1.+nx.number_of_edges(K.H)-nx.number_of_nodes(K.H))/(1.+nx.number_of_edges(K.G)-nx.number_of_nodes(K.G))
                nullity.append(n)
                d=np.divide(F_sq,np.diag(C))
                dissipation.append(np.sum(d))
                volume.append(np.sum(K.C[:]**2))

            self.print_step(i,Num_steps)

        IO.save_kirchhoff_data(OUTPUT_C,OUTPUT_S,K)
        IO.save_nparray(nullity,'nullity_time')
        IO.save_nparray(dissipation,'dissipation_time')
        IO.save_nparray(volume,'volume_time')

    def nsolve_heun_hucai_reduced_radius(self,scale_data,parameters,K,IO):

        Num_steps=scale_data[0]
        dt=scale_data[1]
        sample=scale_data[2]

        gamma=parameters[0]
        g_p1=gamma+1
        g_4m1=4.*(1-gamma)

        c0=parameters[1]
        c1=parameters[2]
        mu=parameters[3]
        var=parameters[4]

        OUTPUT_C, OUTPUT_S = IO.init_kirchhoff_data(scale_data,parameters,K)
        nullity =[]
        dissipation=[]
        volume=[]

        #calc network dynamic
        B,BT=K.get_incidence_matrices()
        # B=ssp.csc_matrix(B)
        # BT=ssp.csc_matrix(BT)
        # print(len(BT.getnnz(1)))

        M=nx.number_of_edges(K.G)
        # N=nx.number_of_nodes(K.G)
        C=K.C[:]

        R=np.power(np.divide(C,self.kappa),0.25)

        # unit/scales
        self.setup_random_fluctuations_reduced(K,mu,var)

        alpha=(c0*K.f**g_p1)/((K.k**gamma)*(K.l**(3*g_p1)))
        beta=(c1**K.k*(K.l**3)/K.f)**gamma
        dT=(np.ones(M)*dt*alpha)
        sigma=self.sigma/2.
        threshold=10.**(-20)

        for i in range(Num_steps):

            #prediction
            dV_sq, F_sq=self.calc_sq_flow(C,B,BT)

            R_gamma=np.power(R,g_4m1)
            shear_sigma=np.power(np.multiply(dV_sq,R_gamma), sigma)
            diff_shear=np.multiply(np.subtract(shear_sigma,beta),R)
            prediction_dR=np.multiply(diff_shear,dT)
            prediction_R=np.add(R,prediction_dR)

            C=np.multiply(np.power(prediction_R,4.),self.kappa)
            control=np.where( C < threshold )
            for m in control:
                dT[m]=0.
                C[m]=10.**(-21)
            #correction
            dV_sq, F_sq=self.calc_sq_flow(C,B,BT)

            R_gamma=np.power(prediction_R,g_4m1)
            shear_sigma=np.power(np.multiply(dV_sq,R_gamma), sigma)
            diff_shear=np.multiply(np.subtract(shear_sigma,beta),prediction_R)
            correction_dR=np.multiply(diff_shear,dT)

            #update
            dR=0.5*(prediction_dR+correction_dR)
            R=np.add(R,dR)
            K.C=np.multiply(np.power(R,4.),self.kappa)
            control=np.where( K.C < threshold )
            for m in control:
                dT[m]=0.
                K.C[m]=10.**(-21)
            C=K.C[:]

            # output
            if i % sample == 0:
                OUTPUT_C=np.vstack((OUTPUT_C,K.C) )
                OUTPUT_S=np.vstack((OUTPUT_S,K.J) )
                s=0
                K.clipp_graph()
                n=(1.+nx.number_of_edges(K.H)-nx.number_of_nodes(K.H))/(1.+nx.number_of_edges(K.G)-nx.number_of_nodes(K.G))
                nullity.append(n)
                D=np.divide(F_sq,K.C[:])
                dissipation.append(np.sum(D))
                A=np.power(R,2.)
                volume.append(np.sum(A))

                dV_sq, F_sq=self.calc_sq_flow(C,B,BT)
                # K.V=np.sqrt(dV_sq)
                K.F=np.sqrt(F_sq)

            self.print_step(i,Num_steps)

        IO.save_kirchhoff_data(OUTPUT_C,OUTPUT_S,K)
        IO.save_nparray(nullity,'nullity_time')
        IO.save_nparray(dissipation,'dissipation_time')
        IO.save_nparray(volume,'volume_time')

    def nsolve_heun_hucai_effective_radius(self,scale_data,parameters,K,IO):

        Num_steps=scale_data[0]
        dt=scale_data[1]
        sample=scale_data[2]

        gamma=parameters[0]
        g_p1=gamma+1
        g_4m1=4.*(1-gamma)

        self.scale=parameters[1]
        self.vol_diss=parameters[2]
        self.noise=parameters[3]

        OUTPUT_C, OUTPUT_S = IO.init_kirchhoff_data(scale_data,parameters,K)
        nullity =[]
        dissipation=[]
        volume=[]

        #calc network dynamic
        B,BT=K.get_incidence_matrices()

        self.M=nx.number_of_edges(K.G)
        self.N=nx.number_of_nodes(K.G)
        C=K.C[:]

        R=np.power(np.divide(C,self.kappa),0.25)

        # unit/scales
        self.setup_random_fluctuations_effective(K)

        alpha=(self.scale*K.f**g_p1)/((K.k**gamma)*(K.l**(3*g_p1)))
        beta=(self.vol_diss**K.k*(K.l**3)/K.f)**gamma
        dT=(np.ones(self.M)*dt*alpha)
        sigma=self.sigma/2.
        threshold=10.**(-20)

        for i in range(Num_steps):

            #prediction
            dV_sq, F_sq=self.calc_sq_flow(C,B,BT)

            R_gamma=np.power(R,g_4m1)
            shear_sigma=np.power(np.multiply(dV_sq,R_gamma), sigma)
            diff_shear=np.multiply(np.subtract(shear_sigma,beta),R)
            prediction_dR=np.multiply(diff_shear,dT)
            prediction_R=np.add(R,prediction_dR)

            C=np.multiply(np.power(prediction_R,4.),self.kappa)
            control=np.where( C < threshold )
            for m in control:
                dT[m]=0.
                C[m]=10.**(-21)
            #correction
            dV_sq, F_sq=self.calc_sq_flow(C,B,BT)

            R_gamma=np.power(prediction_R,g_4m1)
            shear_sigma=np.power(np.multiply(dV_sq,R_gamma), sigma)
            diff_shear=np.multiply(np.subtract(shear_sigma,beta),prediction_R)
            correction_dR=np.multiply(diff_shear,dT)

            #update
            dR=0.5*(prediction_dR+correction_dR)
            R=np.add(R,dR)
            K.C=np.multiply(np.power(R,4.),self.kappa)
            control=np.where( K.C < threshold )
            for m in control:
                dT[m]=0.
                K.C[m]=10.**(-21)
            C=K.C[:]

            # output
            if i % sample == 0:
                OUTPUT_C=np.vstack((OUTPUT_C,K.C) )
                OUTPUT_S=np.vstack((OUTPUT_S,K.J) )
                s=0
                K.clipp_graph()
                n=(1.+nx.number_of_edges(K.H)-nx.number_of_nodes(K.H))/(1.+nx.number_of_edges(K.G)-nx.number_of_nodes(K.G))
                nullity.append(n)
                D=np.divide(F_sq,K.C[:])
                dissipation.append(np.sum(D))
                A=np.power(R,2.)
                volume.append(np.sum(A))

                dV_sq, F_sq=self.calc_sq_flow(C,B,BT)
                # K.V=np.sqrt(dV_sq)
                K.F=np.sqrt(F_sq)

            self.print_step(i,Num_steps)

        IO.save_kirchhoff_data(OUTPUT_C,OUTPUT_S,K)
        IO.save_nparray(nullity,'nullity_time')
        IO.save_nparray(dissipation,'dissipation_time')
        IO.save_nparray(volume,'volume_time')

        return OUTPUT_C

    def nsolve_heun_hucai_multisink_radius(self,scale_data,parameters,K,IO):

        Num_steps=scale_data[0]
        dt=scale_data[1]
        sample=scale_data[2]

        gamma=parameters[0]
        g_p1=gamma+1
        g_4m1=4.*(1-gamma)

        self.scale=parameters[1]
        self.vol_diss=parameters[2]
        self.noise=parameters[3]

        OUTPUT_C, OUTPUT_S = IO.init_kirchhoff_data(scale_data,parameters,K)
        nullity =[]
        dissipation=[]
        volume=[]

        #calc network dynamic
        B,BT=K.get_incidence_matrices()

        self.M=nx.number_of_edges(K.G)
        self.N=nx.number_of_nodes(K.G)
        C=K.C[:]

        R=np.power(np.divide(C,self.kappa),0.25)

        # unit/scales
        self.setup_random_fluctuations_multisink(K)

        alpha=(self.scale*K.f**g_p1)/((K.k**gamma)*(K.l**(3*g_p1)))
        beta=(self.vol_diss**K.k*(K.l**3)/K.f)**gamma
        dT=(np.ones(self.M)*dt*alpha)
        sigma=self.sigma/2.
        threshold=10.**(-20)

        for i in range(Num_steps):

            #prediction
            dV_sq, F_sq=self.calc_sq_flow(C,B,BT)

            R_gamma=np.power(R,g_4m1)
            shear_sigma=np.power(np.multiply(dV_sq,R_gamma), sigma)
            diff_shear=np.multiply(np.subtract(shear_sigma,beta),R)
            prediction_dR=np.multiply(diff_shear,dT)
            prediction_R=np.add(R,prediction_dR)

            C=np.multiply(np.power(prediction_R,4.),self.kappa)
            control=np.where( C < threshold )
            for m in control:
                dT[m]=0.
                C[m]=10.**(-21)
            #correction
            dV_sq, F_sq=self.calc_sq_flow(C,B,BT)

            R_gamma=np.power(prediction_R,g_4m1)
            shear_sigma=np.power(np.multiply(dV_sq,R_gamma), sigma)
            diff_shear=np.multiply(np.subtract(shear_sigma,beta),prediction_R)
            correction_dR=np.multiply(diff_shear,dT)

            #update
            dR=0.5*(prediction_dR+correction_dR)
            R=np.add(R,dR)
            K.C=np.multiply(np.power(R,4.),self.kappa)
            control=np.where( K.C < threshold )
            for m in control:
                dT[m]=0.
                K.C[m]=10.**(-21)
            C=K.C[:]

            # output
            if i % sample == 0:
                OUTPUT_C=np.vstack((OUTPUT_C,K.C) )
                OUTPUT_S=np.vstack((OUTPUT_S,K.J) )
                s=0
                K.clipp_graph()
                n=(1.+nx.number_of_edges(K.H)-nx.number_of_nodes(K.H))/(1.+nx.number_of_edges(K.G)-nx.number_of_nodes(K.G))
                nullity.append(n)
                D=np.divide(F_sq,K.C[:])
                dissipation.append(np.sum(D))
                A=np.power(R,2.)
                volume.append(np.sum(A))

                dV_sq, F_sq=self.calc_sq_flow(C,B,BT)
                # K.V=np.sqrt(dV_sq)
                K.F=np.sqrt(F_sq)

            self.print_step(i,Num_steps)

        IO.save_kirchhoff_data(OUTPUT_C,OUTPUT_S,K)
        IO.save_nparray(nullity,'nullity_time')
        IO.save_nparray(dissipation,'dissipation_time')
        IO.save_nparray(volume,'volume_time')

        return OUTPUT_C

    def nsolve_heun_hucai_terminals(self,scale_data,parameters,K,IO):

        Num_steps=scale_data[0]
        dt=scale_data[1]
        sample=scale_data[2]

        gamma=parameters[0]
        g_p1=gamma+1
        g_4m1=4.*(1-gamma)

        c0=parameters[1]
        c1=parameters[2]
        mu=parameters[3]
        var=parameters[4]

        OUTPUT_C, OUTPUT_S = IO.init_kirchhoff_data(scale_data,parameters,K)
        nullity =[]
        dissipation=[]
        volume=[]

        #calc network dynamic
        B,BT=K.get_incidence_matrices()
        # B=ssp.csc_matrix(B)
        # BT=ssp.csc_matrix(BT)
        # print(len(BT.getnnz(1)))

        M=nx.number_of_edges(K.G)
        # N=nx.number_of_nodes(K.G)
        C=K.C[:]

        R=np.power(np.divide(C,self.kappa),0.25)

        # unit/scales
        self.setup_random_fluctuations_reduced(K,mu,var)
        alpha=(c0*K.f**g_p1)/((K.k**gamma)*(K.l**(3*g_p1)))
        beta=(c1**K.k*(K.l**3)/K.f)**gamma
        dT=(np.ones(M)*dt*alpha)
        sigma=self.sigma/2.
        threshold=10.**(-20)

        for i in range(Num_steps):

            #prediction
            dV_sq, F_sq=self.calc_sq_flow(C,B,BT)

            R_gamma=np.power(R,g_4m1)
            shear_sigma=np.power(np.multiply(dV_sq,R_gamma), sigma)
            diff_shear=np.multiply(np.subtract(shear_sigma,beta),R)
            prediction_dR=np.multiply(diff_shear,dT)
            prediction_R=np.add(R,prediction_dR)

            C=np.multiply(np.power(prediction_R,4.),self.kappa)
            control=np.where( C < threshold )
            for m in control:
                dT[m]=0.
                C[m]=10.**(-21)
            #correction
            dV_sq, F_sq=self.calc_sq_flow(C,B,BT)

            R_gamma=np.power(prediction_R,g_4m1)
            shear_sigma=np.power(np.multiply(dV_sq,R_gamma), sigma)
            diff_shear=np.multiply(np.subtract(shear_sigma,beta),prediction_R)
            correction_dR=np.multiply(diff_shear,dT)

            #update
            dR=0.5*(prediction_dR+correction_dR)
            R=np.add(R,dR)
            K.C=np.multiply(np.power(R,4.),self.kappa)
            control=np.where( K.C < threshold )
            for m in control:
                dT[m]=0.
                K.C[m]=10.**(-21)
            C=K.C[:]


            # output
            if i % sample == 0:
                OUTPUT_C=np.vstack((OUTPUT_C,K.C) )
                OUTPUT_S=np.vstack((OUTPUT_S,K.J) )
                s=0
                K.clipp_graph()
                n=(1.+nx.number_of_edges(K.H)-nx.number_of_nodes(K.H))/(1.+nx.number_of_edges(K.G)-nx.number_of_nodes(K.G))
                nullity.append(n)
                D=np.divide(F_sq,K.C[:])
                dissipation.append(np.sum(D))
                A=np.power(R,2.)
                volume.append(np.sum(A))

                dV_sq, F_sq=self.calc_sq_flow(C,B,BT)
                # K.V=np.sqrt(dV_sq)
                K.F=np.sqrt(F_sq)

            self.print_step(i,Num_steps)

        IO.save_kirchhoff_data(OUTPUT_C,OUTPUT_S,K)
        IO.save_nparray(nullity,'nullity_time')
        IO.save_nparray(dissipation,'dissipation_time')
        IO.save_nparray(volume,'volume_time')

class integrate_bilayer(integrate_kirchoff,object):

    def __init__(self):
        self.a=1
        self.exp=2
        self.epsilon=1.
        self.sigma_1=1.
        self.sigma_2=1.
        self.kappa_1=1.
        self.kappa_2=1.

        self.scales=[0,0]
        self.vol_diss=[0,0]
        self.coupling_diss=[0,0]
        self.coupling_exp=[0,0]
        self.noise=[0,0]
        self.local_flow=0.
        self.x=[0,0]
        self.mu=[]
        self.N=[0,0]
        self.M=[0,0]
        self.D={}
        self.indices=[[],[]]

    def init_parameters(self,parameters):
        self.scales=[parameters[0],parameters[1]]
        self.vol_diss=[parameters[2],parameters[3]]
        self.coupling_diss=[parameters[4],parameters[5]]
        self.coupling_exp=[parameters[6],parameters[7]]
        self.noise=[parameters[8],parameters[9]]

    def test_system_contact(self,e_adj,R_try):
        for k,e in enumerate(e_adj):
            test=R_try[0][e[0]]+R_try[1][e[1]]
            if test > 1.:
                print('time_step too large/ unstable parameters: networks in contact')
                sys.exit()

    def test_threshold(self,R,dT,threshold):
        for j in range(2):
            control=np.where( R[j] < threshold )[0]
            for m in control:
                dT[j][m]=0.
                R[j][m]=10.**(-21)

        return R,dT

    def set_edge_directions(self, K):
        B,BT=K.get_incidence_matrices()
        OP=np.dot(B,np.dot(np.diag(K.C),BT))
        inverse=lina.pinv(OP)
        D=np.dot(BT,inverse)

        x=np.where(K.J > 0)[0][0]
        idx=np.where(K.J < 0)[0]
        N=len(K.J)

        for j,e in enumerate(K.G.edges()):
            dp_j=(D[j,x]*(1-N) + np.sum(D[j,idx]))
            K.G.edges[e]['sign']=dp_j/np.absolute(dp_j)

    def calc_radius_diff(self,B,I,R,dT,KAPPA):

        f={}
        dR_aux={}
        R_aux={}
        C={}
        sgn=self.coupling_exp[0]/np.absolute(self.coupling_exp[0])
        for j in range(2):
            f[j]=np.zeros(self.M[j])

        for j,e in enumerate(B.e_adj):
            r=1.-(R[0][e[0]]+R[1][e[1]])
            d0=r**self.coupling_exp[0]
            d1=r**self.coupling_exp[1]
            f[0][e[0]]+=self.coupling_diss[0]*d0*sgn
            f[1][e[1]]+=self.coupling_diss[1]*d1*sgn

        for j in range(2):

            C[j]= np.multiply(KAPPA[j],np.power(R[j],4))
            dV_sq,F_sq=self.calc_sq_flow(j,C[j],I[j][0],I[j][1])

            R_3=np.power(R[j],3)
            shear_sq=np.multiply(dV_sq,R_3)
            vol=np.multiply(R[j],self.vol_diss[j])
            diff_shearvol=np.subtract(shear_sq,vol)
            diff_shearvol_repulsion=np.add(diff_shearvol,f[j])

            dR_aux[j]=np.multiply(diff_shearvol_repulsion,dT[j]*self.scales[j])
            R_aux[j]=np.add(R[j],dR_aux[j])
            self.D[j]=np.divide(F_sq,C[j])

        return R_aux,dR_aux

    def calc_radius_diff_RK(self,B,I,R,dT,KAPPA):
        f={}
        dR_aux={}
        R_aux={}
        C={}
        for j in range(2):
            f[j]=np.zeros(self.M[j])

        for j,e in enumerate(B.e_adj):
            r=1.-(R[0][e[0]]+R[1][e[1]])
            d0=r**self.coupling_exp[0]
            d1=r**self.coupling_exp[1]
            f[0][e[0]]-=self.coupling_diss[0]/d0
            f[1][e[1]]-=self.coupling_diss[1]/d1

        for j in range(2):

            C[j]= np.multiply(KAPPA[j],np.power(R[j],4))
            dV_sq,F_sq=self.calc_sq_flow(j,C[j],I[j][0],I[j][1])

            R_3=np.power(R[j],3)
            shear_sq=np.multiply(dV_sq,R_3)
            vol=np.multiply(R[j],self.vol_diss[j])
            diff_shearvol=np.subtract(shear_sq,vol)
            diff_shearvol_repulsion=np.add(diff_shearvol,f[j])

            dR_aux[j]=np.multiply(diff_shearvol_repulsion,dT[j]*self.scales[j])
            self.D[j]=np.divide(F_sq,C[j])

        return dR_aux

    def update_radius(self,dR,R):
        R[0]=np.add(R[0],dR[0])
        R[1]=np.add(R[1],dR[1])
        return R

    def calc_radius_diff_Mie(self,B,I,R,dT,KAPPA):
        f={}
        dR_aux={}
        R_aux={}
        C={}
        for j in range(2):
            f[j]=np.zeros(self.M[j])

        for j,e in enumerate(B.e_adj):
            r=1.-(R[0][e[0]]+R[1][e[1]])
            d=self.sigma/r

            f[0][e[0]]-= self.coupling_diss[0] * (d**(self.coupling_exp[0]+1) - self.sigma * d**(self.alpha*self.coupling_exp[0]+1))
            f[1][e[1]]-= self.coupling_diss[1] * (d**(self.coupling_exp[1]+1) - self.sigma * d**(self.alpha*self.coupling_exp[1]+1))

        for j in range(2):

            C[j]= np.multiply(KAPPA[j],np.power(R[j],4))
            dV_sq,F_sq=self.calc_sq_flow(j,C[j],I[j][0],I[j][1])

            R_3=np.power(R[j],3)
            shear_sq=np.multiply(dV_sq,R_3)
            vol=np.multiply(R[j],self.vol_diss[j])
            diff_shearvol=np.subtract(shear_sq,vol)
            diff_shearvol_repulsion=np.add(diff_shearvol,f[j])
            # print(self.scales[j])
            dR_aux[j]=np.multiply(diff_shearvol_repulsion,dT[j]*self.scales[j])
            R_aux[j]=np.add(R[j],dR_aux[j])
            self.D[j]=np.divide(F_sq,C[j])

        return R_aux,dR_aux

    def calc_radius_diff_asymmetric(self,B,I,R,dT,KAPPA):
        f={}
        dR_aux={}
        R_aux={}
        C={}
        D={}
        for j in range(2):
            f[j]=np.zeros(self.M[j])

        for j,e in enumerate(B.e_adj):
            r=1.-(R[0][e[0]]+R[1][e[1]])
            d0=r**self.coupling_exp[0]
            d1=r**self.coupling_exp[1]
            f[0][e[0]]-=self.coupling_diss[0]/d0
            f[1][e[1]]-=self.coupling_diss[1]/d1

        for j in range(2):

            C[j]= np.multiply(KAPPA[j],np.power(R[j],4))

            dV_sq,F_sq=self.calc_sq_flow_local(j,C[j],I[j][0],I[j][1])

            R_3=np.power(R[j],3)
            shear_sq=np.multiply(dV_sq,R_3)
            vol=np.multiply(R[j],self.vol_diss[j])
            diff_shearvol=np.subtract(shear_sq,vol)
            diff_shearvol_repulsion=np.add(diff_shearvol,f[j])
            # print(self.scales[j])
            dR_aux[j]=np.multiply(diff_shearvol_repulsion,dT[j]*self.scales[j])
            R_aux[j]=np.add(R[j],dR_aux[j])
            D[j]=np.divide(F_sq,C[j])

        return R_aux,dR_aux,D

    def calc_sq_flow_random(self,j,C,B,BT):

        OP=np.dot(np.dot(B,C),BT)
        MP=lina.pinv(OP)
        D=np.dot(C,np.dot(BT,MP))
        DT=np.transpose(D)

        var_matrix=np.dot(np.dot(D,self.G[j]),DT)
        mean_matrix=np.dot(np.dot(D,self.H[j]),DT)
        var_flow=np.diag(var_matrix)
        mean_flow=np.diag(mean_matrix)

        F_sq= np.add(self.var[j]*var_flow , self.mu[j]*self.mu[j]*mean_flow)

        return F_sq

    def calc_mu_sq(self,R,B):
        self.mu=[]

        for m in range(self.N[0]):
            if m!=self.x[0]:
                vol=0.
                for e in B.n_adj[0][m]:
                    vol+=R[1][e]**2
                self.mu.append(    (1. + self.local_flow * vol)    )

    def calc_sq_flow(self,j,C,B,BT):

        OP=np.dot(B,np.dot(np.diag(C),BT))
        inverse=lina.pinv(OP)
        D=np.dot(BT,inverse)
        DT=np.transpose(D)
        A=np.dot(D,self.Z[j])
        V=np.dot(A,DT)
        dV_sq=np.diag(V)
        F_sq=np.multiply(np.multiply(C,C),dV_sq)

        return dV_sq,F_sq

    def calc_sq_flow_local(self,j,C,B,BT):

        OP=np.dot(np.dot(B,np.diag(C)),BT)
        MP=lina.pinv(OP)
        A=np.dot(BT,MP)
        if j==0:

            N1=len(B[:,0])-1
            A_src=A[:,self.x[j]]
            A_src_sq=np.power(A_src,2)
            A=np.delete(A,self.x[j],1)
            A_mu=np.multiply(A,self.mu) #???
            trace_mu=np.sum(self.mu[:])
            trace_A_mu=np.apply_along_axis(np.sum,1,A_mu)
            trace_A=np.apply_along_axis(np.sum,1,A)
            trace_A_sq=np.apply_along_axis(np.sum,1,np.power(A,2))
            # fluctuation vector
            ADD_SRC_SINK=np.add( N1*A_src_sq, trace_A_sq)
            MULTIPLY_SRC_SINK=2.*np.multiply(A_src,trace_A)
            fluc= np.subtract(ADD_SRC_SINK,MULTIPLY_SRC_SINK)
            # deterministic vector
            ADD_SRC_SINK=np.add( A_src_sq*trace_mu**2 , np.power(trace_A_mu,2))
            MULTIPLY_SRC_SINK=2.*trace_mu*np.multiply(A_src,trace_A_mu)
            det=np.subtract(ADD_SRC_SINK,MULTIPLY_SRC_SINK)

        if j==1:

            N1=len(B[:,0])-1
            N1_sq=N1*N1
            A_src=A[:,self.x[j]]
            A_src_sq=np.power(A_src,2)
            A=np.delete(A,self.x[j],1)
            trace_A=np.apply_along_axis(np.sum,1,A)
            trace_A_sq=np.apply_along_axis(np.sum,1,np.power(A,2))
            # fluctuation vector
            ADD_SRC_SINK=np.add( N1*A_src_sq, trace_A_sq)
            MULTIPLY_SRC_SINK=2.*np.multiply(A_src,trace_A)
            fluc= np.subtract(ADD_SRC_SINK,MULTIPLY_SRC_SINK)
            # deterministic vector
            ADD_SRC_SINK=np.add( N1_sq*A_src_sq, np.power(trace_A,2))
            MULTIPLY_SRC_SINK=2.*N1*np.multiply(A_src,trace_A)
            det=np.subtract(ADD_SRC_SINK,MULTIPLY_SRC_SINK)

        dV_sq= np.add(self.noise[j]  * fluc,det)
        F_sq=np.multiply(np.power(C,2),dV_sq)

        return dV_sq,F_sq

    def setup_random_fluctuations(self,B):

        self.Z=[]
        for j in range(2):
            x=np.where(B.layer[j].J > 0)[0][0]
            N=len(B.layer[j].J)
            # idx=np.where(B.layer[j].J < 0)[0]

            L0=np.ones((N,N))
            L0[x,:]=0.
            L0[:,x]=0.

            L1=np.identity(N)
            L1[x,x]=0.

            L2=np.zeros((N,N))
            L2[x,:]=1.-N
            L2[:,x]=1.-N
            L2[x,x]=(N-1)**2

            f=1+self.noise[j]/(N-1)
            self.Z.append((L0  + self.noise[j] * L1 + f * L2))

    def setup_random_fluctuations_multisink(self,B):

        self.Z=[]
        for k in range(2):

            num_n=nx.number_of_nodes(B.layer[k].G)
            x=np.where(B.layer[k].J > 0)[0]
            idx=np.where(B.layer[k].J < 0)[0]
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

            self.Z.append(np.add(U,np.multiply(self.noise[k],V)))

    def setup_random_fluctuations_local(self,B):

        self.Z=[]

        x=np.where(B.layer[0].J > 0)[0][0]
        self.x[0]=x
        self.indices[0]=[i for i in range(self.N[0]) if i !=x]
        x=np.where(B.layer[1].J > 0)[0][0]
        self.x[1]=x
        self.indices[1]=[i for i in range(self.N[1]) if i !=x]

    def nsolve_heun_hucai_adapting_optimization(self,scale_data,parameters,B,IO):

        K=[B.layer[0],B.layer[1]]
        # input_parameter
        Num_steps=scale_data[0]
        dt=scale_data[1]
        sample=scale_data[2]

        scales=[parameters[0],parameters[1]]
        vol_diss=[parameters[2],parameters[3]]
        coupling_diss=[parameters[4],parameters[5]]
        coupling_exp=[parameters[6],parameters[7]]
        noise=[parameters[8],parameters[9]]
        # output_measurement
        OUTPUT=[]
        nullity =[[],[]]
        branching =[[],[]]
        dissipation=[[],[]]
        volume=[[],[]]
        #scale system
        threshold=10.**(-20)
        M=[]
        N=[]

        I=[]
        KAPPA=np.array([self.kappa_1,self.kappa_2])
        dT=[]
        for j in range(2):
            OC,OS=IO.init_kirchhoff_data(scale_data,parameters,K[j])
            OUTPUT.append([OC,OS])
            # integration dict_parameters
            M.append(nx.number_of_edges(K[j].G))
            N.append(nx.number_of_nodes(K[j].G))
            incidence,incidence_T=K[j].get_incidence_matrices()
            I.append([incidence,incidence_T])
            dT.append(np.ones(M[j])*dt)

        dT=np.array(dT)

        # auxillary containers
        OUTPUT=np.array(OUTPUT)

        dR_pre={}
        dR_post={}
        R_pre={}
        R_try={}
        F_sq={}
        C=np.array([K[0].C[:],K[1].C[:]])
        R=np.array([np.power(np.divide(C[i],KAPPA[i]),0.25) for i,c in enumerate(C)])

        self.setup_random_fluctuations_reduced(B,noise)
        stationary=[False,False]

        i=0
        while not (stationary[0] and stationary[1]):

            # 1) prediction
            R_pre,dR_pre=self.calc_radius_diff(B,I,M,R,dT,KAPPA)

            # 2) correction
            R_post,dR_post=self.calc_radius_diff(B,I,M,R_pre,dT,KAPPA)
            for j in range(2):
                R_try[j]=np.add(R[j],2.*np.add(dR_pre[j],dR_post[j]))
            #check time_step
            time_check=True
            for k,e in enumerate(B.e_adj):
                test=R_try[0][e[0]]+R_try[1][e[1]]
                if test > 1.:
                    dT[0]=np.divide(dT[0],10.)
                    dT[1]=np.divide(dT[1],10.)
                    print('refining time step:'+str(i))

                    time_check=False
                    break
            if not time_check:
                continue
            else:
                i+=1
            #update
            for j in range(2):
                R[j]=R_try[j]
                K[j].C=np.multiply(np.power(R[j],4.),KAPPA[j])
                control=np.where( R[j] < threshold )
                for m in control:
                    dT[j][m]=0.
                    R[j][m]=10.**(-21)

                # measure/output
                if i % sample == 0:

                    OUTPUT[j][0]=np.vstack((OUTPUT[j][0],K[j].C) )
                    OUTPUT[j][1]=np.vstack((OUTPUT[j][1],K[j].J) )
                    # print((1.+nx.number_of_edges(K[j].G)-nx.number_of_nodes(K[j].G)))
                    K[j].clipp_graph()
                    n=(1.+nx.number_of_edges(K[j].H)-nx.number_of_nodes(K[j].H))/(1.+nx.number_of_edges(K[j].G)-nx.number_of_nodes(K[j].G))
                    hist_H=np.array(nx.degree_histogram(nx.Graph(K[j].H)))
                    hist_G=np.array(nx.degree_histogram(nx.Graph(K[j].G)))
                    h=float(np.sum(hist_H[3:]))/float(np.sum(hist_G[:]))

                    # d=np.divide(F_sq[j],K[j].C[j])
                    dissipation[j].append(
                    0.
                    # np.sum(d)
                    )
                    volume[j].append(np.sum(np.power(R[j],2.)))
                    nullity[j].append(n)
                    branching[j].append(h)

                proof_sum=np.sum(np.power(dR_post[j],2.))
                print(proof_sum)
                # print(dT[j])
                if proof_sum < threshold :
                    stationary[j]=True
            # self.print_step(i,Num_steps)
            # if i%sample:
            print('steps:'+str(i))
            if i==Num_steps:
                break
        IO.save_bilayer_kirchhoff_data([OUTPUT[0][0],OUTPUT[1][0]],[OUTPUT[0][1],OUTPUT[1][1]],[K[0],K[1]])
        for i in range(2):
            IO.save_nparray(nullity[i],'nullity_time_'+str(i+1))
            IO.save_nparray(branching[i],'branching_time_'+str(i+1))
            IO.save_nparray(dissipation[i],'dissipation_time_'+str(i+1))
            IO.save_nparray(volume[i],'volume_time_'+str(i+1))

    def nsolve_heun_hucai_fixed_optimization(self,scale_data,parameters,B,IO):

        K=[B.layer[0],B.layer[1]]
        # input_parameter
        Num_steps=scale_data[0]
        dt=scale_data[1]
        sample=scale_data[2]

        self.scales=[parameters[0],parameters[1]]
        self.vol_diss=[parameters[2],parameters[3]]
        self.coupling_diss=[parameters[4],parameters[5]]
        self.coupling_exp=[parameters[6],parameters[7]]
        self.noise=[parameters[8],parameters[9]]
        # output_measurement
        OUTPUT=[]
        nullity =[[],[]]
        branching =[[],[]]
        dissipation=[[],[]]
        volume=[[],[]]
        #scale system
        threshold=10.**(-20)
        M=[]
        N=[]

        I=[]
        KAPPA=np.array([self.kappa_1,self.kappa_2])
        dT=[]
        dT_pre=[]
        for j in range(2):
            OC,OS=IO.init_kirchhoff_data(scale_data,parameters,K[j])
            OUTPUT.append([OC,OS])
            # integration dict_parameters
            # M.append(nx.number_of_edges(K[j].G))
            # N.append(nx.number_of_nodes(K[j].G))
            self.M[j]=nx.number_of_edges(K[j].G)
            self.N[j]=nx.number_of_nodes(K[j].G)
            incidence,incidence_T=K[j].get_incidence_matrices()
            I.append([incidence,incidence_T])
            dT.append(np.ones(self.M[j])*dt)
            dT_pre.append(np.ones(self.M[j])*dt)

        dT=np.array(dT)
        dT_pre=np.array(dT_pre)
        # auxillary containers
        OUTPUT=np.array(OUTPUT)
        dR_pre={}
        dR_post={}
        R_pre={}
        R_try={}
        F_sq={}
        C=np.array([K[0].C[:],K[1].C[:]])
        R=np.array([np.power(np.divide(C[i],KAPPA[i]),0.25) for i,c in enumerate(C)])

        # self.setup_random_fluctuations(B)
        self.setup_random_fluctuations_multisink(B)

        for i in range(Num_steps):

            # 1) prediction
            R_pre,dR_pre=self.calc_radius_diff(B,I,R,dT,KAPPA)

            control=np.where( R_pre[j] < threshold )[0]
            for m in control:
                dT_pre[j][m]=0.
                R_pre[j][m]=10.**(-21)

            # 2) correction
            R_post,dR_post=self.calc_radius_diff(B,I,R_pre,dT_pre,KAPPA)
            for j in range(2):
                dR_aux=0.5*np.add(dR_pre[j],dR_post[j])
                R_try[j]=np.add(R[j],dR_aux)
            #check time_step
            # print(dR_aux)
            for k,e in enumerate(B.e_adj):
                test=R_try[0][e[0]]+R_try[1][e[1]]
                if test > 1.:
                    print('time_step too large/ unstable parameters:'+str(i))
                    sys.exit()

            #update
            for j in range(2):
                R[j]=R_try[j]
                K[j].C=np.multiply(np.power(R[j],4.),KAPPA[j])
                control=np.where( R[j] < threshold )[0]
                for m in control:
                    dT[j][m]=0.
                    R[j][m]=10.**(-21)
                dT_pre[j]=dT[j][:]
                # measure/output
                if i % sample == 0:

                    OUTPUT[j][0]=np.vstack((OUTPUT[j][0],K[j].C) )
                    OUTPUT[j][1]=np.vstack((OUTPUT[j][1],K[j].J) )
                    # print((1.+nx.number_of_edges(K[j].G)-nx.number_of_nodes(K[j].G)))
                    # self.set_edge_directions(K[j])
                    K[j].clipp_graph()
                    H=nx.Graph(K[j].H)
                    n=(nx.number_connected_components(H)+nx.number_of_edges(H)-nx.number_of_nodes(H))/(1.+self.M[j]-self.N[j])
                    hist_H=np.array(nx.degree_histogram(H))
                    hist_G=np.array(nx.degree_histogram(nx.Graph(K[j].G)))
                    h=float(np.sum(hist_H[3:]))/float(np.sum(hist_G[:]))
                    # print(self.D[j])
                    dissipation[j].append( np.sum(self.D[j]) )
                    volume[j].append(np.sum(np.power(R[j],2.)))
                    nullity[j].append(n)
                    branching[j].append(h)

                proof_sum=np.sum(np.power(dR_post[j],2.))
                # print(proof_sum)

            self.print_step(i,Num_steps)

        IO.save_bilayer_kirchhoff_data([OUTPUT[0][0],OUTPUT[1][0]],[OUTPUT[0][1],OUTPUT[1][1]],[K[0],K[1]])
        for i in range(2):
            IO.save_nparray(nullity[i],'nullity_time_'+str(i+1))
            IO.save_nparray(branching[i],'branching_time_'+str(i+1))
            IO.save_nparray(dissipation[i],'dissipation_time_'+str(i+1))
            IO.save_nparray(volume[i],'volume_time_'+str(i+1))

        return OUTPUT

    def nsolve_RK_hucai_optimization(self,scale_data,parameters,B,IO):

        K=[B.layer[0],B.layer[1]]
        # input_parameter
        Num_steps=scale_data[0]
        dt=scale_data[1]
        sample=scale_data[2]
        self.init_parameters(parameters)

        # output_measurement
        OUTPUT=[]
        nullity =[[],[]]

        #scale system
        threshold=10.**(-20)
        I=[]
        KAPPA=np.array([self.kappa_1,self.kappa_2])
        dT=[]
        for j in range(2):
            OC,OS=IO.init_kirchhoff_data(scale_data,parameters,K[j])
            OUTPUT.append([OC,OS])
            # integration dict_parameters
            self.M[j]=nx.number_of_edges(K[j].G)
            self.N[j]=nx.number_of_nodes(K[j].G)
            incidence,incidence_T=K[j].get_incidence_matrices()
            I.append([incidence,incidence_T])
            dT.append(np.ones(self.M[j])*dt)

        dT=np.array(dT)
        # auxillary containers
        OUTPUT=np.array(OUTPUT)
        C=np.array([K[0].C[:],K[1].C[:]])
        R={}
        for j in range(2):
            R[j]=np.array(np.power(np.divide(C[j],KAPPA[j]),0.25) )
        self.setup_random_fluctuations_multisink(B)

        for i in range(Num_steps):

            # 1) k1
            dR_1=self.calc_radius_diff_RK(B,I,R,dT*0.5,KAPPA)

            # 2) k2
            R_1=self.update_radius(dR_1,R)
            dR_2=self.calc_radius_diff_RK(B,I,R_1,dT*0.5,KAPPA)

            # 3) k3
            R_2=self.update_radius(dR_2,R)
            dR_3=self.calc_radius_diff_RK(B,I,R_2,dT,KAPPA)

            # 4) k4
            R_3=self.update_radius(dR_3,R)
            dR_4=self.calc_radius_diff_RK(B,I,R_3,dT,KAPPA)

            #update, check time_step and contact
            dR={}
            dR[0]=np.add(np.add(dR_1[0]/3.,dR_2[0]*(2./3.)),np.add(dR_3[0]*(2./3.),dR_4[0]/6.))
            dR[1]=np.add(np.add(dR_1[1]/3.,dR_2[1]*(2./3.)),np.add(dR_3[1]*(2./3.),dR_4[1]/6.))

            R_new=self.update_radius(dR,R)
            R_new,dT_=self.test_threshold(R_new,dT,threshold)
            self.test_system_contact(B.e_adj,R_new)

            for j in range(2):
                R[j]=R_new[j]
                K[j].C=np.multiply(np.power(R[j],4.),KAPPA[j])
                proof_sum=np.sum(np.power(dR[j],2.))
                # measure/output
                if i % sample == 0:

                    OUTPUT[j][0]=np.vstack((OUTPUT[j][0],K[j].C) )
                    OUTPUT[j][1]=np.vstack((OUTPUT[j][1],K[j].J) )

                    K[j].clipp_graph()
                    H=nx.Graph(K[j].H)
                    n=(nx.number_connected_components(H)+nx.number_of_edges(H)-nx.number_of_nodes(H))/(1.+self.M[j]-self.N[j])
                    hist_H=np.array(nx.degree_histogram(H))
                    hist_G=np.array(nx.degree_histogram(nx.Graph(K[j].G)))
                    h=float(np.sum(hist_H[3:]))/float(np.sum(hist_G[:]))
                    nullity[j].append(n)

            self.print_step(i,Num_steps)

        IO.save_bilayer_kirchhoff_data([OUTPUT[0][0],OUTPUT[1][0]],[OUTPUT[0][1],OUTPUT[1][1]],[K[0],K[1]])
        for i in range(2):
            IO.save_nparray(nullity[i],'nullity_time_'+str(i+1))

        return OUTPUT

    def nsolve_heun_hucai_Mie_optimization(self,scale_data,parameters,B,IO):

        K=[B.layer[0],B.layer[1]]
        # input_parameter
        Num_steps=scale_data[0]
        dt=scale_data[1]
        sample=scale_data[2]

        self.scales=[parameters[0],parameters[1]]
        self.vol_diss=[parameters[2],parameters[3]]

        self.coupling_diss=[parameters[4],parameters[5]]
        self.coupling_exp=[parameters[6],parameters[7]]
        self.noise=[parameters[8],parameters[9]]
        self.alpha=parameters[10]
        self.sigma=parameters[11]

        # output_measurement
        OUTPUT=[]
        nullity =[[],[]]
        branching =[[],[]]
        dissipation=[[],[]]
        volume=[[],[]]
        #scale system
        threshold=10.**(-20)
        M=[]
        N=[]

        I=[]
        KAPPA=np.array([self.kappa_1,self.kappa_2])
        dT=[]
        dT_pre=[]
        for j in range(2):
            OC,OS=IO.init_kirchhoff_data(scale_data,parameters,K[j])
            OUTPUT.append([OC,OS])
            # integration dict_parameters
            # M.append(nx.number_of_edges(K[j].G))
            # N.append(nx.number_of_nodes(K[j].G))
            self.M[j]=nx.number_of_edges(K[j].G)
            self.N[j]=nx.number_of_nodes(K[j].G)
            incidence,incidence_T=K[j].get_incidence_matrices()
            I.append([incidence,incidence_T])
            dT.append(np.ones(self.M[j])*dt)
            dT_pre.append(np.ones(self.M[j])*dt)

        dT=np.array(dT)
        dT_pre=np.array(dT_pre)
        # auxillary containers
        OUTPUT=np.array(OUTPUT)
        dR_pre={}
        dR_post={}
        R_pre={}
        R_try={}
        F_sq={}
        C=np.array([K[0].C[:],K[1].C[:]])
        R=np.array([np.power(np.divide(C[i],KAPPA[i]),0.25) for i,c in enumerate(C)])

        self.setup_random_fluctuations_multisink(B)

        for i in range(Num_steps):

            # 1) prediction
            R_pre,dR_pre=self.calc_radius_diff_Mie(B,I,R,dT,KAPPA)

            control=np.where( R_pre[j] < threshold )[0]
            for m in control:
                dT_pre[j][m]=0.
                R_pre[j][m]=10.**(-21)

            # 2) correction
            R_post,dR_post=self.calc_radius_diff_Mie(B,I,R_pre,dT_pre,KAPPA)
            for j in range(2):
                dR_aux=0.5*np.add(dR_pre[j],dR_post[j])
                R_try[j]=np.add(R[j],dR_aux)
            #check time_step
            # print(dR_aux)
            for k,e in enumerate(B.e_adj):
                test=R_try[0][e[0]]+R_try[1][e[1]]
                if test > 1.:
                    print('time_step too large/ unstable parameters:'+str(i))
                    sys.exit()

            #update
            for j in range(2):
                R[j]=R_try[j]
                K[j].C=np.multiply(np.power(R[j],4.),KAPPA[j])
                control=np.where( R[j] < threshold )[0]
                for m in control:
                    dT[j][m]=0.
                    R[j][m]=10.**(-21)
                dT_pre[j]=dT[j][:]
                # measure/output
                if i % sample == 0:

                    OUTPUT[j][0]=np.vstack((OUTPUT[j][0],K[j].C) )
                    OUTPUT[j][1]=np.vstack((OUTPUT[j][1],K[j].J) )
                    # print((1.+nx.number_of_edges(K[j].G)-nx.number_of_nodes(K[j].G)))
                    # self.set_edge_directions(K[j])
                    K[j].clipp_graph()
                    H=nx.Graph(K[j].H)
                    n=(nx.number_connected_components(H)+nx.number_of_edges(H)-nx.number_of_nodes(H))/(1.+self.M[j]-self.N[j])
                    hist_H=np.array(nx.degree_histogram(H))
                    hist_G=np.array(nx.degree_histogram(nx.Graph(K[j].G)))
                    h=float(np.sum(hist_H[3:]))/float(np.sum(hist_G[:]))
                    # print(self.D[j])
                    dissipation[j].append( np.sum(self.D[j]) )
                    volume[j].append(np.sum(np.power(R[j],2.)))
                    nullity[j].append(n)
                    branching[j].append(h)

                proof_sum=np.sum(np.power(dR_post[j],2.))
                # print(proof_sum)

            self.print_step(i,Num_steps)

        IO.save_bilayer_kirchhoff_data([OUTPUT[0][0],OUTPUT[1][0]],[OUTPUT[0][1],OUTPUT[1][1]],[K[0],K[1]])
        for i in range(2):
            IO.save_nparray(nullity[i],'nullity_time_'+str(i+1))
            IO.save_nparray(branching[i],'branching_time_'+str(i+1))
            IO.save_nparray(dissipation[i],'dissipation_time_'+str(i+1))
            IO.save_nparray(volume[i],'volume_time_'+str(i+1))

        return OUTPUT

    def nsolve_heun_hucai_fixed_optimization_local(self,scale_data,parameters,B,IO):

        K=[B.layer[0],B.layer[1]]
        # input_parameter
        Num_steps=scale_data[0]
        dt=scale_data[1]
        sample=scale_data[2]

        self.scales=[parameters[0],parameters[1]]
        self.vol_diss=[parameters[2],parameters[3]]
        self.coupling_diss=[parameters[4],parameters[5]]
        self.coupling_exp=[parameters[6],parameters[7]]
        self.noise=[parameters[8],parameters[9]]
        self.local_flow=parameters[10]
        # output_measurement
        OUTPUT=[]
        nullity =[[],[]]
        branching =[[],[]]
        dissipation=[[],[]]
        volume=[[],[]]
        proof_sum=[[],[]]
        #scale system
        threshold=10.**(-20)
        # M=[]
        # N=[]

        I=[]
        KAPPA=np.array([self.kappa_1,self.kappa_2])
        dT=[]
        dT_pre=[]
        for j in range(2):
            OC,OS=IO.init_kirchhoff_data(scale_data,parameters,K[j])
            OUTPUT.append([OC,OS])
            # integration dict_parameters
            # M.append(nx.number_of_edges(K[j].G))
            self.M[j]=nx.number_of_edges(K[j].G)
            # N.append(nx.number_of_nodes(K[j].G))
            self.N[j]=nx.number_of_nodes(K[j].G)
            incidence,incidence_T=K[j].get_incidence_matrices()
            I.append([incidence,incidence_T])
            dT.append(np.ones(self.M[j])*dt)
            dT_pre.append(np.ones(self.M[j])*dt)

        dT=np.array(dT)
        dT_pre=np.array(dT_pre)
        # auxillary containers
        OUTPUT=np.array(OUTPUT)
        dR_pre={}
        dR_post={}
        R_pre={}
        R_try={}
        F_sq={}
        C=np.array([K[0].C[:],K[1].C[:]])
        R=np.array([np.power(np.divide(C[i],KAPPA[i]),0.25) for i,c in enumerate(C)])

        self.setup_random_fluctuations_local(B)

        for i in range(Num_steps):

            # 1) prediction
            self.calc_mu_sq(R,B)
            R_pre,dR_pre,D=self.calc_radius_diff_asymmetric(B,I,R,dT,KAPPA)
            # R_pre,dR_pre=self.calc_radius_diff(B,I,R,dT,KAPPA)

            control=np.where( R_pre[j] < threshold )[0]
            for m in control:
                dT_pre[j][m]=0.
                R_pre[j][m]=10.**(-21)

            # 2) correction
            self.calc_mu_sq(R_pre,B)
            R_post,dR_post,D=self.calc_radius_diff_asymmetric(B,I,R_pre,dT_pre,KAPPA)
            # R_post,dR_post=self.calc_radius_diff(B,I,R_pre,dT_pre,KAPPA)
            for j in range(2):
                dR_aux=0.5*np.add(dR_pre[j],dR_post[j])
                R_try[j]=np.add(R[j],dR_aux)

            #update
            for j in range(2):
                R[j]=R_try[j]
                K[j].C=np.multiply(np.power(R[j],4.),KAPPA[j])
                control=np.where( R[j] < threshold )[0]
                for m in control:
                    dT[j][m]=0.
                    R[j][m]=10.**(-21)
                dT_pre[j]=dT[j][:]
                # measure/output
                if i % sample == 0:

                    OUTPUT[j][0]=np.vstack((OUTPUT[j][0],K[j].C) )
                    OUTPUT[j][1]=np.vstack((OUTPUT[j][1],K[j].J) )

                    K[j].clipp_graph()
                    H=nx.Graph(K[j].H)
                    n=(nx.number_connected_components(H)+nx.number_of_edges(H)-nx.number_of_nodes(H))/(1.+self.M[j]-self.N[j])
                    hist_H=np.array(nx.degree_histogram(H))
                    hist_G=np.array(nx.degree_histogram(nx.Graph(K[j].G)))
                    h=float(np.sum(hist_H[3:]))/float(np.sum(hist_G[:]))

                    # dissipation[j].append( D[j])
                    dissipation[j].append(0 )
                    volume[j].append(np.sum(np.power(R[j],2.)))
                    nullity[j].append(n)
                    branching[j].append(h)

                    proof_sum[j].append(np.sum(np.power(dR_post[j],2.)))

            self.print_step(i,Num_steps)

        IO.save_bilayer_kirchhoff_data([OUTPUT[0][0],OUTPUT[1][0]],[OUTPUT[0][1],OUTPUT[1][1]],[K[0],K[1]])

        for i in range(2):
            IO.save_nparray(nullity[i],'nullity_time_'+str(i+1))
            IO.save_nparray(branching[i],'branching_time_'+str(i+1))
            IO.save_nparray(dissipation[i],'dissipation_time_'+str(i+1))
            IO.save_nparray(volume[i],'volume_time_'+str(i+1))
            IO.save_nparray(proof_sum[i],'proof_sum_time_'+str(i+1))

class integrate_surface_mechanics(integrate_kirchoff,object):
    def find_root(self,G):
        for n in G.nodes():
            if G.nodes[n]['source']>0:
                return n
        return 0
    def calc_peclet_PE(self,Q,R_sq,L,D):
        V=np.divide(Q,np.pi*R_sq)
        return np.absolute(np.multiply(V,L)/D)
    def calc_surface_transport_S(self,Q,R_sq,L,D,gamma):

        V=np.absolute(np.divide(Q,np.pi*R_sq))
        R=np.sqrt(R_sq)
        S=gamma*D*np.divide(L,np.multiply(R,V))
        alpha=gamma*L
        return alpha,S
    def calc_uptake_rate_beta(self,alpha,PE,S):
        A1=48.*np.ones(len(PE))
        A2=np.power(np.divide(alpha,S),2)
        A=np.divide(PE,np.add(A1,A2))

        B1=np.divide(S,PE)
        B2=np.divide(np.power(alpha,2),np.multiply(PE,S)*6.)
        B=np.sqrt(np.add(np.ones(len(PE)),np.add(B1,B2)))

        beta=np.multiply(A,np.subtract(B,np.ones(len(PE))))
        return beta
    def calc_flux_orientations(self,J,B,Q):
        G=nx.Graph(J.G)
        dict_incoming,dict_outcoming={},{}
        BQ=np.zeros((len(B[:,0]),len(B[0,:])))
        for n in G.nodes():
            idx_n=G.nodes[n]['label']
            b=B[idx_n,:]
            BQ[idx_n,:]=np.multiply(b,Q)
            dict_incoming[n]=[]
            dict_outcoming[n]=[]
        for n in G.nodes():
            idx_n=G.nodes[n]['label']
            E=G.edges(n)

            for e in E:
                idx_e=G.edges[e]['label']
                if BQ[idx_n,idx_e]>0:
                    dict_outcoming[n].append(idx_e)
                if BQ[idx_n,idx_e]<0:
                    dict_incoming[n].append(idx_e)
        return dict_incoming,dict_outcoming,BQ
    def calc_nodal_concentrations(self,J,B,Q,PE,beta,c0):
        G=nx.Graph(J.G)
        N=len(G.nodes)
        M=len(G.edges)
        c=np.ones(N)*(-1)
        F=np.ones(M)
        n=self.find_root(G)
        n_idx=G.nodes[n]['label']
        c[n_idx]=c0
        nodes_left_undetermined=True

        master_list=[n_idx]
        E=G.edges(n)
        push_list_nodes=[]
        dict_incoming,dict_outcoming,BQ=self.calc_flux_orientations(J,B,Q)
        dict_fluxes={}
        for n in G.nodes():
            dict_fluxes[n]=[]
        for e in E:
            idx_e=G.edges[e]['label']
            if BQ[n_idx,idx_e]>0:
                idx_e=G.edges[e]['label']
                F[idx_e]=np.absolute(Q[idx_e])*c0*np.exp(-beta[idx_e])*(1.+beta[idx_e]/PE[idx_e])

                for n in e:
                    idx_n=G.nodes[n]['label']
                    if idx_n not in master_list:
                        push_list_nodes.append(idx_n)
                        dict_fluxes[n].append(idx_e)

        while(nodes_left_undetermined):
            push_list_cache=[]
            for n in push_list_nodes:
                idx_n=G.nodes[n]['label']
                if (sorted(dict_fluxes[n]) == sorted(dict_incoming[n])):
                    if len(dict_outcoming[n])!=0:
                        X=np.add(np.ones(len(dict_outcoming[n])),np.divide(beta[dict_outcoming[n]],PE[dict_outcoming[n]]))
                        c[idx_n]=np.divide(np.sum(F[dict_incoming[n]]),np.sum(np.multiply(BQ[idx_n,dict_outcoming[n]],X)))

                        master_list.append(idx_n)
                        for idx_e in dict_outcoming[n]:
                            dict_fluxes[n].append(idx_e)
                            F[idx_e]=np.absolute(Q[idx_e])*c[idx_n]*np.exp(-beta[idx_e])*(1.+beta[idx_e]/PE[idx_e])

                        E=G.edges(n)
                        for e in E:
                            idx_e=G.edges[e]['label']
                            for m in e:
                                idx_n=G.nodes[m]['label']
                                if (idx_n not in master_list) :
                                    dict_fluxes[m].append(idx_e)
                                    if (idx_n not in push_list_cache) :
                                        push_list_cache.append(idx_n)
                    else:
                        master_list.append(idx_n)

                else:
                    push_list_cache.append(n)

            push_list_nodes=push_list_cache

            if len(master_list)==N:
                nodes_left_undetermined=False

        return c
    def calc_flows_pressures(self,B,BT,C,S):
        OP=np.dot(B,np.dot(np.diag(C),BT))
        P,RES,RG,si=np.linalg.lstsq(OP,S,rcond=None)
        Q=np.dot(np.diag(C),np.dot(BT,P))
        return Q,P

class integrate_surface_absorption(integrate_kirchoff,object):

    #preaparing local variables and saveing fucntions
    def prepare_it(self, K):
        B,BT=K.get_incidence_matrices()
        L=np.ones(len(B[0,:]))*K.l
        return B,BT,L

    def save_it(self,t_span,nsol,K,abs_par,IO):

        K.C=np.power(nsol[-1,:],4)
        K.set_network_attributes()
        PHI,CON,BETA,PE_,S_,Q,J,DISS,VOL=self.evaluate_timeline(t_span,nsol[:,:],K,abs_par)

        IO.save_nparray(nsol,'radius_time')
        IO.save_nparray(PHI,'phi_time')
        IO.save_nparray(CON,'concentration_time')
        IO.save_nparray(DISS,'dissipation_time')
        IO.save_nparray(VOL,'volume_time')

    # dynamic systems
    def calc_absorption_flux(self,R,network_par,abs_par,K):

        B,BT,L=network_par
        diff_R,diff_S,PHI0,c0=abs_par[:4]
        K.C=np.power(R,4)
        K.set_network_attributes()
        Q,dP,P=cf.calc_flows_pressures(B,BT,K.C,K.J)

        phi,c,F,beta,PE,S,alpha,R_sq,dict_incoming,dict_outcoming,dict_mem_n,dict_mem_nodes,dict_mem_e,dict_incoming_noroot,dict_incoming_noroot_e=cf.calc_flux_parameters([B,BT,L,Q],[diff_R,diff_S,c0],K)
        flux_par=[B,BT,L,Q,F,K.C,R_sq,np.sqrt(R_sq),c]
        abs_par=[PE,S,alpha,beta,diff_R]
        dicts_par=[dict_incoming,dict_outcoming,dict_mem_n,dict_mem_nodes,dict_mem_e,dict_incoming_noroot,dict_incoming_noroot_e]
        J_Phi=cf.calc_absorption_jacobian(flux_par,abs_par,dicts_par,nx.Graph(K.G))

        return phi,J_Phi,dP

    def dR_absorption_dissipation(self,R,t,network_par,abs_par,K):

        diff_R,diff_S,PHI0,c0,beta_1,beta_2=abs_par
        B,BT,L=network_par
        phi,J_Phi,dP=self.calc_absorption_flux(R,network_par,abs_par,K)
        dr=np.zeros(len(K.C))
        I=np.ones(len(K.C))
        for i in range(len(dr)):
            dr[i]=-2.*(np.sum(np.multiply(np.subtract(phi,PHI0),J_Phi[i,:])))
        dr=np.add( dr,np.multiply(beta_2*np.subtract(beta_1*np.power(np.multiply(R,dP),2),I),R) )
        dr=np.multiply(dr,R)
        return dr

    def dR_absorption(self,R,t,network_par,abs_par,K):

        diff_R,diff_S,PHI0,c0=abs_par
        B,BT,L=network_par
        phi,J_Phi,dP=self.calc_absorption_flux(R,network_par,abs_par,K)
        dr=np.zeros(len(K.C))
        for i in range(len(dr)):
            dr[i]=-2.*(np.sum(np.multiply(np.subtract(phi,PHI0),J_Phi[i,:])))
        dr=np.multiply(dr,R)
        return dr

    def evaluate_timeline(self,t_span,nsol,K,abs_par):

        diff_R,diff_S,PHI0,c0=abs_par[:4]
        B,BT,L=self.prepare_it(K)

        I=np.ones(len(B[0,:]))
        PHI=[]
        CON=[]
        BETA=[]
        J_PHI=[]
        DR=[]
        S_=[]
        PE_=[]
        q=[]
        f=[]
        diss=[]
        vol=[]
        for i,t in enumerate(t_span):
            K.C=np.power(nsol[i,:],4)
            K.set_network_attributes()
            Q,dP,P=cf.calc_flows_pressures(B,BT,K.C,K.J)
            phi,c,F,beta,PE,S,alpha,R_sq,dict_incoming,dict_outcoming,dict_mem_n,dict_mem_nodes,dict_mem_e,dict_incoming_noroot,dict_incoming_noroot_e=cf.calc_flux_parameters([B,BT,L,Q],[diff_R,diff_S,c0],K)
            DISS=np.sum(np.divide(np.power(Q,2),K.C))
            VOL=np.sum( np.power(nsol[i,:],2) )

            diss.append(DISS)
            vol.append(VOL)
            PHI.append(phi)
            CON.append(c)
            BETA.append(beta)
            PE_.append(PE)
            S_.append(S)
            q.append(Q)
            f.append(np.divide(phi,np.multiply(F,np.exp(beta))))

        return np.array(PHI),np.array(CON),np.array(BETA),np.array(PE_),np.array(S_),np.array(q),np.array(f),np.array(diss),np.array(vol)

    #integration routines
    def integrate_absorption(self,t_span,B,BT,diff_R,diff_S,K,PHI0):

        I=np.ones(len(K.C))
        L=np.ones(len(B[0,:]))*K.l
        R=np.zeros((len(t_span)+1,len(I)))
        R[0,:]=np.power(K.C,0.25)
        dt=t_span[1]-t_span[0]
        dr_pre=np.zeros(len(I))
        dr_post=np.zeros(len(I))


        for i,t in enumerate(t_span):

            K.C=np.power(R[i,:],4)
            Q,dP,P=calc_flows_pressures(B,BT,K.C,K.J)
            phi,c,F,beta,PE,S,alpha,R_sq,dict_incoming,dict_outcoming,dict_mem_n,dict_mem_nodes,dict_mem_e,dict_incoming_noroot,dict_incoming_noroot_e=calc_flux_parameters([B,BT,L,Q],[diff_R,diff_S],K)
            flux_par=[B,BT,L,Q,F,K.C,R_sq,np.sqrt(R_sq),c]
            abs_par=[PE,S,alpha,beta,diff_R]
            dicts_par=[dict_incoming,dict_outcoming,dict_mem_n,dict_mem_nodes,dict_mem_e,dict_incoming_noroot,dict_incoming_noroot_e]
            J_Phi=calc_absorption_jacobian(flux_par,abs_par,dicts_par,nx.Graph(K.G))

            for j in range(len(I)):
                dr_pre[j]=-np.sum(np.multiply(np.subtract(phi,PHI0),J_Phi[j,:]))
            R_aux=np.add( R[i,:],dr_pre*dt )

            K.C=np.power(R_aux,4)
            Q,dP,P=calc_flows_pressures(B,BT,K.C,K.J)
            phi,c,F,beta,PE,S,alpha,R_sq,dict_incoming,dict_outcoming,dict_mem_n,dict_mem_nodes,dict_mem_e,dict_incoming_noroot,dict_incoming_noroot_e=calc_flux_parameters([B,BT,L,Q],[diff_R,diff_S],K)
            flux_par=[B,BT,L,Q,F,K.C,R_sq,np.sqrt(R_sq),c]
            abs_par=[PE,S,alpha,beta,diff_R]
            dicts_par=[dict_incoming,dict_outcoming,dict_mem_n,dict_mem_nodes,dict_mem_e,dict_incoming_noroot,dict_incoming_noroot_e]
            J_Phi=calc_absorption_jacobian(flux_par,abs_par,dicts_par,nx.Graph(K.G))

            for j in range(len(I)):
                dr_post[j]=-np.sum(np.multiply(np.subtract(phi,PHI0),J_Phi[j,:]))

            R[i+1,:]=np.add( R[i,:],np.add(dr_pre,dr_post)*0.5*dt )

        return R[1:,:]

    def nsolve_absorption(self,K,t_span,abs_par,IO):

        B,BT,L=self.prepare_it(K)
        network_par=[B,BT,L]
        nsol=si.odeint(self.dR_absorption,np.power(K.C[:],0.25),t_span,args=(network_par,abs_par,K),rtol=1e-10)
        self.save_it(t_span,nsol,K,abs_par,IO)

        return nsol

    def nsolve_absorption_dissipation(self,K,t_span,abs_par,IO):

        B,BT,L=self.prepare_it(K)
        network_par=[B,BT,L]
        nsol=si.odeint(self.dR_absorption_dissipation,np.power(K.C[:],0.25),t_span,args=(network_par,abs_par,K),rtol=1e-10)
        self.save_it(t_span,nsol,K,abs_par,IO)

        return nsol
