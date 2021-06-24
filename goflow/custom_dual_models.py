# @Author:  Felix Kramer
# @Date:   2021-06-23T22:33:51+02:00
# @Email:  kramer@mpi-cbg.de
# @Project: go-with-the-flow
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-06-24T10:05:45+02:00
# @License: MIT



class kramer_modes():
  pass
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
