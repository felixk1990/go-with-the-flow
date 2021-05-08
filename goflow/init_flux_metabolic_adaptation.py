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
import init_flux
import line_profiler
import atexit
profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats)


class MySteps(object):
    def __init__(self, stepsize ):
        self.stepsize = stepsize
    def __call__(self, x):
        # print(x)
        rx=np.add(x,np.random.rand(len(x))*self.stepsize)
        return rx

class metabolic_adaptation(init_flux.simple_flux_uptake_network_OFH, object):

    def __init__(self):
        super(metabolic_adaptation,self).__init__()

    # calc and optimize networ costs with established sovers
    def calc_absorption_cost(self, R,*args):

        K=args[0]
        K.C=np.power(R[:],4)*K.k
        K.R=R[:]
        c,B_new,K=self,calc_profile_concentration(K)

        phi=self.calc_absorption(R, dicts[0],K)
        J_phi=self.calc_absorption_jacobian(R, B_new,K)

        phi0=np.ones(m)*K.phi0
        F=np.sum(np.power(np.subtract(phi,phi0),2))
        DF=np.array( [ 2.*np.sum( np.multiply( np.subtract(phi,phi0), J_phi[j,:] ) ) for j in range(m) ] )
        # print(F)
        return F,DF
        # return F

    def calc_absorption_dissipation_cost(self, R,*args):

        K=args[0]
        m=len(K.R)
        K.C=np.power(R[:],4)*K.k
        c,B_new,S,K=self.calc_profile_concentration(K)
        Q,dP,P=self.calc_flows_pressures(K)

        phi=self.calc_absorption(R, dicts[0],K)
        J_phi=self.calc_absorption_jacobian(R, B_new,K)

        phi0=np.ones(self.M)*K.phi0
        F=np.sum(np.power(np.subtract(phi,phi0),2))+ K.alpha_0*np.sum(np.multiply(np.power(dP,2),np.power(R,4)))
        DF=np.array( [ 2.*np.sum( np.multiply( np.subtract(phi,phi0), J_phi[j,:] ) ) for j in range(self.M) ] )
        DF=np.subtract(DF,4.*K.alpha_0*np.multiply(np.power(dP,2),np.power(R,3)))

        return F,DF

    def calc_absorption_volume_cost(self, R,*args):

        K=args[0]
        m=len(K.R)
        K.C=np.power(R[:],4)*K.k
        c,B_new,S,K=self.calc_profile_concentration(K)

        phi=self.calc_absorption(R, dicts[0],K)
        J_phi=self.calc_absorption_jacobian(R, B_new,K)

        phi0=np.ones(m)*K.phi0
        F=np.sum(np.power(np.subtract(phi,phi0),2))+K.alpha*np.sum(np.power(R,2))
        DF=np.array( [ 2.*np.sum( np.multiply( np.subtract(phi,phi0), J_phi[j,:] ) ) for j in range(m) ] )
        DF=np.add(DF,K.alpha*R)

        return F,DF

    def calc_absorption_dissipation_volume_cost(self, R,*args):

        K=args[0]
        K.C=np.power(R[:],4)*K.k
        K.R=R[:]
        c,B_new,S,K=self.calc_profile_concentration(K)

        phi=self.calc_absorption(R,K)
        J_phi=self.calc_absorption_jacobian(R, B_new,K)
        phi0=np.ones(self.M)*K.phi0

        Q,dP,P=self.calc_flows_pressures(K)
        sq_R=np.power(R,2)
        F=np.sum(np.power(np.subtract(phi,phi0),2)) + K.alpha_1*np.sum( np.add( np.multiply(np.power(dP,2),np.power(R,4)), K.alpha_0*sq_R ) )
        DF1=np.array( [ 2.*np.sum( np.multiply( np.subtract(phi,phi0), J_phi[j,:] ) ) for j in range(m) ] )
        DF2=2.*K.alpha_1*np.multiply( np.subtract( np.ones(m)*K.alpha_0 ,2.*np.multiply(np.power(dP,2),sq_R) ),R )

        return F,np.add(DF1,DF2)
        # return F

    def calc_dissipation_volume_cost(self, R,*args):

        K=args[0]
        K.C=np.power(R[:],4)*K.k
        Q,dP,P=self.calc_flows_pressures(K)

        sq_R=np.power(R,2)
        F = np.sum( np.add( K.alpha_1*np.multiply( np.power(dP,2),np.power(R,4)), sq_R*K.alpha_0 ))
        DF=2.*np.subtract( np.ones(m)*K.alpha_0, 2.*K.alpha_1*np.multiply( np.power(dP,2),sq_R ) )
        DF=np.multiply(DF,R)

        return F,DF

    def optimize_network_targets(self, K,mode):

        m=nx.number_of_edges(K.G)
        b0=1e-25
        mysteps=MySteps(1.)

        # sol=sc.basinhopping(calc_absorption_cost,np.ones(m),niter=100,T=1.,take_step=mysteps,minimizer_kwargs={'method':'L-BFGS-B', 'bounds':[(b0,None) for x in range(m)],'args':(K),'jac':True})
        # c,Q,dicts,B_new,S,K=calc_profile_concentration(K)
        if mode=='uptake' :
            sol=sc.basinhopping(self.calc_absorption_cost,K.R,niter=10,T=10.,take_step=mysteps,minimizer_kwargs={'method':'L-BFGS-B', 'bounds':[(b0,None) for x in range(m)],'args':(K),'jac':True,'tol':1e-10})
        if mode=='uptake+volume' :
            sol=sc.minimize(self.calc_absorption_volume_cost,K.R, method='L-BFGS-B', bounds=[(b0,None) for x in range(m)],args=(K),jac=True)
        if mode=='uptake+dissipation' :
            sol=sc.minimize(self.calc_absorption_dissipation_cost,K.R, method='L-BFGS-B', bounds=[(b0,None) for x in range(m)],args=(K),jac=True,tol=1e-10)
            # sol=sc.basinhopping(calc_absorption_dissipation_cost,K.R,niter=100,T=1.,take_step=mysteps,minimizer_kwargs={'method':'L-BFGS-B', 'bounds':[(b0,None) for x in range(m)],'args':(K),'jac':True})
        if mode=='uptake+dissipation+volume' :
            sol=sc.basinhopping(self.calc_absorption_dissipation_volume_cost,K.R,niter=100,T=10.,take_step=mysteps,minimizer_kwargs={'method':'L-BFGS-B', 'bounds':[(b0,None) for x in range(m)],'args':(K),'jac':True,'tol':1e-10})
            # sol=sc.minimize(calc_absorption_dissipation_volume_cost,K.R, method='L-BFGS-B', bounds=[(b0,None) for x in range(m)],args=(K),jac=True,tol=1e-10)
            # sol=sc.minimize(calc_absorption_dissipation_volume_cost,K.R, method='L-BFGS-B', bounds=[(b0,None) for x in range(m)],args=(K))
        if mode=='dissipation+volume' :
            sol=sc.basinhopping(self.calc_dissipation_volume_cost,K.R,niter=10,T=10.,take_step=mysteps,minimizer_kwargs={'method':'L-BFGS-B', 'bounds':[(b0,None) for x in range(m)],'args':(K),'jac':True,'tol':1e-10})
            # sol=sc.minimize(calc_dissipation_volume_cost,np.ones(m), method='L-BFGS-B', bounds=[(b0,None) for x in range(m)],args=(K),jac=True,tol=1e-10)
        # sol=sc.minimize(calc_absorption_cost,np.ones(m),method='L-BFGS-B', bounds=[(b0,None) for x in range(m)],args=(K))

        return sol

    # calc the incerements for various update stimuli
    def calc_absorption_shear_dr(self,K):

        dr=np.zeros(len(K.R))
        c,B_new,K=self.calc_profile_concentration(K)
        phi=self.calc_absorption(K.R, K)

        J_phi=self.calc_absorption_jacobian( K.R,K )

        Q,dP,X=self.calc_flows_pressures(K)
        phi0=np.ones(self.M)*K.phi0

        for i in range(self.M):
            dr[i]=-2.*np.sum( np.multiply(np.subtract(phi,phi0),J_phi[i,:]))

        # DR=np.multiply(np.subtract(2.*K.alpha_1*np.power(np.multiply(dP,K.R),2),K.alpha_0*np.ones(len(phi))), K.R)
        DR=self.calc_shear_dr(K)
        dr=np.add(dr,2.*DR)

        return dr,K.PE

    def calc_shear_absorption_dr(self,K):

        dr=np.zeros(len(K.R))
        c,B_new,K=self.calc_profile_concentration(K)
        phi=self.calc_absorption(K.R, K)
        J_phi=self.calc_absorption_jacobian( K.R,K )

        Q,dP,X=self.calc_flows_pressures(K)
        phi0=np.ones(self.M)*K.phi0

        for i in range(self.M):
            dr[i]=-2.*K.alpha_0*np.sum( np.multiply(np.subtract(phi,phi0),J_phi[i,:]))

        DR=np.multiply(2.*np.subtract(np.power(np.multiply(dP,K.R),2)*K.k, K.alpha_1*np.ones(len(phi))*np.sqrt(K.k)), K.R)
        dr=np.add(dr,2.*DR)

        return dr, K.PE

    def calc_shear_fluctuation_dr(self,K):

        dr=np.zeros(len(K.R))
        # shear_sq,dV_sq, F_sq, avg_phi=self.calc_sq_flow_broken_link(K)
        # shear_sq,dV_sq, F_sq=self.calc_sq_flow_broken_link(K)
        diss,dV_sq,F_sq,R = self.calc_sq_flow_broken_link(K)
        K.dV_sq=dV_sq[:]
        # DR=np.multiply(np.subtract(2.*K.alpha_1*shear_sq,K.alpha_0*np.ones(len(K.R))), K.R)
        DR=np.subtract(2.*K.alpha_1*diss,K.alpha_0*R)
        dr=np.add(dr,2.*DR)

        # return dr,shear_sq,dV_sq,F_sq,avg_phi
        # return dr,shear_sq,dV_sq,F_sq
        return dr,diss,dV_sq,F_sq

    def calc_absorption_dr(self,K):

        dr=np.zeros(self.M)
        c,B_new,K=self.calc_profile_concentration(K)

        phi=self.calc_absorption(K.R, K)
        J_phi=self.calc_absorption_jacobian(K.R, K)
        phi0=np.ones(self.M)*K.phi0

        for i in range(self.M):
            dr[i]=-2.*np.sum( np.multiply(np.subtract(phi,phi0),J_phi[i,:]))

        return dr,K.PE

    def calc_absorption_volumetric_dr(self,K):

        dr=np.zeros(len(K.R))
        ones=np.ones(len(K.dict_volumes.values()))
        c,B_new,K=self.calc_profile_concentration(K)

        phi=self.calc_absorption(K.R, K)
        J_phi=self.calc_absorption_jacobian(K.R, K)
        phi0=ones*K.phi0
        dphi=ones

        for i,v in enumerate(K.dict_volumes.keys()):
            dphi[i]=np.sum(phi[K.dict_volumes[v]])-phi0[i]

        for j,e in enumerate(self.list_e):
            for i,v in enumerate(K.dict_volumes.keys()):
                dr[j]-=2.*dphi[i]*np.sum(J_phi[j,K.dict_volumes[v]])

        return dr

    def calc_absorption_volumetric_shear_dr(self,K):

        DR1=self.calc_shear_dr(K)
        DR2=self.calc_absorption_volumetric_dr(K)
        # ones=np.ones(len(K.dict_volumes.values()))
        # c,B_new,K=self.calc_profile_concentration(K)
        #
        # phi=self.calc_absorption(K.R,K)
        # J_phi=self.calc_absorption_jacobian(K.R,K)
        # phi0=ones*K.phi0
        # dphi=ones
        #
        # for i,v in enumerate(K.dict_volumes.keys()):
        #     dphi[v]=np.sum(phi[K.dict_volumes[v]]-phi0[v])
        #
        # for j,e in enumerate(self.list_e):
        #     for i,v in enumerate(K.dict_volumes.keys()):
        #         dr[j]-=2.*dphi[v]*np.sum(J_phi[j,K.dict_volumes[v]])
        dr=np.add(DR1,DR2)
        return dr

    def calc_shear_dr(self,K):


        Q,dP,X=self.calc_flows_pressures(K)
        K.dV=dP[:]
        dr=np.multiply(np.subtract(2.*K.alpha_1*np.power(np.multiply(dP,K.R),2),K.alpha_0*np.ones(self.M)), K.R)

        return 2.*dr

    def save_dynamic_output(self,*args):

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

    # evaluate time series
    def update_radii(self,R,K):
        K.R=R
        K.C=np.power(R,4)*K.k

    def evaluate_timeline(self,nsol,K,mode):

        options=['absorption+shear','shear+absorption','absorption','shear','absorption+volumetric','absorption+volumetric+shear']
        func=[self.evaluate_absorption_shear, self.evaluate_shear_absorption,self.evaluate_absorption,self.evaluate_shear,self.evaluate_absorption_volumetric,self.evaluate_absorption_volumetric_shear]
        dict_output={}
        check=True

        for i,op in enumerate(options):
            if op==mode:
                check=False
                dict_output=func[i](nsol,K)
                break
        if check:
            print(' nothing to do with my time')

        return dict_output

    def evaluate_absorption_shear(self,nsol,K):

        dict_output={}
        phi0=np.ones(self.M)*K.phi0
        F, PHI, C, PE =[],[],[],[]

        for i in range(len(nsol[:,0])):

            self.update_radii(nsol[i,:],K)
            sq_R=np.power(K.R,2)

            c,B_new,K=self.calc_profile_concentration(K)
            phi=self.calc_absorption( K.R, K )

            overall_uptake_diff=np.sum(np.power(np.subtract(phi,phi0),2))
            dissipation=np.sum(np.multiply(np.power(K.dV,2),np.power(sq_R,2)))
            volume_penalty=np.sum( sq_R )

            F.append( overall_uptake_diff+K.alpha_1*dissipation+K.alpha_0*volume_penalty )
            PHI.append(phi)
            C.append(c)
            PE.append(self.calc_PE(K))

        dict_output['radii_temporal']=nsol
        dict_output['cost']=F
        dict_output['uptake']=PHI
        dict_output['concentration']=C
        dict_output['PE']=PE

        return dict_output

    def evaluate_shear_absorption(self,nsol,K):

        dict_output={}
        phi0=np.ones(self.M)*K.phi0
        F, PHI, C, PE =[],[],[],[]

        for i in range(len(nsol[:,0])):

            self.update_radii(nsol[i,:],K)
            sq_R=np.power(K.R,2)

            c,B_new,K=self.calc_profile_concentration(K)
            phi=self.calc_absorption( K.R, K )

            overall_uptake_diff=np.sum(np.power(np.subtract(phi,phi0),2))
            dissipation=np.sum(np.multiply(np.power(K.dV,2),np.power(sq_R,2)))
            volume_penalty=np.sum( sq_R )

            F.append( K.alpha_0* overall_uptake_diff+dissipation+K.alpha_1*volume_penalty )

            PHI.append(phi)
            C.append(c)
            PE.append(self.calc_PE(K))

        dict_output['radii_temporal']=nsol
        dict_output['cost']=F
        dict_output['uptake']=PHI
        dict_output['concentration']=C
        dict_output['PE']=PE

        return dict_output

    def evaluate_absorption(self,nsol,K):

        dict_output={}
        phi0=np.ones(self.M)*K.phi0
        F, PHI, C, PE =[],[],[],[]

        for i in range(len(nsol[:,0])):

            self.update_radii(nsol[i,:],K)
            sq_R=np.power(K.R,2)

            c,B_new,K=self.calc_profile_concentration(K)
            phi=self.calc_absorption( K.R, K )

            F.append(np.sum(np.power(np.subtract(phi,phi0),2)))
            PHI.append(phi)
            C.append(c)
            PE.append(self.calc_PE(K))

        dict_output['radii_temporal']=nsol
        dict_output['cost']=F
        dict_output['uptake']=PHI
        dict_output['concentration']=C
        dict_output['PE']=PE

        return dict_output

    def evaluate_absorption_volumetric(self,nsol,K):

        dict_output={}
        ones=np.ones(len(K.dict_volumes.values()))
        phi0=ones*K.phi0
        dphi=ones
        F, PHI, C, PE =[],[],[],[]

        for i in range(len(nsol[:,0])):

            self.update_radii(nsol[i,:],K)
            sq_R=np.power(K.R,2)

            c,B_new,K=self.calc_profile_concentration(K)
            phi=self.calc_absorption( K.R, K )

            for j,v in enumerate(K.dict_volumes.keys()):
                dphi[j]=np.sum(phi[K.dict_volumes[v]])-phi0[j]

            F.append(np.sum(np.power(dphi,2)))
            PHI.append(phi)
            C.append(c)
            PE.append(self.calc_PE(K))

        dict_output['radii_temporal']=nsol
        dict_output['cost']=F
        dict_output['uptake']=PHI
        dict_output['concentration']=C
        dict_output['PE']=PE

        return dict_output

    def evaluate_absorption_volumetric_shear(self,nsol,K):

        dict_output={}
        ones=np.ones(len(K.dict_volumes.values()))
        phi0=ones*K.phi0
        dphi=ones
        F, PHI, C, PE =[],[],[],[]

        for i in range(len(nsol[:,0])):

            self.update_radii(nsol[i,:],K)
            sq_R=np.power(K.R,2)

            c,B_new,K=self.calc_profile_concentration(K)
            phi=self.calc_absorption( K.R, K )
            dissipation=np.sum(np.multiply(np.power(K.dV,2),np.power(sq_R,2)))
            volume_penalty=np.sum( sq_R )

            for j,v in enumerate(K.dict_volumes.keys()):
                dphi[j]=np.sum(phi[K.dict_volumes[v]])-phi0[j]

            F.append(np.sum(np.power(dphi,2))+ K.alpha_1*dissipation+K.alpha_0*volume_penalty)

            # F.append( K.alpha_1*dissipation+K.alpha_0*volume_penalty )
            PHI.append(phi)
            C.append(c)
            PE.append(self.calc_PE(K))

        dict_output['radii_temporal']=nsol
        dict_output['cost']=F
        dict_output['uptake']=PHI
        dict_output['concentration']=C
        dict_output['PE']=PE

        return dict_output

    def evaluate_shear(self,nsol,K):

        dict_output={}
        phi0=np.ones(self.M)*K.phi0
        F, PHI, C, PE =[],[],[],[]

        for i in range(len(nsol[:,0])):

            self.update_radii(nsol[i,:],K)
            sq_R=np.power(K.R,2)

            c,B_new,K=self.calc_profile_concentration(K)
            phi=self.calc_absorption( K.R, K )

            dissipation=np.sum(np.multiply(np.power(K.dV,2),np.power(sq_R,2)))
            volume_penalty=np.sum( sq_R )
            F.append( K.alpha_1*dissipation+K.alpha_0*volume_penalty )

            PHI.append(phi)
            C.append(c)
            PE.append(self.calc_PE(K))

        dict_output['radii_temporal']=nsol
        dict_output['cost']=F
        dict_output['uptake']=PHI
        dict_output['concentration']=C
        dict_output['PE']=PE

        return dict_output

class ivp_metabolic_adaptation(metabolic_adaptation, object):

    def __init__(self):
        super(ivp_metabolic_adaptation,self).__init__()

    # optimize networks with custom gradient descent process, define termination handling

    def flatlining_absorption_shear(self,t,R,K):

        # define needed cuttoff digit
        x=16
        # print('time_event:'+str(t))
        # calc cost function
        K.C=np.power(R,4)*K.k
        K.R=R
        sq_R=np.power(R,2)
        c,B_new,K=self.calc_profile_concentration(K)
        phi=self.calc_absorption(K.R, K)
        phi0=np.ones(self.M)*K.phi0

        Q,dP,X=self.calc_flows_pressures(K)
        F=np.sum(np.power(np.subtract(phi,phi0),2))+np.sum( np.add( K.alpha_1*np.multiply(np.power(dP,2),np.power(sq_R,2)), K.alpha_0*sq_R ))

        dr,PE=self.calc_absorption_shear_dr(K)
        # test relative changes
        # dF=np.subtract(F,K.F)
        dF=np.sum(np.power(dr,2))
        # K.F=F
        z=np.round( np.divide(dF,F) ,x  )-np.power(10.,-(x-1))
        # dr,PE=self.calc_absorption_shear_dr(K)
        # z=np.round(np.linalg.norm(np.divide(dr,R)),x)-np.power(10.,-(x-1))
        # print('flatlining:'+str(z))
        return z
    flatlining_absorption_shear.terminal= True
    flatlining_absorption_shear.direction = -1

    def flatlining_shear_absorption(self,t,R,K):

        # define needed cuttoff digit
        x=16

        # calc cost function
        K.C=np.power(R,4)*K.k
        K.R=R
        sq_R=np.power(R,2)
        c,B_new,K=self.calc_profile_concentration(K)
        phi=self.calc_absorption(K.R,K)
        phi0=np.ones(len(R))*K.phi0

        Q,dP,X=self.calc_flows_pressures(K)
        F=K.alpha_0*np.sum(np.power(np.subtract(phi,phi0),2))+np.sum( np.add( K.k*np.multiply(np.power(dP,2),np.power(sq_R,2)), np.sqrt(K.k)*K.alpha_1*sq_R ))
        dr,PE=self.calc_shear_absorption_dr(K)
        # test relative changes
        # dF=np.subtract(F,K.F)
        # K.F=F
        dF=np.sum(np.power(dr,2))
        z=np.round( np.divide(dF,F) ,x  )-np.power(10.,-(x-1))

        return z
    #
    flatlining_shear_absorption.terminal= True
    flatlining_shear_absorption.direction = -1

    def flatlining_absorption(self,t,R,K):

        # define needed cuttoff digit
        x=16

        # calc cost function
        K.C=np.power(R,4)*K.k
        K.R=R
        sq_R=np.power(R,2)
        c,B_new,K=self.calc_profile_concentration(K)
        phi=self.calc_absorption(K.R,K)
        phi0=np.ones(len(R))*K.phi0

        F=np.sum(np.power(np.subtract(phi,phi0),2))
        dr,PE=self.calc_absorption_dr(K)

        dF=np.sum(np.power(dr,2))
        z=np.round( np.divide(dF,F) ,x  )-np.power(10.,-(x-1))

        return z

    flatlining_absorption.terminal= True
    flatlining_absorption.direction = -1

    def flatlining_absorption_volumetric(self,t,R,K):

        # define needed cuttoff digit
        x=16

        # calc cost function
        K.C=np.power(R,4)*K.k
        K.R=R
        sq_R=np.power(R,2)
        c,B_new,K=self.calc_profile_concentration(K)
        phi=self.calc_absorption(K.R, K)

        ones=np.ones(len(K.dict_volumes.values()))
        phi0=ones*K.phi0
        dphi=ones
        sum_phi=0.
        for i,v in enumerate(K.dict_volumes.keys()):
            dphi[i]=np.sum(phi[K.dict_volumes[v]])-phi0[i]
            # sum_phi+=np.sum(phi[K.dict_volumes[v]])
        F=np.sum(np.power(dphi,2))
        Q,dP,X=self.calc_flows_pressures(K)
        F+=np.sum(np.power(np.subtract(phi,phi0),2))+np.sum( np.add( K.alpha_1*np.multiply(np.power(dP,2),np.power(sq_R,2)), K.alpha_0*sq_R ))

        dr=self.calc_absorption_volumetric_dr(K)
        # test relative changes
        dF=np.sum(np.power(dr,2))

        z=np.round( np.divide(dF,F) ,x  )-np.power(10.,-(x-1))

        return z
    flatlining_absorption_volumetric.terminal= True
    flatlining_absorption_volumetric.direction = -1

    def flatlining_absorption_volumetric_shear(self,t,R,K):

        # define needed cuttoff digit
        x=16

        # calc cost function
        K.C=np.power(R,4)*K.k
        K.R=R
        sq_R=np.power(R,2)
        c,B_new,K=self.calc_profile_concentration(K)
        phi=self.calc_absorption(K.R,K)

        ones=np.ones(len(K.dict_volumes.values()))
        phi0=ones*K.phi0
        dphi=ones
        sum_phi=0.
        for i,v in enumerate(K.dict_volumes.keys()):
            dphi[i]=np.sum(phi[K.dict_volumes[v]])-phi0[i]
            # sum_phi+=np.sum(phi[K.dict_volumes[v]])
        F=np.sum(np.power(dphi,2))
        dr=self.calc_absorption_volumetric_shear_dr(K)
        # test relative changes
        # dF=np.subtract(F,K.F)
        # K.F=F
        dF=np.sum(np.power(dr,2))

        z=np.round( np.divide(dF,F) ,x  )-np.power(10.,-(x-1))

        return z
    flatlining_absorption_volumetric_shear.terminal= True
    flatlining_absorption_volumetric_shear.direction = -1

    def flatlining_shear(self,t,R,K):

        # define needed cuttoff digit
        x=30

        # calc cost function
        K.R=R
        K.C=np.power(R,4)*K.k
        sq_R=np.power(R,2)
        Q,dP,X=self.calc_flows_pressures(K)
        F=np.sum( np.add( K.alpha_1*np.multiply(np.power(dP,2)*K.k,np.power(sq_R,2)), K.alpha_0*sq_R*np.sqrt(K.k) ))

        # test relative changes
        # dF=np.subtract(F,K.F)
        # K.F=F
        dr=self.calc_shear_dr(K)
        dF=np.sum(np.power(dr,2))
        z=np.round( np.divide(dF,F) ,x  )-np.power(10.,-(x-1))

        return z

    flatlining_shear.terminal= True
    flatlining_shear.direction = -1

    # have an update routine which constantly produces output/reports from which new simulations may be started

    # solving using conventional scipy ode methods, exploiting stiffness handling of LSODA, no noise handling

    def propagate_system_dynamic_report(self,K,t_span,t_eval,mode):

        options=['absorption+shear','shear+absorption','absorption','shear','absorption+volumetric','absorption+volumetric+shear']
        func=[self.solve_absorption_shear, self.solve_shear_absorption,self.solve_absorption,self.solve_shear,self.solve_absorption_volumetric,self.solve_absorption_volumetric_shear]
        events=[self.flatlining_absorption_shear,self.flatlining_shear_absorption,self.flatlining_absorption,self.flatlining_shear,self.flatlining_absorption_volumetric,self.flatlining_absorption_volumetric_shear]
        check=True

        for i,op in enumerate(options):
            if op==mode:
                check=False
                nsol=si.solve_ivp(func[i],t_span,K.R,args=( [K] ),t_eval=t_eval,method='LSODA',events=events[i])
                break
        if check:
            sys.exit('no legitimate mode')

        sol=np.transpose(np.array(nsol.y))
        K.R=sol[-1,:]
        K.C=np.power(K.R,4)*K.k
        K.set_network_attributes()

        return sol,K

    def solve_shear(self,t,R,K):

        K.C=np.power(R,4)*K.k
        K.R=R

        dr=self.calc_shear_dr(K)
        self.save_dynamic_output([R],['radii'],K)

        return dr

    def solve_absorption_volumetric_shear(self,t,R,K):

        K.C=np.power(R,4)*K.k
        K.R=R

        dr=self.calc_absorption_volumetric_shear_dr(K)
        self.save_dynamic_output([R,K.PE],['radii','PE'],K)

        return dr

    def solve_absorption_volumetric(self,t,R,K):

        K.C=np.power(R,4)*K.k
        K.R=R

        dr=self.calc_absorption_volumetric_dr(K)
        self.save_dynamic_output([R,K.PE],['radii','PE'],K)

        return dr

    def solve_absorption(self,t,R,K):

        K.C=np.power(R,4)*K.k
        K.R=R

        dr,PE=self.calc_absorption_dr(K)
        self.save_dynamic_output([R,PE],['radii','PE'],K)

        return dr

    def solve_shear_absorption(self,t,R,K):

        K.C=np.power(R,4)*K.k
        K.R=R

        dr,PE=self.calc_shear_absorption_dr(K)
        self.save_dynamic_output([R,PE],['radii','PE'],K)

        return dr

    def solve_absorption_shear(self,t,R,K):

        K.C=np.power(R,4)*K.k
        K.R=R

        dr,PE=self.calc_absorption_shear_dr(K)
        self.save_dynamic_output([R,PE],['radii','PE'],K)

        # x=16
        # z=np.round(np.linalg.norm(np.divide(dr,R)),x)-np.power(10.,-(x-1))
        # print('num_solve:'+str(z))
        # print('time_calc:'+str(t))
        return dr

class custom_metabolic_adaptation(metabolic_adaptation, object):

    def __init__(self):
        super(custom_metabolic_adaptation,self).__init__()

    def propagate_system_custom(self,K,mode):

        if 'absorption+shear' ==  mode:

            sol,K=self.solve_absorption_shear_volumetric_custom(K)

        elif 'shear+absorption' ==  mode:

            sol,K=self.solve_shear_absorption_volumetric_custom(K)

        elif 'absorption' ==  mode:

            sol,K=self.solve_absorption_custom(K)

        elif 'absorption+volumetric' ==  mode:

            sol,K=self.solve_absorption_volumetric_custom(K)

        elif 'absorption+volumetric+shear' ==  mode:

            sol,K=self.solve_absorption_volumetric_shear_custom(K)

        elif 'shear' ==  mode:

            sol,K=self.solve_shear_custom(K)

        elif 'shear+fluctuation' ==  mode:

            sol,K=self.solve_shear_fluctuation_custom(K)

        else:
            os.abort('no legitimate mode')

        K.C=np.power(K.R,4)*K.k
        K.set_network_attributes()

        return np.array(sol),K

    def solve_absorption_shear_custom(self,K):

        sol=[]
        for i in range(self.iterations):
            try:
                dr,PE=self.calc_absorption_shear_dr(K)
                self.save_dynamic_output([R,PE],['radii','PE'],K)
                K.R=np.add(K.R,dr*self.dt)
                K.C=np.power(K.R,4)*K.k
                sol.append(K.R)
            except:
                break

        return sol,K

    def solve_shear_absorption_custom(self,K):

        sol=[]
        for i in range(self.iterations):
            try:
                dr,PE=self.calc_shear_absorption_dr(K)
                self.save_dynamic_output([R,PE],['radii','PE'],K)
                K.R=np.add(K.R,dr*self.dt)
                K.C=np.power(K.R,4)*K.k
                sol.append(K.R)
            except:
                break

        return sol,K

    def solve_volumetric_custom(self,K):

        sol=[]
        for i in range(self.iterations):
            try:
                dr,PE=self.calc_absorption_dr(K)
                self.save_dynamic_output([R,PE],['radii','PE'],K)
                K.R=np.add(K.R,dr*self.dt)
                K.C=np.power(K.R,4)*K.k
                sol.append(K.R)
            except:
                break

        return sol,K

    def solve_absorption_volumetric_shear_custom(self,K):

        sol=[]
        for i in range(self.iterations):
            try:
                dr=self.calc_absorption_volumetric_shear_dr(K)
                self.save_dynamic_output([R,K.PE],['radii','PE'],K)
                K.R=np.add(K.R,dr*self.dt)
                K.C=np.power(K.R,4)*K.k
                sol.append(K.R)
            except:
                break

        return sol,K

    def solve_shear_custom(self,K):

        sol=[]
        for i in range(self.iterations):
            try:
                dr,PE=self.calc_shear_dr(K)
                self.save_dynamic_output([R],['radii'],K)
                K.R=np.add(K.R,dr*self.dt)
                K.C=np.power(K.R,4)*K.k
                sol.append(K.R)
            except:
                break

        return sol,K

    def solve_shear_fluctuation_custom(self,K):

        sol=[]
        for i in range(self.iterations):

            # try:
                # dr,shear_sq,dV_sq,F_sq, avg_phi=self.calc_shear_fluctuation_dr(K)
                # self.save_dynamic_output([shear_sq,dV_sq,F_sq, avg_phi],['shear_sq','dV_sq','F_sq','Phi'],K)
                dr,shear_sq,dV_sq,F_sq=self.calc_shear_fluctuation_dr(K)
                self.save_dynamic_output([shear_sq,dV_sq,F_sq],['shear_sq','dV_sq','F_sq'],K)
                K.R=np.add(K.R,dr*self.dt)
                K.C=np.power(K.R,4)*K.k
                sol.append(K.R)
            # except:
            #     break

        return sol,K
