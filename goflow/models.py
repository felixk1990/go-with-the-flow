# @Author: Felix Kramer <kramer>
# @Date:   23-06-2021
# @Email:  kramer@mpi-cbg.de
# @Project: phd_network_remodelling
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-09-05T00:29:15+02:00

import numpy as np
import kirchhoff.circuit_flow as kfc
import hailhydro.flow_init as hfi

# custom
from goflow.init_ivp import *


def init(mode='default',**kwargs):

    model_mode={
        'default': murray,
        'murray': murray,

    }

    if mode=='default':
        model=model_mode['default'](**kwargs)
    else:
        model=model_mode[mode](**kwargs)

    return model

def initialize_flow_on_crystal(dict_pars):

    circuit=kfc.initialize_flow_circuit_from_crystal(crystal_type=dict_pars['type'],periods=dict_pars['periods'])
    circuit.set_source_landscape(dict_pars['source'])
    circuit.set_plexus_landscape(dict_pars['plexus'])
    flow=hfi.initialize_flow_on_circuit(circuit)

    return flow
# def init_custom_adaptation(flow_type,morph_mode='default',**kwargs):
#
#
#     if morph_mode=='default':
#         IM=morph_dynamic(flow_type)
#         MR=murray(IM)
#         if 'sys_pars' in kwargs:
#             MR.set_system_parameters(kwargs['sys_pars'])
#         if 'solv_opt' in kwargs:
#             MR.set_solver_options(kwargs['solv_opt'])
#
#     return MR

class model():

    def __init__(self):

        self.ivp_options={
            't0': 0.,
            't1': 1.,
            'x0':1,
            'num':100,
        }

        self.model_args=[]
        self.solver_options={
            't_eval':np.linspace(self.ivp_options['t0'],self.ivp_options['t1'],num=self.ivp_options['num'])
        }



class murray(model, object):

    def __init__(self,**kwargs):

        self.null_decimal=30
        self.jac=False
        super(murray,self).__init__()

        self.events={
                'default':self.flatlining_default,
                'dynamic':self.flatlining_dynamic,
            }
        self.model_args=[1.,1.]
        self.solver_options.update({ 'events':'default'})
        if 'pars' in kwargs:
            self.set_model_parameters(kwargs['pars'])

    def update_event_func(self):

        try:
            self.solver_options['events']=self.events[self.solver_options['events']]

        except:
            print('Warning: Event handling got inadequadt event functin, falling back to default')
            self.solver_options['events']=self.events['default']

    def set_model_parameters(self,model_pars):

        for k,v in model_pars.items():

            if 'alpha_0' == k :
                self.model_args[0]=v

            if 'alpha_1' == k :
                self.model_args[1]=v

    def set_solver_options(self,solv_opt):

        for k,v in solv_opt.items():

            self.solver_options[k]=v

        self.update_event_func()

    def flatlining_default(self,  t,  x_0,  *args):

        flow, alpha_1, alpha_2 = args

        self.jac=True
        F,dF=self.calc_cost_stimuli( t, x_0, *args)
        dF_abs=np.linalg.norm(dF)
        quality=np.round( np.divide(dF_abs,F) ,self.null_decimal  )

        z=  quality   -   np.power(10.,-(self.null_decimal-1))

        return z

    flatlining_default.terminal= True
    flatlining_default.direction = -1

    def flatlining_dynamic(self,  t,  x_0,  *args):

        flow, alpha_1, alpha_2 = args

        self.jac=True
        dx=self.calc_update_stimuli( t, x_0, *args)
        dx_abs=np.absolute(dx)
        rel_r=np.divide(dx_abs,x_0)
        quality=np.round( np.linalg.norm(rel_r)  ,self.null_decimal  )

        z=  quality   -   np.power(10.,-(self.null_decimal-1))

        return z

    flatlining_dynamic.terminal= True
    flatlining_dynamic.direction = -1

    def calc_update_stimuli(self,t,x_0, *args):

        flow, alpha_0, alpha_1 = args

        x_sq,p_sq,k,src=self.get_stimuli_pars(flow,x_0)

        s1=2.*alpha_1*np.multiply(  p_sq,   x_sq    )
        s2=alpha_0*np.ones(len(x_0))
        ds=np.subtract(s1,s2)
        dx=2*np.multiply(ds, x_0)

        return dx

    def calc_cost_stimuli(self, t, x_0, *args):

        flow, alpha_0, alpha_1 = args

        x_sq,p_sq,k,src=self.get_stimuli_pars(flow,x_0)

        f1= alpha_1*np.multiply( p_sq, np.power(x_sq,2))
        f2= alpha_0 *x_sq
        F = np.sum( np.add( f1  , f2 ))

        if self.jac:

            dF=-self.calc_update_stimuli(t,x_0, *args)

            return F,dF

        return F

    def get_stimuli_pars(self,flow,x_0):

        k=  flow.circuit.scales['conductance']
        src= flow.circuit.nodes['source']

        conductivity =    flow.calc_conductivity_from_cross_section(  np.power(x_0,2),    k )
        x_sq=   flow.calc_cross_section_from_conductivity(  conductivity , k )
        p_sq, q_sq= flow.calc_sq_flow(  conductivity    , src   )

        return x_sq,p_sq,k,src

# class corson(murray, object):
#     pass

        #
        # def nsolve_heun_hucai_manual_fluctuation(self,scale_data,parameters , K, IO):
        #
        #     Num_steps=scale_data[0]
        #     dt=scale_data[1]
        #     sample=scale_data[2]
        #     nullity=[]
        #     c_mean=[]
        #     gamma=parameters[0]
        #     scale=parameters[1]
        #     volume_penalty=parameters[2]
        #     mode=parameters[3]
        #     num_realizations=parameters[4]
        #     p=parameters[5]
        #     source=np.where(K.J> 0)[0][0]
        #     sinks=np.where(K.J < 0)[0]
        #     OUTPUT_C, OUTPUT_S = IO.init_kirchhoff_data(scale_data,parameters,K)
        #
        #     #scale system
        #     g1_m=1.-gamma
        #
        #     #calc network dynamic
        #     B,BT=K.get_incidence_matrices()
        #     M=nx.number_of_edges(K.G)
        #     N=nx.number_of_nodes(K.G)
        #     dC=np.zeros(M)
        #     dT=(np.ones(M)*dt)
        #     threshold=10.**(-20)
        #
        #     for i in range(Num_steps):
        #
        #
        #         #1) prediction
        #         f_sq,v_sq=self.calc_sq_flow_random_manual(K.C,B,BT,[mode,N,num_realizations,source,sinks,p])
        #         cg_m=np.power(K.C,g1_m)
        #         stress_volume=np.subtract( np.multiply(v_sq,cg_m) , volume_penalty )
        #         dC= np.multiply( scale * dt, np.multiply(K.C, stress_volume  ))
        #         c_aux= np.add(K.C,dC)
        #         control=np.where( c_aux < threshold )
        #         for m in control:
        #             dT[m]=0.
        #             c_aux[m]=10.**(-21)
        #
        #         #2) correction
        #         f_sq_aux,v_sq_aux=self.calc_sq_flow_random_manual(c_aux,B,BT,[mode,N,num_realizations,source,sinks,p])
        #         cg_aux=np.power(c_aux,g1_m)
        #         stress_volume=np.subtract( np.multiply(v_sq_aux,cg_aux) , volume_penalty )
        #         dC_aux= np.multiply( scale * dt, np.multiply(c_aux, stress_volume  ))
        #
        #         K.C = np.add( K.C, np.divide(np.add( dC , dC_aux),2.) )
        #         control=np.where( K.C < threshold )
        #         for m in control:
        #             dT[m]=0.
        #             K.C[m]=10.**(-21)
        #         if i % sample == 0:
        #             # K.E[j] = np.add( np.sum(np.multiply( v_sq_aux, K.C)) ,  * np.sum( np.power(K.C, gamma)))
        #             OUTPUT_C = np.vstack((OUTPUT_C,K.C[:]))
        #             # j+=1
        #
        #             K.clipp_graph()
        #             H=nx.Graph(K.H)
        #             n=(nx.number_connected_components(H)+nx.number_of_edges(H)-nx.number_of_nodes(H))/(1.+M-N)
        #             nullity.append(n)
        #             c_mean.append(np.mean(K.H_C))
        #
        #         self.print_step(i,Num_steps)
        #
        #     IO.save_kirchhoff_data(OUTPUT_C,OUTPUT_S,K)
        #     IO.save_nparray(nullity,'nullity_time')
        #
        # def nsolve_euler_hucai_R_random(self,scale_data,parameters, K, IO):
        #
        #     Num_steps=scale_data[0]
        #     dt=scale_data[1]
        #     sample=scale_data[2]
        #
        #     gamma=parameters[0]
        #     g1=gamma+1
        #     c0=parameters[1]
        #     c1=parameters[2]
        #     mu=parameters[3]
        #     var=parameters[4]
        #     kappa=1.
        #     threshold=10.**(-20)
        #     OUTPUT_C, OUTPUT_S = IO.init_kirchhoff_data(scale_data,parameters,K)
        #     nullity =[]
        #     dissipation=[]
        #     volume=[]
        #     #calc network dynamic
        #
        #     B,BT=K.get_inicidence_matrices()
        #     M=nx.number_of_edges(K.G)
        #     N=nx.number_of_nodes(K.G)
        #     dC=np.zeros(M)
        #     self.setup_random_fluctuations(N,mu,var)
        #
        #     # unit/scales
        #     alpha=(c0*K.f**g1)/((K.k**gamma)*(K.l**(3*g1)))
        #     ALPHA=(np.ones(M)*alpha)
        #     beta=(c1**K.k*(K.l**3)/K.f)**gamma
        #     BETA=(np.ones(M)*beta)
        #     HALF=(np.ones(M)*0.5)
        #     REC_3=(np.ones(M)*(-3.))
        #     SIGMA=(np.ones(M)*gamma)
        #     dT=(np.ones(M)*dt)
        #     KAPPA=(np.ones(M)*kappa)
        #
        #     for i in range(Num_steps):
        #
        #         c_aux=K.C[:]
        #         C=np.diag( np.multiply(KAPPA,c_aux**4 ) )
        #         F_sq=self.calc_sq_flow_random(C,B,BT)
        #
        #         tau=np.multiply(np.power(F_sq,HALF),np.power(c_aux,REC_3))
        #         shear_sigma=np.power(tau,SIGMA)
        #         diff_shear=np.multiply(np.subtract(shear_sigma,BETA),c_aux)
        #         dC=np.multiply(np.multiply(ALPHA,diff_shear),dT)
        #
        #         K.C[:]=np.add(c_aux,dC)
        #
        #         if np.any(K.C[:] < threshold):
        #             for m in range(M):
        #                 if K.C[m] < threshold:
        #                     dT[m]=0.
        #                     K.C[m]=10.**(-21)
        #
        #         if i % sample == 0:
        #             OUTPUT_C=np.vstack((OUTPUT_C,K.C) )
        #             OUTPUT_S=np.vstack((OUTPUT_S,K.J) )
        #             s=0
        #             K.clipp_graph()
        #             n=(1.+nx.number_of_edges(K.H)-nx.number_of_nodes(K.H))/(1.+nx.number_of_edges(K.G)-nx.number_of_nodes(K.G))
        #             nullity.append(n)
        #             d=np.divide(F_sq,np.diag(C))
        #             dissipation.append(np.sum(d))
        #             volume.append(np.sum(K.C[:]**2))
        #
        #         self.print_step(i,Num_steps)
        #
        #     IO.save_kirchhoff_data(OUTPUT_C,OUTPUT_S,K)
        #     IO.save_nparray(nullity,'nullity_time')
        #     IO.save_nparray(dissipation,'dissipation_time')
        #     IO.save_nparray(volume,'volume_time')
        #
        # def nsolve_euler_hucai_R_random_manual_sink(self,scale_data,parameters, data_manual_sink, K, IO):
        #
        #     Num_steps=scale_data[0]
        #     dt=scale_data[1]
        #     sample=scale_data[2]
        #
        #     gamma=parameters[0]
        #     g1=gamma+1
        #     c0=parameters[1]
        #     c1=parameters[2]
        #
        #     kappa=1.
        #     threshold=10.**(-20)
        #     OUTPUT_C, OUTPUT_S = IO.init_kirchhoff_data(scale_data,parameters,K)
        #     nullity =[]
        #     dissipation=[]
        #     volume=[]
        #     #calc network dynamic
        #
        #     B,BT=K.get_inicidence_matrices()
        #     M=nx.number_of_edges(K.G)
        #     N=nx.number_of_nodes(K.G)
        #     dC=np.zeros(M)
        #
        #     # unit/scales
        #     alpha=(c0*K.f**g1)/((K.k**gamma)*(K.l**(3*g1)))
        #     ALPHA=(np.ones(M)*alpha)
        #     beta=(c1**K.k*(K.l**3)/K.f)**gamma
        #     BETA=(np.ones(M)*beta)
        #     HALF=(np.ones(M)*0.5)
        #     REC_3=(np.ones(M)*(-3.))
        #     SIGMA=(np.ones(M)*gamma)
        #     dT=(np.ones(M)*dt)
        #     KAPPA=(np.ones(M)*kappa)
        #
        #     for i in range(Num_steps):
        #
        #         c_aux=K.C[:]
        #         C=np.diag( np.multiply(KAPPA,c_aux**4 ) )
        #         F_sq=self.calc_sq_flow_random_manual(C,B,BT,data_manual_sink)
        #
        #         tau=np.multiply(np.power(F_sq,HALF),np.power(c_aux,REC_3))
        #         shear_sigma=np.power(tau,SIGMA)
        #         diff_shear=np.multiply(np.subtract(shear_sigma,BETA),c_aux)
        #         dC=np.multiply(np.multiply(ALPHA,diff_shear),dT)
        #
        #         K.C[:]=np.add(c_aux,dC)
        #
        #         if np.any(K.C[:] < threshold):
        #             for m in range(M):
        #                 if K.C[m] < threshold:
        #                     dT[m]=0.
        #                     K.C[m]=10.**(-21)
        #
        #         if i % sample == 0:
        #             OUTPUT_C=np.vstack((OUTPUT_C,K.C) )
        #             OUTPUT_S=np.vstack((OUTPUT_S,K.J) )
        #             s=0
        #             K.clipp_graph()
        #             n=(1.+nx.number_of_edges(K.H)-nx.number_of_nodes(K.H))/(1.+nx.number_of_edges(K.G)-nx.number_of_nodes(K.G))
        #             nullity.append(n)
        #             d=np.divide(F_sq,np.diag(C))
        #             dissipation.append(np.sum(d))
        #             volume.append(np.sum(K.C[:]**2))
        #
        #         self.print_step(i,Num_steps)
        #
        #     IO.save_kirchhoff_data(OUTPUT_C,OUTPUT_S,K)
        #     IO.save_nparray(nullity,'nullity_time')
        #     IO.save_nparray(dissipation,'dissipation_time')
        #     IO.save_nparray(volume,'volume_time')
        #
        # def nsolve_euler_hucai_random_scaling(self,scale_data,parameters,K,IO):
        #
        #     Num_steps=scale_data[0]
        #     dt=scale_data[1]
        #     sample=scale_data[2]
        #
        #     gamma=parameters[0]
        #     g1=gamma+1
        #     c0=parameters[1]
        #     c1=parameters[2]
        #     mu=parameters[3]
        #     var=parameters[4]
        #     kappa=1.
        #     threshold=10.**(-20)
        #     OUTPUT_C, OUTPUT_S = IO.init_kirchhoff_data(scale_data,parameters,K)
        #     nullity =[]
        #     dissipation=[]
        #     volume=[]
        #     #calc network dynamic
        #
        #     B,BT=K.get_inicidence_matrices()
        #     M=nx.number_of_edges(K.G)
        #     N=nx.number_of_nodes(K.G)
        #     dC=np.zeros(M)
        #     self.setup_random_fluctuations(N,mu,var)
        #
        #     # unit/scales
        #     alpha=(c0*K.f**g1)/((K.k**gamma)*(K.l**(3*g1)))
        #     ALPHA=(np.ones(M)*alpha)
        #     beta=(c1**K.k*(K.l**3)/K.f)**gamma
        #     BETA=(np.ones(M)*beta)
        #     dT=(np.ones(M)*dt)
        #     KAPPA=(np.ones(M)*kappa)
        #
        #     for i in range(Num_steps):
        #         self.mu+=(1./tau)*self.mu*dT[0]
        #         c_aux=K.C[:]
        #         F_sq=self.calc_sq_flow_random(K.C[:],B,BT)
        #
        #         dissipation=np.divide(F_sq,np.power(K.C[:],g1))
        #         diff_shear=np.multiply(np.subtract(shear_sigma,BETA),K.C[:])
        #         dC=np.multiply(np.multiply(ALPHA,diff_shear),dT)
        #         K.C[:]=np.add(K.C[:],dC)
        #
        #         if np.any(K.C[:] < threshold):
        #             for m in range(M):
        #                 if K.C[m] < threshold:
        #                     dT[m]=0.
        #                     K.C[m]=10.**(-21)
        #
        #         if i % sample == 0:
        #             OUTPUT_C=np.vstack((OUTPUT_C,K.C) )
        #             OUTPUT_S=np.vstack((OUTPUT_S,K.J) )
        #             s=0
        #             K.clipp_graph()
        #             n=(1.+nx.number_of_edges(K.H)-nx.number_of_nodes(K.H))/(1.+nx.number_of_edges(K.G)-nx.number_of_nodes(K.G))
        #             nullity.append(n)
        #             d=np.divide(F_sq,np.diag(C))
        #             dissipation.append(np.sum(d))
        #             volume.append(np.sum(K.C[:]**2))
        #
        #         self.print_step(i,Num_steps)
        #
        #     IO.save_kirchhoff_data(OUTPUT_C,OUTPUT_S,K)
        #     IO.save_nparray(nullity,'nullity_time')
        #     IO.save_nparray(dissipation,'dissipation_time')
        #     IO.save_nparray(volume,'volume_time')
        #
        # def nsolve_euler_hucai_random_reduced(self,scale_data,parameters,K,IO):
        #
        #     Num_steps=scale_data[0]
        #     dt=scale_data[1]
        #     sample=scale_data[2]
        #
        #     gamma=parameters[0]
        #     g1=gamma+1
        #     c0=parameters[1]
        #     c1=parameters[2]
        #     mu=parameters[3]
        #     var=parameters[4]
        #     kappa=1.
        #     threshold=10.**(-20)
        #     OUTPUT_C, OUTPUT_S = IO.init_kirchhoff_data(scale_data,parameters,K)
        #     nullity =[]
        #     dissipation=[]
        #     volume=[]
        #     #calc network dynamic
        #
        #     B,BT=K.get_inicidence_matrices()
        #     M=nx.number_of_edges(K.G)
        #     N=nx.number_of_nodes(K.G)
        #     dC=np.zeros(M)
        #     self.setup_random_fluctuations_reduced(N,mu,var)
        #
        #     # unit/scales
        #     alpha=(c0*K.f**g1)/((K.k**gamma)*(K.l**(3*g1)))
        #     ALPHA=(np.ones(M)*alpha)
        #     beta=(c1**K.k*(K.l**3)/K.f)**gamma
        #     BETA=(np.ones(M)*beta)
        #     HALF=(np.ones(M)*0.5)
        #     REC_3=(np.ones(M)*(-3.))
        #     SIGMA=(np.ones(M)*gamma)
        #     dT=(np.ones(M)*dt)
        #     KAPPA=(np.ones(M)*kappa)
        #
        #
        #     for i in range(Num_steps):
        #
        #         c_aux=K.C[:]
        #         C=np.diag( np.multiply(KAPPA,c_aux**4 ) )
        #         F_sq=self.calc_sq_flow_random_reduced(C,B,BT)
        #
        #         tau=np.multiply(np.power(F_sq,HALF),np.power(c_aux,REC_3))
        #         shear_sigma=np.power(tau,SIGMA)
        #         diff_shear=np.multiply(np.subtract(shear_sigma,BETA),c_aux)
        #         dC=np.multiply(np.multiply(ALPHA,diff_shear),dT)
        #
        #         K.C[:]=np.add(c_aux,dC)
        #
        #         if np.any(K.C[:] < threshold):
        #             for m in range(M):
        #                 if K.C[m] < threshold:
        #                     dT[m]=0.
        #                     K.C[m]=10.**(-21)
        #
        #         if i % sample == 0:
        #             OUTPUT_C=np.vstack((OUTPUT_C,K.C) )
        #             OUTPUT_S=np.vstack((OUTPUT_S,K.J) )
        #             s=0
        #             K.clipp_graph()
        #             n=(1.+nx.number_of_edges(K.H)-nx.number_of_nodes(K.H))/(1.+nx.number_of_edges(K.G)-nx.number_of_nodes(K.G))
        #             nullity.append(n)
        #             d=np.divide(F_sq,np.diag(C))
        #             dissipation.append(np.sum(d))
        #             volume.append(np.sum(K.C[:]**2))
        #
        #         self.print_step(i,Num_steps)
        #
        #     IO.save_kirchhoff_data(OUTPUT_C,OUTPUT_S,K)
        #     IO.save_nparray(nullity,'nullity_time')
        #     IO.save_nparray(dissipation,'dissipation_time')
        #     IO.save_nparray(volume,'volume_time')
        #
        # def nsolve_heun_hucai_reduced_radius(self,scale_data,parameters,K,IO):
        #
        #     Num_steps=scale_data[0]
        #     dt=scale_data[1]
        #     sample=scale_data[2]
        #
        #     gamma=parameters[0]
        #     g_p1=gamma+1
        #     g_4m1=4.*(1-gamma)
        #
        #     c0=parameters[1]
        #     c1=parameters[2]
        #     mu=parameters[3]
        #     var=parameters[4]
        #
        #     OUTPUT_C, OUTPUT_S = IO.init_kirchhoff_data(scale_data,parameters,K)
        #     nullity =[]
        #     dissipation=[]
        #     volume=[]
        #
        #     #calc network dynamic
        #     B,BT=K.get_incidence_matrices()
        #     # B=ssp.csc_matrix(B)
        #     # BT=ssp.csc_matrix(BT)
        #     # print(len(BT.getnnz(1)))
        #
        #     M=nx.number_of_edges(K.G)
        #     # N=nx.number_of_nodes(K.G)
        #     C=K.C[:]
        #
        #     R=np.power(np.divide(C,self.kappa),0.25)
        #
        #     # unit/scales
        #     self.setup_random_fluctuations_reduced(K,mu,var)
        #
        #     alpha=(c0*K.f**g_p1)/((K.k**gamma)*(K.l**(3*g_p1)))
        #     beta=(c1**K.k*(K.l**3)/K.f)**gamma
        #     dT=(np.ones(M)*dt*alpha)
        #     sigma=self.sigma/2.
        #     threshold=10.**(-20)
        #
        #     for i in range(Num_steps):
        #
        #         #prediction
        #         dV_sq, F_sq=self.calc_sq_flow(C,B,BT)
        #
        #         R_gamma=np.power(R,g_4m1)
        #         shear_sigma=np.power(np.multiply(dV_sq,R_gamma), sigma)
        #         diff_shear=np.multiply(np.subtract(shear_sigma,beta),R)
        #         prediction_dR=np.multiply(diff_shear,dT)
        #         prediction_R=np.add(R,prediction_dR)
        #
        #         C=np.multiply(np.power(prediction_R,4.),self.kappa)
        #         control=np.where( C < threshold )
        #         for m in control:
        #             dT[m]=0.
        #             C[m]=10.**(-21)
        #         #correction
        #         dV_sq, F_sq=self.calc_sq_flow(C,B,BT)
        #
        #         R_gamma=np.power(prediction_R,g_4m1)
        #         shear_sigma=np.power(np.multiply(dV_sq,R_gamma), sigma)
        #         diff_shear=np.multiply(np.subtract(shear_sigma,beta),prediction_R)
        #         correction_dR=np.multiply(diff_shear,dT)
        #
        #         #update
        #         dR=0.5*(prediction_dR+correction_dR)
        #         R=np.add(R,dR)
        #         K.C=np.multiply(np.power(R,4.),self.kappa)
        #         control=np.where( K.C < threshold )
        #         for m in control:
        #             dT[m]=0.
        #             K.C[m]=10.**(-21)
        #         C=K.C[:]
        #
        #         # output
        #         if i % sample == 0:
        #             OUTPUT_C=np.vstack((OUTPUT_C,K.C) )
        #             OUTPUT_S=np.vstack((OUTPUT_S,K.J) )
        #             s=0
        #             K.clipp_graph()
        #             n=(1.+nx.number_of_edges(K.H)-nx.number_of_nodes(K.H))/(1.+nx.number_of_edges(K.G)-nx.number_of_nodes(K.G))
        #             nullity.append(n)
        #             D=np.divide(F_sq,K.C[:])
        #             dissipation.append(np.sum(D))
        #             A=np.power(R,2.)
        #             volume.append(np.sum(A))
        #
        #             dV_sq, F_sq=self.calc_sq_flow(C,B,BT)
        #             # K.V=np.sqrt(dV_sq)
        #             K.F=np.sqrt(F_sq)
        #
        #         self.print_step(i,Num_steps)
        #
        #     IO.save_kirchhoff_data(OUTPUT_C,OUTPUT_S,K)
        #     IO.save_nparray(nullity,'nullity_time')
        #     IO.save_nparray(dissipation,'dissipation_time')
        #     IO.save_nparray(volume,'volume_time')
        #
        # def nsolve_heun_hucai_effective_radius(self,scale_data,parameters,K,IO):
        #
        #     Num_steps=scale_data[0]
        #     dt=scale_data[1]
        #     sample=scale_data[2]
        #
        #     gamma=parameters[0]
        #     g_p1=gamma+1
        #     g_4m1=4.*(1-gamma)
        #
        #     self.scale=parameters[1]
        #     self.vol_diss=parameters[2]
        #     self.noise=parameters[3]
        #
        #     OUTPUT_C, OUTPUT_S = IO.init_kirchhoff_data(scale_data,parameters,K)
        #     nullity =[]
        #     dissipation=[]
        #     volume=[]
        #
        #     #calc network dynamic
        #     B,BT=K.get_incidence_matrices()
        #
        #     self.M=nx.number_of_edges(K.G)
        #     self.N=nx.number_of_nodes(K.G)
        #     C=K.C[:]
        #
        #     R=np.power(np.divide(C,self.kappa),0.25)
        #
        #     # unit/scales
        #     self.setup_random_fluctuations_effective(K)
        #
        #     alpha=(self.scale*K.f**g_p1)/((K.k**gamma)*(K.l**(3*g_p1)))
        #     beta=(self.vol_diss**K.k*(K.l**3)/K.f)**gamma
        #     dT=(np.ones(self.M)*dt*alpha)
        #     sigma=self.sigma/2.
        #     threshold=10.**(-20)
        #
        #     for i in range(Num_steps):
        #
        #         #prediction
        #         dV_sq, F_sq=self.calc_sq_flow(C,B,BT)
        #
        #         R_gamma=np.power(R,g_4m1)
        #         shear_sigma=np.power(np.multiply(dV_sq,R_gamma), sigma)
        #         diff_shear=np.multiply(np.subtract(shear_sigma,beta),R)
        #         prediction_dR=np.multiply(diff_shear,dT)
        #         prediction_R=np.add(R,prediction_dR)
        #
        #         C=np.multiply(np.power(prediction_R,4.),self.kappa)
        #         control=np.where( C < threshold )
        #         for m in control:
        #             dT[m]=0.
        #             C[m]=10.**(-21)
        #         #correction
        #         dV_sq, F_sq=self.calc_sq_flow(C,B,BT)
        #
        #         R_gamma=np.power(prediction_R,g_4m1)
        #         shear_sigma=np.power(np.multiply(dV_sq,R_gamma), sigma)
        #         diff_shear=np.multiply(np.subtract(shear_sigma,beta),prediction_R)
        #         correction_dR=np.multiply(diff_shear,dT)
        #
        #         #update
        #         dR=0.5*(prediction_dR+correction_dR)
        #         R=np.add(R,dR)
        #         K.C=np.multiply(np.power(R,4.),self.kappa)
        #         control=np.where( K.C < threshold )
        #         for m in control:
        #             dT[m]=0.
        #             K.C[m]=10.**(-21)
        #         C=K.C[:]
        #
        #         # output
        #         if i % sample == 0:
        #             OUTPUT_C=np.vstack((OUTPUT_C,K.C) )
        #             OUTPUT_S=np.vstack((OUTPUT_S,K.J) )
        #             s=0
        #             K.clipp_graph()
        #             n=(1.+nx.number_of_edges(K.H)-nx.number_of_nodes(K.H))/(1.+nx.number_of_edges(K.G)-nx.number_of_nodes(K.G))
        #             nullity.append(n)
        #             D=np.divide(F_sq,K.C[:])
        #             dissipation.append(np.sum(D))
        #             A=np.power(R,2.)
        #             volume.append(np.sum(A))
        #
        #             dV_sq, F_sq=self.calc_sq_flow(C,B,BT)
        #             # K.V=np.sqrt(dV_sq)
        #             K.F=np.sqrt(F_sq)
        #
        #         self.print_step(i,Num_steps)
        #
        #     IO.save_kirchhoff_data(OUTPUT_C,OUTPUT_S,K)
        #     IO.save_nparray(nullity,'nullity_time')
        #     IO.save_nparray(dissipation,'dissipation_time')
        #     IO.save_nparray(volume,'volume_time')
        #
        #     return OUTPUT_C
        #
        # def nsolve_heun_hucai_multisink_radius(self,scale_data,parameters,K,IO):
        #
        #     Num_steps=scale_data[0]
        #     dt=scale_data[1]
        #     sample=scale_data[2]
        #
        #     gamma=parameters[0]
        #     g_p1=gamma+1
        #     g_4m1=4.*(1-gamma)
        #
        #     self.scale=parameters[1]
        #     self.vol_diss=parameters[2]
        #     self.noise=parameters[3]
        #
        #     OUTPUT_C, OUTPUT_S = IO.init_kirchhoff_data(scale_data,parameters,K)
        #     nullity =[]
        #     dissipation=[]
        #     volume=[]
        #
        #     #calc network dynamic
        #     B,BT=K.get_incidence_matrices()
        #
        #     self.M=nx.number_of_edges(K.G)
        #     self.N=nx.number_of_nodes(K.G)
        #     C=K.C[:]
        #
        #     R=np.power(np.divide(C,self.kappa),0.25)
        #
        #     # unit/scales
        #     self.setup_random_fluctuations_multisink(K)
        #
        #     alpha=(self.scale*K.f**g_p1)/((K.k**gamma)*(K.l**(3*g_p1)))
        #     beta=(self.vol_diss**K.k*(K.l**3)/K.f)**gamma
        #     dT=(np.ones(self.M)*dt*alpha)
        #     sigma=self.sigma/2.
        #     threshold=10.**(-20)
        #
        #     for i in range(Num_steps):
        #
        #         #prediction
        #         dV_sq, F_sq=self.calc_sq_flow(C,B,BT)
        #
        #         R_gamma=np.power(R,g_4m1)
        #         shear_sigma=np.power(np.multiply(dV_sq,R_gamma), sigma)
        #         diff_shear=np.multiply(np.subtract(shear_sigma,beta),R)
        #         prediction_dR=np.multiply(diff_shear,dT)
        #         prediction_R=np.add(R,prediction_dR)
        #
        #         C=np.multiply(np.power(prediction_R,4.),self.kappa)
        #         control=np.where( C < threshold )
        #         for m in control:
        #             dT[m]=0.
        #             C[m]=10.**(-21)
        #         #correction
        #         dV_sq, F_sq=self.calc_sq_flow(C,B,BT)
        #
        #         R_gamma=np.power(prediction_R,g_4m1)
        #         shear_sigma=np.power(np.multiply(dV_sq,R_gamma), sigma)
        #         diff_shear=np.multiply(np.subtract(shear_sigma,beta),prediction_R)
        #         correction_dR=np.multiply(diff_shear,dT)
        #
        #         #update
        #         dR=0.5*(prediction_dR+correction_dR)
        #         R=np.add(R,dR)
        #         K.C=np.multiply(np.power(R,4.),self.kappa)
        #         control=np.where( K.C < threshold )
        #         for m in control:
        #             dT[m]=0.
        #             K.C[m]=10.**(-21)
        #         C=K.C[:]
        #
        #         # output
        #         if i % sample == 0:
        #             OUTPUT_C=np.vstack((OUTPUT_C,K.C) )
        #             OUTPUT_S=np.vstack((OUTPUT_S,K.J) )
        #             s=0
        #             K.clipp_graph()
        #             n=(1.+nx.number_of_edges(K.H)-nx.number_of_nodes(K.H))/(1.+nx.number_of_edges(K.G)-nx.number_of_nodes(K.G))
        #             nullity.append(n)
        #             D=np.divide(F_sq,K.C[:])
        #             dissipation.append(np.sum(D))
        #             A=np.power(R,2.)
        #             volume.append(np.sum(A))
        #
        #             dV_sq, F_sq=self.calc_sq_flow(C,B,BT)
        #             # K.V=np.sqrt(dV_sq)
        #             K.F=np.sqrt(F_sq)
        #
        #         self.print_step(i,Num_steps)
        #
        #     IO.save_kirchhoff_data(OUTPUT_C,OUTPUT_S,K)
        #     IO.save_nparray(nullity,'nullity_time')
        #     IO.save_nparray(dissipation,'dissipation_time')
        #     IO.save_nparray(volume,'volume_time')
        #
        #     return OUTPUT_C
        #
        # def nsolve_heun_hucai_terminals(self,scale_data,parameters,K,IO):
        #
        #     Num_steps=scale_data[0]
        #     dt=scale_data[1]
        #     sample=scale_data[2]
        #
        #     gamma=parameters[0]
        #     g_p1=gamma+1
        #     g_4m1=4.*(1-gamma)
        #
        #     c0=parameters[1]
        #     c1=parameters[2]
        #     mu=parameters[3]
        #     var=parameters[4]
        #
        #     OUTPUT_C, OUTPUT_S = IO.init_kirchhoff_data(scale_data,parameters,K)
        #     nullity =[]
        #     dissipation=[]
        #     volume=[]
        #
        #     #calc network dynamic
        #     B,BT=K.get_incidence_matrices()
        #     # B=ssp.csc_matrix(B)
        #     # BT=ssp.csc_matrix(BT)
        #     # print(len(BT.getnnz(1)))
        #
        #     M=nx.number_of_edges(K.G)
        #     # N=nx.number_of_nodes(K.G)
        #     C=K.C[:]
        #
        #     R=np.power(np.divide(C,self.kappa),0.25)
        #
        #     # unit/scales
        #     self.setup_random_fluctuations_reduced(K,mu,var)
        #     alpha=(c0*K.f**g_p1)/((K.k**gamma)*(K.l**(3*g_p1)))
        #     beta=(c1**K.k*(K.l**3)/K.f)**gamma
        #     dT=(np.ones(M)*dt*alpha)
        #     sigma=self.sigma/2.
        #     threshold=10.**(-20)
        #
        #     for i in range(Num_steps):
        #
        #         #prediction
        #         dV_sq, F_sq=self.calc_sq_flow(C,B,BT)
        #
        #         R_gamma=np.power(R,g_4m1)
        #         shear_sigma=np.power(np.multiply(dV_sq,R_gamma), sigma)
        #         diff_shear=np.multiply(np.subtract(shear_sigma,beta),R)
        #         prediction_dR=np.multiply(diff_shear,dT)
        #         prediction_R=np.add(R,prediction_dR)
        #
        #         C=np.multiply(np.power(prediction_R,4.),self.kappa)
        #         control=np.where( C < threshold )
        #         for m in control:
        #             dT[m]=0.
        #             C[m]=10.**(-21)
        #         #correction
        #         dV_sq, F_sq=self.calc_sq_flow(C,B,BT)
        #
        #         R_gamma=np.power(prediction_R,g_4m1)
        #         shear_sigma=np.power(np.multiply(dV_sq,R_gamma), sigma)
        #         diff_shear=np.multiply(np.subtract(shear_sigma,beta),prediction_R)
        #         correction_dR=np.multiply(diff_shear,dT)
        #
        #         #update
        #         dR=0.5*(prediction_dR+correction_dR)
        #         R=np.add(R,dR)
        #         K.C=np.multiply(np.power(R,4.),self.kappa)
        #         control=np.where( K.C < threshold )
        #         for m in control:
        #             dT[m]=0.
        #             K.C[m]=10.**(-21)
        #         C=K.C[:]
        #
        #
        #         # output
        #         if i % sample == 0:
        #             OUTPUT_C=np.vstack((OUTPUT_C,K.C) )
        #             OUTPUT_S=np.vstack((OUTPUT_S,K.J) )
        #             s=0
        #             K.clipp_graph()
        #             n=(1.+nx.number_of_edges(K.H)-nx.number_of_nodes(K.H))/(1.+nx.number_of_edges(K.G)-nx.number_of_nodes(K.G))
        #             nullity.append(n)
        #             D=np.divide(F_sq,K.C[:])
        #             dissipation.append(np.sum(D))
        #             A=np.power(R,2.)
        #             volume.append(np.sum(A))
        #
        #             dV_sq, F_sq=self.calc_sq_flow(C,B,BT)
        #             # K.V=np.sqrt(dV_sq)
        #             K.F=np.sqrt(F_sq)
        #
        #         self.print_step(i,Num_steps)
        #
        #     IO.save_kirchhoff_data(OUTPUT_C,OUTPUT_S,K)
        #     IO.save_nparray(nullity,'nullity_time')
        #     IO.save_nparray(dissipation,'dissipation_time')
        #     IO.save_nparray(volume,'volume_time')
        #

# class hu_cai(murray, object):
#     pass
    # def nsolve_euler_hucai(self,scale_data,parameters, K, IO):
    #
    #
    #     Num_steps=scale_data[0]
    #     dt=scale_data[1]
    #     sample=scale_data[2]
    #
    #     gamma=parameters[0]
    #     c0=parameters[1]
    #     c1=parameters[2]
    #
    #     OUTPUT_C, OUTPUT_S = IO.init_kirchhoff_data(scale_data,parameters,K)
    #
    #     #scale system
    #     g1=gamma+1
    #     alpha=(c0*K.f*K.l**3)/(K.k**g1)
    #     beta=(c1*gamma*K.k**g1)/(K.f*K.f)
    #
    #     #calc network dynamic
    #     B,BT=K.get_inicidence_matrices()
    #     M=K.G.number_of_edges()
    #     j=0
    #
    #     for i in range(Num_steps):
    #
    #         OP=np.dot(np.dot(B,K.C),BT)
    #         MP=lina.pinv(OP)
    #         K.V=MP.dot(K.J)
    #         K.F=np.dot(K.C,np.dot(BT,K.V))
    #
    #         for m in range(M):
    #             f_sq=K.F[m] * K.F[m]
    #             #calc energy
    #             if i % sample == 0:
    #                 K.E[j]+= ( ( f_sq ) / K.C[m,m] + (beta/gamma)*  K.C[m,m] ** gamma)
    #             #calc median
    #             cg=K.C[m,m]**g1
    #             dC= alpha * ( ( f_sq )/cg - beta ) * K.C[m,m] * dt
    #
    #             K.C[m,m] += dC
    #
    #         if i % sample == 0:
    #             OUTPUT_C=np.vstack((OUTPUT_C,np.diag(K.C)) )
    #             OUTPUT_S=np.vstack((OUTPUT_S,K.J) )
    #             j+=1
    #         self.print_step(i,Num_steps)
    #
    #     IO.save_kirchhoff_data(OUTPUT_C,OUTPUT_S,K)
    #
    # def nsolve_heun_hucai_reduced(self,scale_data,parameters , K, IO):
    #     # calculate steady state equivalent to hucai-model
    #     Num_steps=scale_data[0]
    #     dt=scale_data[1]
    #     sample=scale_data[2]
    #
    #     gamma=parameters[0]
    #     c0=parameters[1]
    #     c1=parameters[2]
    #
    #     eqv_factor=c1**((2.*gamma+1)/(gamma+1.))
    #     c0*=eqv_factor
    #     c1/=eqv_factor
    #
    #     OUTPUT_C, OUTPUT_S = IO.init_kirchhoff_data(scale_data,parameters,K)
    #
    #     #scale system
    #     g1=gamma+1
    #     reciproc_g1=1./g1
    #     alpha=(c0*K.f*K.l**3)/(K.k**g1)
    #     beta=(c1*gamma*K.k**g1)/(K.f*K.f)
    #
    #     #calc network dynamic
    #     B,BT=K.get_incidence_matrices()
    #     M=K.G.number_of_edges()
    #     dC=np.zeros(M)
    #     j=0
    #
    #     for i in range(Num_steps):
    #
    #         #1) prediction
    #         OP=np.dot(np.dot(B,np.diag(K.C)),BT)
    #         MP=lina.pinv(OP)
    #         K.V=MP.dot(K.J)
    #         K.F=np.dot(np.diag(K.C),np.dot(BT,K.V))
    #
    #         c_aux=np.copy(np.diag(K.C))
    #
    #         for m in range(M):
    #
    #             f_sq=K.F[m] * K.F[m]
    #             dC[m]= alpha*( ( f_sq )**reciproc_g1 - beta  * np.diag(K.C)[m,m] )* dt
    #             c_aux[m,m] += dC[m]
    #
    #         OP_aux=np.dot(np.dot(B,c_aux),BT)
    #         MP_aux=lina.pinv(OP_aux)
    #         V_aux=MP_aux.dot(K.J)
    #         F_aux=np.dot(c_aux,np.dot(BT,V_aux))
    #
    #         #2) correction
    #         for m in range(M):
    #             f_sq=K.F[m] * K.F[m]
    #             #calc energy
    #             if i % sample == 0:
    #                 K.E[j]+= ( ( f_sq ) / np.diag(K.C)[m,m] + (beta/(eqv_factor*gamma))*  np.diag(K.C)[m,m] ** gamma)
    #             dC_aux= alpha * ( ( F_aux[m] * F_aux[m] )**reciproc_g1 - beta  * c_aux[m,m]) * dt
    #             #calc median
    #             np.diag(K.C)[m,m] += ( ( dC[m]+ dC_aux)/2. )
    #
    #         if i % sample == 0:
    #             OUTPUT_C=np.vstack((OUTPUT_C,K.C) )
    #             OUTPUT_S=np.vstack((OUTPUT_S,K.J) )
    #             j+=1
    #         self.print_step(i,Num_steps)
    #
    #     IO.save_kirchhoff_data(OUTPUT_C,OUTPUT_S,K)
    # # @profile
    # def nsolve_heun_hucai(self,scale_data,parameters , K, IO):
    #
    #     Num_steps=scale_data[0]
    #     dt=scale_data[1]
    #     sample=scale_data[2]
    #
    #     gamma=parameters[0]
    #     c0=parameters[1]
    #     c1=parameters[2]
    #
    #     OUTPUT_C, OUTPUT_S = IO.init_kirchhoff_data(scale_data,parameters,K)
    #
    #     #scale system
    #     g1=gamma+1
    #     alpha=(c0*K.f*K.l**3)/(K.k**g1)
    #     beta=(c1*gamma*K.k**g1)/(K.f*K.f)
    #
    #     #calc network dynamic
    #     B,BT=K.get_incidence_matrices()
    #     M=nx.number_of_edges(K.G)
    #     dC=np.zeros(M)
    #     j=0
    #     # print(K.C)
    #     for i in range(Num_steps):
    #
    #         #1) prediction
    #         OP=np.dot(np.dot(B,np.diag(K.C)),BT)
    #         MP=lina.pinv(OP)
    #         K.V=MP.dot(K.J)
    #         K.F=np.multiply(K.C,np.dot(BT,K.V))
    #
    #         c_aux=np.diag(np.copy(K.C))
    #
    #         for m in range(M):
    #
    #             f_sq=K.F[m] * K.F[m]
    #             cg=K.C[m]**g1
    #
    #             dC[m]= alpha*( ( f_sq )/cg - beta ) * K.C[m] * dt
    #             c_aux[m,m] += dC[m]
    #
    #         OP_aux=np.dot(np.dot(B,c_aux),BT)
    #         MP_aux=lina.pinv(OP_aux)
    #         V_aux=MP_aux.dot(K.J)
    #         F_aux=np.dot(c_aux,np.dot(BT,V_aux))
    #
    #         #2) correction
    #         for m in range(M):
    #             f_sq=K.F[m] * K.F[m]
    #             #calc energy
    #             if i % sample == 0:
    #                 K.E[j]+= ( ( f_sq ) / K.C[m] + (beta/gamma)*  K.C[m] ** gamma)
    #             #calc median
    #             cg_aux=c_aux[m,m]**g1
    #             dC_aux= alpha * ( ( F_aux[m] * F_aux[m] )/cg_aux - beta ) * c_aux[m,m] * dt
    #
    #             K.C[m] += ( ( dC[m] + dC_aux)/2. )
    #
    #         if i % sample == 0:
    #             OUTPUT_C=np.vstack((OUTPUT_C,K.C[:]))
    #             OUTPUT_S=np.vstack((OUTPUT_S,K.J) )
    #             j+=1
    #         self.print_step(i,Num_steps)
    #
    #     IO.save_kirchhoff_data(OUTPUT_C,OUTPUT_S,K)
    #
    # def nsolve_rungekutta_hucai(self,scale_data,parameters , K, IO):
    #
    #     Num_steps=scale_data[0]
    #     dt=scale_data[1]
    #     dt2=0.5* dt
    #     sample=scale_data[2]
    #
    #     gamma=parameters[0]
    #     c0=parameters[1]
    #     c1=parameters[2]
    #
    #     OUTPUT_C, OUTPUT_S = IO.init_kirchhoff_data(scale_data,parameters,K)
    #
    #     #scale system
    #     g1=gamma+1
    #     alpha=(c0*K.f*K.l**3)/(K.k**g1)
    #     beta=(c1*gamma*K.k**g1)/(K.f*K.f)
    #
    #     #calc network dynamic
    #     B,BT=K.get_inicidence_matrices()
    #     M=K.G.number_of_edges()
    #
    #     k1=np.zeros(M)
    #     k2=np.zeros(M)
    #     k3=np.zeros(M)
    #     k4=np.zeros(M)
    #     j=0
    #
    #     for i in range(Num_steps):
    #
    #         OP=np.dot(np.dot(B,K.C),BT)
    #         MP=lina.pinv(OP)
    #         K.V=MP.dot(K.J)
    #         K.F=np.dot(K.C,np.dot(BT,K.V))
    #         c_aux1=np.copy(K.C)
    #
    #         #1) prediction k1
    #         for m in range(M):
    #
    #             f_sq=K.F[m] * K.F[m]
    #             cg=K.C[m,m]**g1
    #             k1[m]= alpha*( ( f_sq )/cg - beta ) * K.C[m,m]
    #             c_aux1[m,m] += k1[m]* dt2
    #             if i % sample == 0:
    #                 K.E[j]+= ( ( f_sq ) / K.C[m,m] + (beta/gamma)*  K.C[m,m] ** gamma)
    #
    #         #2) prediction k2
    #         OP_aux=np.dot(np.dot(B,c_aux1),BT)
    #         MP_aux=lina.pinv(OP_aux)
    #         V_aux=MP.dot(K.J)
    #         F_aux=np.dot(c_aux1,np.dot(BT,V_aux))
    #         c_aux2=np.copy(K.C)
    #
    #         for m in range(M):
    #             f_sq=F_aux[m] * F_aux[m]
    #             cg=c_aux1[m,m]**g1
    #             k2[m]= alpha*( ( f_sq )/cg - beta ) * c_aux1[m,m]
    #             c_aux2[m,m] += k2[m]*dt2
    #
    #         #3) prediction k3
    #         OP_aux=np.dot(np.dot(B,c_aux2),BT)
    #         MP_aux=lina.pinv(OP_aux)
    #         V_aux=MP.dot(K.J)
    #         F_aux=np.dot(c_aux2,np.dot(BT,V_aux))
    #         c_aux3=np.copy(K.C)
    #
    #         for m in range(M):
    #             f_sq=F_aux[m] * F_aux[m]
    #             cg=c_aux2[m,m]**g1
    #             k3[m]= alpha*( ( f_sq )/cg - beta ) * c_aux2[m,m]
    #             c_aux3[m,m] += k3[m]* dt
    #
    #         #3) prediction k4 and correction
    #         OP_aux=np.dot(np.dot(B,c_aux3),BT)
    #         MP_aux=lina.pinv(OP_aux)
    #         V_aux=MP.dot(K.J)
    #         F_aux=np.dot(c_aux3,np.dot(BT,V_aux))
    #         for m in range(M):
    #
    #             f_sq=F_aux[m] * F_aux[m]
    #             cg=c_aux3[m,m]**g1
    #             k4[m]= alpha*( ( f_sq )/cg - beta ) * c_aux3[m,m]
    #
    #             K.C[m,m] += (k1[m]+2*k2[m]+2*k3[m]+k4[m])*dt/6
    #
    #         if i % sample == 0:
    #             OUTPUT_C=np.vstack((OUTPUT_C,K.C))
    #             OUTPUT_S=np.vstack((OUTPUT_S,K.J) )
    #             j+=1
    #         self.print_step(i,Num_steps)
    #
    #     IO.save_kirchhoff_data(OUTPUT_C,OUTPUT_S,K)
    #
    # def nsolve_heun_hucai_R_tracked(self,scale_data,parameters, K, IO):
    #     K1=K
    #     nullity =[]
    #     shear = []
    #
    #     # rebranding conductivity Ki.C -> K_i
    #     kappa=1.
    #
    #     Num_steps=scale_data[0]
    #     dt=scale_data[1]
    #     sample=scale_data[2]
    #
    #     gamma=parameters[0]
    #     c0=parameters[1]
    #     c1=parameters[2]
    #
    #     OUTPUT_C1, OUTPUT_S1 = IO.init_kirchhoff_data(scale_data,parameters,K1)
    #
    #     #scale system
    #     g1=gamma-1
    #     alpha1=(c0*K1.f**g1)/((K1.k**gamma)*(K1.l**(3*g1)))
    #     beta1=(c1**K1.k*(K1.l**3)/K1.f)**gamma
    #
    #     #calc network dynamic
    #     B1,BT1=K1.get_incidence_matrices()
    #     M1=K1.G.number_of_edges()
    #     dC1=np.zeros(M1)
    #
    #     j=0
    #     # establish threshold for pruning events
    #     threshold=10.**(-20)
    #     for i in range(Num_steps):
    #
    #         # 1) prediction
    #
    #         K_1=np.diag( kappa* np.diag(np.copy(K1.C))**4 )
    #         OP1=np.dot(np.dot(B1,K_1),BT1)
    #         MP1=lina.pinv(OP1)
    #         K1.V=MP1.dot(K1.J)
    #         d_V1=np.dot(BT1,K1.V)
    #
    #         c_aux1=np.copy(K1.C)
    #         for m in range(M1):
    #             if K1.C[m,m] > threshold:
    #                 dC1[m]= (alpha1*( (np.fabs(d_V1[m])* K1.C[m,m])**gamma - beta1 ) * K1.C[m,m] )* dt
    #                 c_aux1[m,m] += dC1[m]
    #             else:
    #                 K1.C[m,m]=10.**(-21)
    #                 c_aux1[m,m] =10.**(-21)
    #
    #         #2) correction
    #         K_1_aux=np.diag( kappa* np.diag(np.copy(c_aux1))**4 )
    #
    #         OP1_aux=np.dot(np.dot(B1,K_1_aux),BT1)
    #         MP1_aux=lina.pinv(OP1_aux)
    #         V1_aux=MP1_aux.dot(K1.J)
    #         d_V1_aux=np.dot(BT1,V1_aux)
    #
    #         for m in range(M1):
    #             #calc median
    #             if K1.C[m,m] > threshold:
    #                 dC_aux1= (alpha1 * ( (np.fabs(d_V1_aux)[m] * c_aux1[m,m])**gamma - beta1 ) * c_aux1[m,m]  ) * dt
    #                 K1.C[m,m] += ( ( dC1[m] + dC_aux1)/2. )
    #
    #         if i % sample == 0:
    #             OUTPUT_C1=np.vstack((OUTPUT_C1,np.diag(K1.C)) )
    #             OUTPUT_S1=np.vstack((OUTPUT_S1,K1.J) )
    #             s1=0
    #             K1.clipp_graph()
    #             n1=(1.+nx.number_of_edges(K1.H)-nx.number_of_nodes(K1.H))/(1.+nx.number_of_edges(K1.G)-nx.number_of_nodes(K1.G))
    #
    #             for m in range(M1):
    #                 s1+=np.fabs(d_V1)[m]*OUTPUT_C1[-1,m]
    #
    #             nullity.append(n1)
    #             shear.append(s1)
    #
    #             j+=1
    #
    #         self.print_step(i,Num_steps)
    #
    #     IO.save_kirchhoff_data(OUTPUT_C1,OUTPUT_S1,K1)
    #     IO.save_nparray(nullity,'nullity_time')
    #     IO.save_nparray(shear,'shear_time')
    #
    # def nsolve_heun_chang_C(self,scale_data,parameters, K, IO):
    #
    #         nullity =[]
    #         cost = []
    #
    #         # rebranding conductivity Ki.C -> K_i
    #         kappa=1.
    #
    #         Num_steps=scale_data[0]
    #         dt=scale_data[1]
    #         sample=scale_data[2]
    #
    #         OUTPUT_C, OUTPUT_S = IO.init_kirchhoff_data(scale_data,parameters,K)
    #
    #         B,BT=K.get_incidence_matrices()
    #
    #         M=K.G.number_of_edges()
    #         Q_ref=np.ones(M)
    #
    #         # establish threshold for pruning events
    #
    #         for i in range(Num_steps):
    #             # predict
    #             K_C=np.diag(K.C)
    #             OP=np.dot(np.dot(B,K_C),BT)
    #             D=lina.pinv(OP)
    #             BD=np.dot(BT,D)
    #
    #             dV=np.dot(BD,K.J)
    #             stress=np.diag(np.multiply(K.C,dV))
    #             P=np.subtract(np.dot(K_C,np.dot(BD,B)),np.identity(M))
    #             Q=np.dot(K_C,dV)
    #             sign=np.divide(Q,np.absolute(Q))
    #             dQ=np.subtract(Q_ref,np.multiply(Q,sign))
    #
    #             dC_aux=np.dot(dQ,np.dot(P,stress))*dt
    #
    #             # correct
    #             K_C_aux=np.diag(np.add(K.C,dC_aux))
    #             OP=np.dot(np.dot(B,K_C_aux),BT)
    #             D=lina.pinv(OP)
    #             BD=np.dot(BT,D)
    #
    #             dV=np.dot(BD,K.J)
    #             stress=np.diag(np.multiply(np.diag(K_C_aux),dV))
    #             P=np.subtract(np.dot(K_C_aux,np.dot(BD,B)),np.identity(M))
    #             Q=np.dot(K_C_aux,dV)
    #             sign=np.divide(Q,np.absolute(Q))
    #             dQ=np.subtract(Q_ref,np.multiply(Q,sign))
    #
    #             dC=np.dot(dQ,np.dot(P,stress))*dt
    #
    #             #final increment
    #             K.C=np.add(K.C,np.add(dC,dC_aux)/2.)
    #
    #             if i % sample == 0:
    #                 OUTPUT_C=np.vstack((OUTPUT_C,K.C) )
    #                 OUTPUT_S=np.vstack((OUTPUT_S,K.J) )
    #
    #                 K.clipp_graph()
    #                 n=(1.+nx.number_of_edges(K.H)-nx.number_of_nodes(K.H))/(1.+nx.number_of_edges(K.G)-nx.number_of_nodes(K.G))
    #                 s=np.dot(dQ,dQ)
    #
    #                 nullity.append(n)
    #                 cost.append(s)
    #             self.print_step(i,Num_steps)
    #
    #         IO.save_kirchhoff_data(OUTPUT_C,OUTPUT_S,K)
    #         IO.save_nparray(nullity,'nullity_time')
    #         IO.save_nparray(cost,'cost_time')
    #
    # def nsolve_heun_hucai_oscillation(self,scale_data,parameters , K, IO):
    #
    #     Num_steps=scale_data[0]
    #     dt=scale_data[1]
    #     sample=scale_data[2]
    #     nullity=[]
    #     c_mean=[]
    #     gamma=parameters[0]
    #     scale=parameters[1]
    #     volume_penalty=parameters[2]
    #     mode=parameters[3]
    #
    #     OUTPUT_C, OUTPUT_S = IO.init_kirchhoff_data(scale_data,parameters,K)
    #
    #     #scale system
    #     g1_p=gamma+1
    #     g1_m=1.-gamma
    #
    #     #calc network dynamic
    #     B,BT=K.get_incidence_matrices()
    #     M=nx.number_of_edges(K.G)
    #     N=nx.number_of_nodes(K.G)
    #     dC=np.zeros(M)
    #     j=0
    #     for i in range(Num_steps):
    #         S=self.propagate_sources(mode,K,i*dt)
    #
    #         #1) prediction
    #         v_sq=self.calc_sq_pressure(K.C,B,BT,S)
    #         cg_m=np.power(K.C,g1_m)
    #         stress_volume=np.subtract( np.multiply(v_sq,cg_m) , volume_penalty )
    #         dC= np.multiply( scale * dt, np.multiply(K.C, stress_volume  ))
    #         c_aux= np.add(K.C,dC)
    #
    #         #2) correction
    #
    #         S_aux=self.propagate_sources(mode,K,(i+1)*dt)
    #
    #         v_sq_aux=self.calc_sq_pressure(c_aux,B,BT,S_aux)
    #         cg_aux=np.power(c_aux,g1_m)
    #         stress_volume=np.subtract( np.multiply(v_sq_aux,cg_aux) , volume_penalty )
    #         dC_aux= np.multiply( scale * dt, np.multiply(c_aux, stress_volume  ))
    #
    #         K.C = np.add( K.C, np.divide(np.add( dC , dC_aux),2.) )
    #         if i % sample == 0:
    #             # K.E[j] = np.add( np.sum(np.multiply( v_sq_aux, K.C)) ,  * np.sum( np.power(K.C, gamma)))
    #             OUTPUT_C = np.vstack((OUTPUT_C,K.C[:]))
    #             OUTPUT_S = np.vstack((OUTPUT_S,S) )
    #             # j+=1
    #
    #             K.clipp_graph()
    #             H=nx.Graph(K.H)
    #             n=(nx.number_connected_components(H)+nx.number_of_edges(H)-nx.number_of_nodes(H))/(1.+M-N)
    #             nullity.append(n)
    #             c_mean.append(np.mean(K.H_C))
    #
    #         self.print_step(i,Num_steps)
    #
    #     IO.save_kirchhoff_data(OUTPUT_C,OUTPUT_S,K)
    #     IO.save_nparray(nullity,'nullity_time')
    #     IO.save_nparray(c_mean,'conductivity_mean_time')
    #
    # def nsolve_heun_hucai_oscillation_temporal_avg(self,scale_data,parameters , K, IO):
    #
    #     Num_steps=scale_data[0]
    #     dt=scale_data[1]
    #     sample=scale_data[2]
    #     nullity=[]
    #     c_mean=[]
    #     gamma=parameters[0]
    #     scale=parameters[1]
    #     volume_penalty=parameters[2]
    #     mode=parameters[3]
    #
    #     OUTPUT_C, OUTPUT_S = IO.init_kirchhoff_data(scale_data,parameters,K)
    #
    #     #scale system
    #     g1_p=gamma+1
    #     g1_m=1.-gamma
    #
    #     #calc network dynamic
    #     B,BT=K.get_incidence_matrices()
    #     M=nx.number_of_edges(K.G)
    #     N=nx.number_of_nodes(K.G)
    #     dC=np.zeros(M)
    #
    #     idx=np.where(K.J < 0)[0]
    #     x=np.where(K.J > 0)[0][0]
    #     # j=0
    #     for i in range(Num_steps):
    #
    #         #1) prediction
    #         v_sq=self.calc_sq_pressure_temporal_mean(mode,[idx,x],K.C,i*dt,B,BT)
    #         cg_m=np.power(K.C,g1_m)
    #         stress_volume=np.subtract( np.multiply(v_sq,cg_m) , volume_penalty )
    #         dC= np.multiply( scale * dt, np.multiply(K.C, stress_volume  ))
    #         c_aux= np.add(K.C,dC)
    #
    #         #2) correction
    #         v_sq_aux=self.calc_sq_pressure_temporal_mean(mode,[idx,x],c_aux,(i+1)*dt,B,BT)
    #         cg_aux=np.power(c_aux,g1_m)
    #         stress_volume=np.subtract( np.multiply(v_sq_aux,cg_aux) , volume_penalty )
    #         dC_aux= np.multiply( scale * dt, np.multiply(c_aux, stress_volume  ))
    #
    #         K.C = np.add( K.C, np.divide(np.add( dC , dC_aux),2.) )
    #         if i % sample == 0:
    #             # K.E[j] = np.add( np.sum(np.multiply( v_sq_aux, K.C)) ,  * np.sum( np.power(K.C, gamma)))
    #             S=np.zeros(N)
    #             OUTPUT_C = np.vstack((OUTPUT_C,K.C[:]))
    #             OUTPUT_S = np.vstack((OUTPUT_S,S) )
    #             # j+=1
    #
    #             K.clipp_graph()
    #             H=nx.Graph(K.H)
    #             n=(nx.number_connected_components(H)+nx.number_of_edges(H)-nx.number_of_nodes(H))/(1.+M-N)
    #             nullity.append(n)
    #             c_mean.append(np.mean(K.H_C))
    #
    #         self.print_step(i,Num_steps)
    #
    #     IO.save_kirchhoff_data(OUTPUT_C,OUTPUT_S,K)
    #     IO.save_nparray(nullity,'nullity_time')
    #     IO.save_nparray(c_mean,'conductivity_mean_time')
    #
    # def nsolve_heun_hucai_fluctuation(self,scale_data,parameters , K, IO):
    #
    #     Num_steps=scale_data[0]
    #     dt=scale_data[1]
    #     sample=scale_data[2]
    #     nullity=[]
    #     gamma=parameters[0]
    #     scale=parameters[1]
    #     volume_penalty=parameters[2]
    #     coupling=[parameters[3],parameters[4]]
    #     OUTPUT_C, OUTPUT_S = IO.init_kirchhoff_data(scale_data,parameters,K)
    #
    #     #scale system
    #     g1_p=gamma+1
    #     g1_m=1.-gamma
    #
    #     #calc network dynamic
    #     B,BT=K.get_incidence_matrices()
    #     M=nx.number_of_edges(K.G)
    #     N=nx.number_of_nodes(K.G)
    #     dC=np.zeros(M)
    #     j=0
    #     idx=np.where(K.J < 0)[0]
    #     x=np.where(K.J > 0)[0][0]
    #     for i in range(Num_steps):
    #
    #         #1) prediction
    #         v_sq=self.calc_sq_pressure_general(K.C,B,BT,[idx,x],coupling)
    #         cg_m=np.power(K.C,g1_m)
    #         stress_volume=np.subtract( np.multiply(v_sq,cg_m) , volume_penalty )
    #         dC= np.multiply( scale * dt, np.multiply( K.C, stress_volume  ))
    #         c_aux= np.add(K.C,dC)
    #
    #         #2) correction
    #         v_sq_aux=self.calc_sq_pressure_general(c_aux,B,BT,[idx,x],coupling)
    #         cg_aux=np.power(c_aux,g1_m)
    #         stress_volume=np.subtract( np.multiply(v_sq_aux,cg_aux) , volume_penalty )
    #         dC_aux= np.multiply( scale * dt, np.multiply(c_aux, stress_volume  ))
    #
    #         K.C = np.add( K.C, np.divide(np.add( dC , dC_aux),2.) )
    #         if i % sample == 0:
    #             K.E[j] = np.add( np.sum(np.multiply( v_sq_aux, K.C)) ,  np.sum( np.power(K.C, gamma)))
    #             OUTPUT_C = np.vstack((OUTPUT_C,K.C[:]))
    #             S=np.zeros(N)
    #             OUTPUT_S = np.vstack((OUTPUT_S,S) )
    #             j+=1
    #
    #             K.clipp_graph()
    #             H=nx.Graph(K.H)
    #             n=(nx.number_connected_components(H)+nx.number_of_edges(H)-nx.number_of_nodes(H))/(1.+M-N)
    #             nullity.append(n)
    #
    #         self.print_step(i,Num_steps)
    #
    #     IO.save_kirchhoff_data(OUTPUT_C,OUTPUT_S,K)
    #     IO.save_nparray(nullity,'nullity_time')

class bohn_katifori(murray, object):

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

class meigel_alim(murray, object):

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

    def solve_absorption(self,t,R,K):

        K.C=np.power(R,4)*K.k
        K.R=R

        dr,PE=self.calc_absorption_dr(K)
        self.save_dynamic_output([R,PE],['radii','PE'],K)

        return dr

    def calc_absorption_dr(self,K):

        dr=np.zeros(self.M)
        c,B_new,K=self.calc_profile_concentration(K)

        phi=self.calc_absorption(K.R, K)
        J_phi=self.calc_absorption_jacobian(K.R, K)
        phi0=np.ones(self.M)*K.phi0

        for i in range(self.M):
            dr[i]=-2.*np.sum( np.multiply(np.subtract(phi,phi0),J_phi[i,:]))

        return dr,K.PE

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
class kramer_modes(meigel_alim,murray,object):

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
