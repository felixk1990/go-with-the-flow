# @Author: Felix Kramer <kramer>
# @Date:   23-06-2021
# @Email:  kramer@mpi-cbg.de
# @Project: phd_network_remodelling
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-09-04T23:45:10+02:00

import numpy as np
import scipy.optimize as sc
import scipy.integrate as si
# general initial value problem for network morpgogenesis

class morph(  ):

    def __init__(self, **kwargs):

        c1=False
        c2=False
        if 'flow' in kwargs:
            self.flow=kwargs['flow']
            c1=True
        if 'model' in kwargs:
            self.model=kwargs['model']
            c2=True

        if c1 and c2 :
            self.link_model_flow()

    def link_model_flow(self):
        self.model.solver_options.update({'args': (self.flow,self.model.model_args[0],self.model.model_args[1]) })
        self.model.update_event_func()

class morph_dynamic( morph, object ):

    def __init__(self,**kwargs):

        super(morph_dynamic,self).__init__(**kwargs)

    def nsolve(self,ds_func,t_span,x0, **kwargs):

        self.options={
            'method':'LSODA',
            'atol':1e-10,

        }
        for k,v in kwargs.items():
            self.options[k]=v

        self.evals=100
        self.options.update({
            't_eval':np.linspace(t_span[0],t_span[1],num=self.evals)
        })

        nsol=si.solve_ivp(ds_func, t_span ,x0 , **self.options)

        return nsol

    def nsolve_fw_euler(self,ds_func,x0, **kwargs):

        self.options={
            'num_steps':1,
            'samples':1,
            'step':1,
            'args': [self.flow]
        }
        for k,v in kwargs.items():
            if k in options:
                self.options[k]=v

        ns,sr = self.set_integration_scale(self.options['num_steps',self.options['samples']])
        self.options['sample_rate']=sr
        self.options['num_steps']=ns

        nsol=solve_fw_euler(ds_func ,x0 , **self.options)

        return nsol

    def nsolve_fw_euler(self,ds_func, x0 , **kwargs):

        t_samples=np.arange(0,kwargs['num_steps'],step=kwargs['sample_rate'])*kwargs['step']
        sol=np.zeros((kwargs['samples'],len(x0)))
        c_m=0
        x_0=np.array(x0)

        for i in range(kwargs['num_steps']):

            try:

                if (i % kwargs['sample_rate'] )==0:
                    sol[c_m]=x_0[:]
                    c_m+=1

                dx=ds_func(i*kwargs['step'],x_0,args=kwargs['args'])
                x_0=np.add(x_0,dx*kwargs['step'])

            except:
                print('Warning: Ending integration due to bad numerics....find out more at ...')
                break

        return [t_sample,sol]

    def set_integration_scale(self,Num_steps,sample):

        #reshape number of integration steps & sample rates for consistency
        sample_rate=int(Num_steps/sample)
        if (sample_rate*sample) < Num_steps:
            Num_steps=sample_rate*sample

        return Num_steps,sample_rate

class MySteps(object):
    def __init__(self, stepsize ):
        self.stepsize = stepsize
    def __call__(self, x):
        rx=np.add(x,np.random.rand(len(x))*self.stepsize)
        return rx

class morph_optimize( morph, object):

    def __init__(self,flow):

        super(morph_optimize,self).__init__(flow)

        mysteps=MySteps(1.)
        b0=1e-25
        self.options={
            'step':mysteps,
            'niter':100,
            'T':10.,
            'minimizer_kwargs':{
                'method':'L-BFGS-B',
                'bounds':[(b0,None) for x in range(len(self.flow.circuit.list_graph_edges))],
                'args':(self.flow.circuit),
                'jac':True,
                'tol':1e-10
                }
        }

    def update_minimizer_options(**kwargs):

        if 'step' in kwargs:
            mysteps=MySteps(kwargs['step'])
            kwargs['step']=mysteps

        for k,v in kwargs.items():
            if k in self.options:
                options[k]=v

        if 'minimizer_kwargs' in kwargs:
            for ks,vs in kwargs['minimizer_kwargs']:
                minimizer_kwargs[ks]=vs

    def optimize_network(self,cost_func,x0, **kwargs):

        update_minimizer_options(**kwargs)

        sol=sc.basinhopping(cost_func,x0,**self.options)

        return sol
