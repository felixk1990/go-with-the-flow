# @Author: Felix Kramer <kramer>
# @Date:   23-06-2021
# @Email:  kramer@mpi-cbg.de
# @Project: phd_network_remodelling
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-06-24T11:12:59+02:00

import numpy as np
import scipy.optimize as sc

# custom
import goflow.init_ivp as gi

class MySteps(object):
    def __init__(self, stepsize ):
        self.stepsize = stepsize
    def __call__(self, x):
        rx=np.add(x,np.random.rand(len(x))*self.stepsize)
        return rx

class morph_opt( gi.morph, object):

    def __init__(self,flow):

        super(morph_opt,self).__init__(flow)

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
