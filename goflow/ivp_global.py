# @Author: Felix Kramer <kramer>
# @Date:   23-06-2021
# @Email:  kramer@mpi-cbg.de
# @Project: phd_network_remodelling
# @Last modified by:   kramer
# @Last modified time: 23-06-2021

import numpy as np
import scipy.linalg as lina
import sys
import scipy as sy
import scipy.integrate as si
import scipy.optimize as sc
import networkx as nx
import functions_template as ft

import goflow.init_ivp as gi

class MySteps(object):
    def __init__(self, stepsize ):
        self.stepsize = stepsize
    def __call__(self, x):
        # print(x)
        rx=np.add(x,np.random.rand(len(x))*self.stepsize)
        return rx

class morph_global( gi.morph_ivp, object):

    def __init__(self,flow):

        super(morph_global,self).__init__(flow)
        
        mysteps=MySteps(1.)
        b0=1e-25
        self.options={
            'take_step'=mysteps,
            'niter'=100,
            'T'=10.,
            'minimizer_kwargs':{
                'method':'L-BFGS-B',
                'bounds':[(b0,None) for x in range(nx.number_of_edges(self.flow.circuit.G))],
                'args':(self.flow.circuit),
                'jac':True,
                'tol':1e-10
                }
        }

    def update_minimizer_options(**kwargs):

        if 'take_step' in kwargs:
            mysteps=MySteps(kwargs['take_step'])
            kwargs['take_step']=mysteps

        for k,v in kwargs.items():
            if k in self.options:
                options[k]=v

        if 'minimizer_kwargs' in kwargs:
            for ks,vs in kwargs['minimizer_kwargs']:
                minimizer_kwargs[ks]=vs

    def optimize_network_targets(self,cost_func,x0 **kwargs):

        update_minimizer_options(**kwargs)

        sol=sc.basinhopping(cost_func,x0,**self.options)

        return sol
