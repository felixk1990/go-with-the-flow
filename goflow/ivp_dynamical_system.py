# @Author: Felix Kramer <kramer>
# @Date:   23-06-2021
# @Email:  kramer@mpi-cbg.de
# @Project: phd_network_remodelling
# @Last modified by:   kramer
# @Last modified time: 23-06-2021

import scipy.integrate as si
import goflow.init_ivp as gi

class morph_dynamical_system( gi.morph_ivp, object ):

    def __init__(self, flow):

        super(morph_dynamical_system,self).__init__(flow)

        self.options={
            'method':'LSODA',
            'args':( [self.flow] )
        }

    def propagate_dynamic_system(self,dynsys_func,t_span,x0, **kwargs):

        for k,v in kwargs.items():
            if k in options:
                self.options[k]=v

        nsol=si.solve_ivp(dynsys_func, t_span ,x0 , **self.options)

    return nsol
