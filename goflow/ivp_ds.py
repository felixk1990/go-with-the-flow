# @Author: Felix Kramer <kramer>
# @Date:   23-06-2021
# @Email:  kramer@mpi-cbg.de
# @Project: phd_network_remodelling
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-06-24T11:10:24+02:00

import scipy.integrate as si
import goflow.init_ivp as gi

class morph_ds( gi.morph, object ):

    def __init__(self, flow):

        super(morph_ds,self).__init__(flow)

    def propagate_ds(self,ds_func,t_span,x0, **kwargs):

        self.options={
            'method':'LSODA',
            'args':( [self.flow] )
        }
        for k,v in kwargs.items():
            if k in options:
                self.options[k]=v

        nsol=si.solve_ivp(ds_func, t_span ,x0 , **self.options)

        return nsol

    def propagate_ds_fw_euler(self,ds_func,x0, **kwargs):

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

    def solve_fw_euler(self,ds_func, x0 , **kwargs):

        t_samples=np.arrange(0,kwargs['num_steps'],step=kwargs['sample_rate'],endpoint=True)*kwargs['step']
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
