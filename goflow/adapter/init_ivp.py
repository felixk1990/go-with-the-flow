# @Author: Felix Kramer <kramer>
# @Date:   23-06-2021
# @Email:  kramer@mpi-cbg.de
# @Project: phd_network_remodelling
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-09-08T20: 54: 23+02: 00

import numpy as np
import scipy.optimize as sc
import scipy.integrate as si
from ..models import base, binder
from dataclasses import dataclass, field
# general initial value problem for network morpgogenesis


@dataclass
class proxy_solver():

    defVal = dict(default_factory=list, init=True, repr=False)
    t_samples: list = field(**defVal)
    sol: list = field(**defVal)

    def __post_init__(self):

        self.t = self.t_samples
        self.y = self.sol.transpose()

@dataclass
class morph():

    defVal = dict(init=True, repr=True)

    constrct: str = field(**defVal)
    mode: str = field(default='default', **defVal)
    args: tuple = field(default_factory=tuple(),**defVal)

    def __post_init__(self):

        self.init_model_and_flow()
        self.link_model_flow()

    def link_model_flow(self):

        self.model.solver_options.update({'args': (self.flow, *self.model.model_args) })
        self.model.update_event_func()

    def init_model_and_flow(self):

        if self.mode in binder.modelBinder:

            self.model = binder.modelBinder[self.mode](self.args[0])
            self.flow = binder.circuitBinder[self.mode](self.constrct, *self.args[1:])

        elif isinstance(self.mode, base.model):

            self.flow = self.contrct
            self.model = self.mode

@dataclass
class morph_dynamic(morph):

    def __post_init__(self):
        self.init_model_and_flow()
        self.link_model_flow()

    def autoSolve(self, t_span, x0):

        self.options = {
            # 'method': 'RK45',
            'method': 'LSODA',
            # 'method': 'BDF',
            'atol': 1e-10,
            'rtol': 1e-7,
            # 'dense_output': True,
        }

        self.evals = 100

        for k, v in self.model.solver_options.items():
            self.options[k] = v
        self.options.update({
            't_eval': np.linspace(t_span[0], t_span[1], num=self.evals)
        })

        ds_func = self.model.calc_update_stimuli

        nsol = si.solve_ivp(ds_func, t_span, x0, **self.options)

        return nsol

    def nsolve(self, ds_func, t_span, x0, **kwargs):

        self.options = {
            'method': 'LSODA',
            'atol': 1e-10,
            'dense_output': True,
        }
        for k, v in kwargs.items():
            self.options[k] = v

        self.evals = 100
        self.options.update({
            't_eval': np.linspace(t_span[0], t_span[1], num=self.evals)
        })

        nsol = si.solve_ivp(ds_func, t_span, x0, **self.options)

        return nsol

    def nsolve_custom(self, ds_func, x0, **kwargs):

        self.options = {
            'num_steps': 1,
            'samples': 1,
            'step': 1,
        }
        for k, v in kwargs.items():
            self.options[k] = v

        ns, sr  = self.set_integration_scale(self.options['num_steps'], self.options['samples'])
        self.options['sample_rate'] = sr
        self.options['num_steps'] = ns

        nsol = self.nsolve_fw_euler(ds_func, x0, **self.options)

        return nsol

    def nsolve_fw_euler(self, ds_func, x0, **kwargs):

        t_samples = np.arange(0, kwargs['num_steps'], step=kwargs['sample_rate'])*kwargs['step']
        sol = np.zeros((kwargs['samples'], len(x0)))
        c_m = 0
        x_0 = np.array(x0)

        for i in range(kwargs['num_steps']):

            # try:

                if (i % kwargs['sample_rate'] ) == 0:
                    sol[c_m] = x_0[: ]
                    c_m += 1

                dx = ds_func(i*kwargs['step'], x_0, *kwargs['args'])
                x_0 = np.add(x_0, dx*kwargs['step'])

            # except:
            #     print('Warning: Ending integration due to bad numerics....find out more at ...')
            #     break

        nsol = proxy_solver(t_samples, sol)

        return nsol

    def set_integration_scale(self, Num_steps, sample):

        #reshape number of integration steps & sample rates for consistency
        sample_rate = int(Num_steps/sample)
        if (sample_rate*sample) < Num_steps:
            Num_steps = sample_rate*sample

        return Num_steps, sample_rate

# @dataclass
# class morph():
#
#     defVal = dict(default='unknown',init=True, repr=True)
#     flow: str = field(**defVal)
#     model: str = field(**defVal)
#
#     def __post_init__(self):
#
#         self.link_model_flow()
#
#     def link_model_flow(self):
#
#         self.model.solver_options.update({'args': (self.flow, *self.model.model_args) })
#         self.model.update_event_func()
#
# @dataclass
# class morph_dynamic(morph):
#
#     def __post_init__(self):
#
#         self.link_model_flow()
#
#     def nsolve(self, ds_func, t_span, x0, **kwargs):
#
#         self.options = {
#             'method': 'LSODA',
#             'atol': 1e-10,
#             'dense_output': True,
#         }
#         for k, v in kwargs.items():
#             self.options[k] = v
#
#         self.evals = 100
#         self.options.update({
#             't_eval': np.linspace(t_span[0], t_span[1], num=self.evals)
#         })
#
#         nsol = si.solve_ivp(ds_func, t_span, x0, **self.options)
#
#         return nsol
#
#     def nsolve_custom(self, ds_func, x0, **kwargs):
#
#         self.options = {
#             'num_steps': 1,
#             'samples': 1,
#             'step': 1,
#         }
#         for k, v in kwargs.items():
#             self.options[k] = v
#
#         ns, sr  = self.set_integration_scale(self.options['num_steps'], self.options['samples'])
#         self.options['sample_rate'] = sr
#         self.options['num_steps'] = ns
#
#         nsol = self.nsolve_fw_euler(ds_func, x0, **self.options)
#
#         return nsol
#
#     def nsolve_fw_euler(self, ds_func, x0, **kwargs):
#
#         t_samples = np.arange(0, kwargs['num_steps'], step=kwargs['sample_rate'])*kwargs['step']
#         sol = np.zeros((kwargs['samples'], len(x0)))
#         c_m = 0
#         x_0 = np.array(x0)
#
#         for i in range(kwargs['num_steps']):
#
#             # try:
#
#                 if (i % kwargs['sample_rate'] ) == 0:
#                     sol[c_m] = x_0[: ]
#                     c_m += 1
#
#                 dx = ds_func(i*kwargs['step'], x_0, *kwargs['args'])
#                 x_0 = np.add(x_0, dx*kwargs['step'])
#
#             # except:
#             #     print('Warning: Ending integration due to bad numerics....find out more at ...')
#             #     break
#
#         nsol = proxy_solver(t_samples, sol)
#
#         return nsol
#
#     def set_integration_scale(self, Num_steps, sample):
#
#         #reshape number of integration steps & sample rates for consistency
#         sample_rate = int(Num_steps/sample)
#         if (sample_rate*sample) < Num_steps:
#             Num_steps = sample_rate*sample
#
#         return Num_steps, sample_rate

@dataclass
class MySteps():

    stepsize: float = 0.
    # def __init__(self, stepsize ):
    #     self.stepsize  = stepsize
    def __call__(self, x):
        rx = np.add(x, np.random.rand(len(x))*self.stepsize)
        return rx

@dataclass
class morph_optimize(morph):

    def __post_init__(self):

        mysteps = MySteps(1.)
        b0 = 1e-25
        self.options = {
            'step': mysteps,
            'niter': 100,
            'T': 10.,
            'minimizer_kwargs': {
                'method': 'L-BFGS-B',
                'bounds': [(b0, None) for x in range(len(self.flow.circuit.list_graph_edges))],
                'args': (self.flow.circuit),
                'jac': False,
                'tol': 1e-10
                }
        }

    # def __init__(self, flow):
    #
    #     super(morph_optimize, self).__init__(flow)
    #
    #     mysteps = MySteps(1.)
    #     b0 = 1e-25
    #     self.options = {
    #         'step': mysteps,
    #         'niter': 100,
    #         'T': 10.,
    #         'minimizer_kwargs': {
    #             'method': 'L-BFGS-B',
    #             'bounds': [(b0, None) for x in range(len(self.flow.circuit.list_graph_edges))],
    #             'args': (self.flow.circuit),
    #             'jac': False,
    #             'tol': 1e-10
    #             }
    #     }

    def update_minimizer_options(**kwargs):

        if 'step' in kwargs:
            mysteps = MySteps(kwargs['step'])
            kwargs['step'] = mysteps

        for k, v in kwargs.items():
            if k in self.options:
                options[k] = v

        if 'minimizer_kwargs' in kwargs:
            for ks, vs in kwargs['minimizer_kwargs']:
                minimizer_kwargs[ks] = vs

    def optimize_network(self, cost_func, x0, **kwargs):

        update_minimizer_options(**kwargs)

        sol = sc.basinhopping(cost_func, x0, **self.options)

        return sol
