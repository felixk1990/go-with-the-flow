# @Author: Felix Kramer <felix>
# @Date:   2022-06-29T18:27:48+02:00
# @Email:  felixuwekramer@proton.me
# @Filename: init_ivp.py
# @Last modified by:   felix
# @Last modified time: 2022-07-01T16:32:42+02:00

import sys
import numpy as np
import networkx as nx
# import scipy.optimize as sc
import scipy.integrate as si
from ..models import base, binder
from dataclasses import dataclass, field
# general initial value problem for network morpgogenesis


@dataclass
class proxy_solver():
    """
    A proxy solver class for custom numeric integration.

    Args:
        tsamples (array):\n
         Setting the internal sample time points.
        \n
        sol (array):\n
         Time series of data to be stored internally.
    Returns:
        type: proxy_solver
    """

    defVal = dict(default_factory=list, init=True, repr=False)
    t_samples: list = field(**defVal)
    sol: list = field(**defVal)

    def __post_init__(self):

        self.t = self.t_samples
        self.y = self.sol.transpose()


@dataclass
class morph():
    """
    Basic Class for simulating flow network adaption dynamics.

    'Morph' computes the long-term network development on the basis of
    flow and gradient descent. The class acts as a wrapper to ensure correct
    pre-/post-proccessing of the network's data and output. Utilizing the
    modules 'kirchhoff' and 'hailhydro' as well as goflow.models.

    Args:
        construct (nx.Graph):\n
         A network plexus, recommended to be 'kirchhoff' circuit or an
         initialized 'hailhydro' flow/flux. Handling for deviations only
         provided for standard models.\n
        mode (string/model):\n
         A descriptive string for selection internal model object.\n
        args (tuple):\n
         A parameter tuple for evaluation of the model given a plexus.\n
    Returns:
        type: morph
    """

    defVal = dict(init=True, repr=True)

    construct: nx.Graph = field(**defVal)
    mode: str = field(default='default', **defVal)
    args: tuple = field(default_factory=tuple(), **defVal)

    def __post_init__(self):
        """
        Post object initialization methods.

        Streamline provided model and flow parameters and set dynamics of the
        ODE system.

        Args:
            None

        Returns:
            None
        """

        self.init_model_and_flow()
        self.link_model_flow()

    def link_model_flow(self):
        """
        Transfer model and flow parameters to internal variables of solver.

        Args:
            None

        Returns:
            None
        """

        args = dict(args=(self.flow, *self.model.model_args))
        self.model.solver_options.update(args)
        self.model.update_event_func()

    def init_model_and_flow(self):
        """
        Set internal model and flow variables with constructor values.

        Args:
            None

        Returns:
            None

        Raises:
            Exception: \n
             sys.exit('Terminate Program: No valid adaptation model provided.')

        """

        if isinstance(self.mode, base.model):

            self.flow = self.construct
            self.model = self.mode

        elif self.mode in binder.modelBinder:

            model = binder.modelBinder[self.mode]
            flow = binder.circuitBinder[self.mode]

            self.model = model(self.args[0])
            self.flow = flow(self.construct, *self.args[1:])

        else:
            sys.exit('Terminate Program: No valid adaptation model provided.')


@dataclass
class morph_dynamic(morph):
    """
    Derived class definition from morph, for specific dynamic models.

    Defines explicit ODE solvers and customizable forward integration solvers.

    Args:
        construct (nx.Graph):\n
         A network plexus, recommended to be 'kirchhoff' circuit or an
         initialized 'hailhydro' flow/flux. Handling for deviations only
         provided for standard models.\n
        mode (string/model):\n
         A descriptive string for selection internal model object.\n
        args (tuple):\n
         A parameter tuple for evaluation of the model given a plexus.\n
    Returns:
        type: morph_dynamic

    """

    def __post_init__(self):
        """
        Post object initialization methods.

        Streamline provided model and flow parameters and set dynamics of the
        ODE system.

        Args:
            None

        Returns:
            None
        """
        self.evals = 100
        self.init_model_and_flow()
        self.link_model_flow()

    def autoSolve(self, t_span, x0):
        """
        Default ODE solver/wrapper for given adaptation models.

        Based on methods of scipy.integrate, hence solver options are
        customizable via the internal model.solver_options attribute and align
        with scipy standards.

        Args:
            t_span (tuple):\n
             A tuple setting the linear range of values for
             explicit evaluation.\n
            x0 (array):\n
             An array of initial values to start the numeric integration of the
             dynamic system.\n
        Returns:
            Iterable: Same set of return values as scipy.integrate.solve_ivp
            (1.7.3)

        """

        self.options = {
            # 'method': 'RK45',
            'method': 'LSODA',
            # 'method': 'BDF',
            'atol': 1e-10,
            'rtol': 1e-7,
            # 'dense_output': True,
        }

        for k, v in self.model.solver_options.items():
            self.options[k] = v
        self.options.update({
            't_eval': np.linspace(t_span[0], t_span[1], num=self.evals)
        })

        ds_func = self.model.calc_update_stimuli

        nsol = si.solve_ivp(ds_func, t_span, x0, **self.options)

        return nsol

    def nlogSolve(self, t_span, x0):
        """
        Custom ODE solver/wrapper for given adaptation models with logarithmic
        evaluation timescales.

        Based on methods of scipy.integrate, hence solver options are
        customizable via the internal model.solver_options attribute and align
        with scipy standards.

        Args:
            t_span (tuple):\n
             A tuple setting the logarithmic range of values for
             explicit evaluation.\n
            x0 (array):\n
             An array of initial values to start the numeric
             integration of the dynamic system.\n
        Returns:
            Iterable: Same set of return values as scipy.integrate.solve_ivp
            (1.7.3)

        """

        self.options = {
            # 'method': 'RK45',
            'method': 'LSODA',
            # 'method': 'BDF',
            'atol': 1e-10,
            'rtol': 1e-7,
            # 'dense_output': True,
        }

        for k, v in self.model.solver_options.items():
            self.options[k] = v
        self.options.update({
            't_eval': np.logspace(
                        np.log10(t_span[0]),
                        np.log10(t_span[1]),
                        num=self.evals
                        )
        })

        ds_func = self.model.calc_update_stimuli

        nsol = si.solve_ivp(ds_func, t_span, x0, **self.options)

        return nsol

    def nsolve(self, ds_func, t_span, x0, **kwargs):
        """
        Custom ODE solver/wrapper for given adaptation models with linear
        evaluation timescales and custom update function. Based on methods of
        scipy.integrate.

        Args:
            ds_func (callable function):\n
             The update function of the dynamic system, which is supposed to
             define dx/dt, according to scipy standards.\n
            t_span (tuple):\n
             A tuple setting the logarithmic range of values for
             explicit evaluation.\n
            x0 (array):\n
             An array of initial values to start the numeric
             integration of the dynamic system.\n
            kwargs (dict):\n
             Dictionary for setting solver options according to
             scipy.integrate.solve_ivp standards.\n
        Returns:
            Iterable: Same set of return values as scipy.integrate.solve_ivp
            (1.7.3)
        """

        self.options = {
            'method': 'LSODA',
            'atol': 1e-10,
            'dense_output': True,
        }
        for k, v in kwargs.items():
            self.options[k] = v

        self.options.update({
            't_eval': np.linspace(t_span[0], t_span[1], num=self.evals)
        })

        nsol = si.solve_ivp(ds_func, t_span, x0, **self.options)

        return nsol

    def nsolve_custom(self, ds_func, x0, **kwargs):
        """
        Custom ODE solver onthe basis of Foward-Euler, for given adaptation
        models with custom evaluation timescales and step width and custom
        update function. Acts as a wrapper for nsolve_fw_euler.

        Args:
            ds_func (callable function):\n
             The update function of the dynamic system, which is supposed to
            define dx/dt, according to scipy standards.\n
            t_span (tuple):\n
             A tuple setting the logarithmic range of values for explicit
             evaluation.\n
            x0 (array):\n
             An array of initial values to start the numeric
             integration of the dynamic system.\n
            kwargs (dict):\n
             A dictionary setting custom solver options such as
             number of evaluations 'sample', number of steps 'num_steps'\n
        Returns:
            Iterable: Tuple of two arrays (t, x(t))
        """

        self.options = {
            'num_steps': 1,
            'samples': 1,
            'step': 1,
        }
        for k, v in kwargs.items():
            self.options[k] = v

        ns, sr = self.set_integration_scale(
                        self.options['num_steps'],
                        self.options['samples']
                        )
        self.options['sample_rate'] = sr
        self.options['num_steps'] = ns

        nsol = self.nsolve_fw_euler(ds_func, x0, **self.options)

        return nsol

    def nsolve_fw_euler(self, ds_func, x0, **kwargs):
        """
        Custom ODE solver onthe basis of Foward-Euler, for given adaptation
        models with custom evaluation timescales and step width and custom
        update function.

        Args:
            ds_func (callable):\n
             The update function of the dynamic system, which is supposed to
             define dx/dt, according to scipy standards.\n
            t_span (tuple):\n
             A tuple setting the logarithmic range of values for
             explicit evaluation.\n
            x0 (array):\n
             An array of initial values to start the numeric integration of the
             dynamic system.\n
            kwargs (dict):\n
             A dictionary setting custom solver options.\n
        Returns:
            Iterable: Tuple of two arrays (t, x(t))
        """

        t_samples = np.arange(
                        0,
                        kwargs['num_steps'],
                        step=kwargs['sample_rate']
                        )
        t_samples *= kwargs['step']

        sol = np.zeros((kwargs['samples'], len(x0)))
        c_m = 0
        x_0 = np.array(x0)

        for i in range(kwargs['num_steps']):

            if (i % kwargs['sample_rate']) == 0:
                sol[c_m] = x_0[:]
                c_m += 1

            dx = ds_func(i*kwargs['step'], x_0, *kwargs['args'])
            x_0 = np.add(x_0, dx*kwargs['step'])

        nsol = proxy_solver(t_samples, sol)

        return nsol

    def set_integration_scale(self, Num_steps, sample):
        """
        Computes the sample rate for the given dynamic system.

        Given the pre-set number of computing steps and number of evaluations,
        Adjusts/increases the number of integration cycles such that last
        evaluation coincicdes with last cycle.

        Args:
            Num_steps (int):\n
             Number of integration steps to perform for a Forward-Euler scheme.
             \n
            sample(int):\n
             Number of desired function evaluations for output.\n
        Returns:
            Iterable: Num_steps, sample_rate
        """

        # reshape number of integration steps & sample rates for consistency
        sample_rate = int(Num_steps/sample)
        if (sample_rate*sample) < Num_steps:
            Num_steps = sample_rate*sample

        return Num_steps, sample_rate
# TODO: GlobalOptimizers
# @dataclass
# class MySteps():
#
#     stepsize: float = 0.
#     # def __init__(self, stepsize ):
#     #     self.stepsize  = stepsize
#     def __call__(self, x):
#         rx = np.add(x, np.random.rand(len(x))*self.stepsize)
#         return rx

# @dataclass
# class morph_optimize(morph):
#
#     def __post_init__(self):
#         E = self.flow.circuit.list_graph_edges
#         mysteps = MySteps(1.)
#         b0 = 1e-25
#         self.options = {
#             'step': mysteps,
#             'niter': 100,
#             'T': 10.,
#             'minimizer_kwargs': {
#                 'method': 'L-BFGS-B',
#                 'bounds': [(b0, None) for x in range(len(E))],
#                 'args': (self.flow.circuit),
#                 'jac': False,
#                 'tol': 1e-10
#                 }
#         }
#
#     def update_minimizer_options(**kwargs):
#
#         if 'step' in kwargs:
#             mysteps = MySteps(kwargs['step'])
#             kwargs['step'] = mysteps
#
#         for k, v in kwargs.items():
#             if k in self.options:
#                 options[k] = v
#
#         if 'minimizer_kwargs' in kwargs:
#             for ks, vs in kwargs['minimizer_kwargs']:
#                 minimizer_kwargs[ks] = vs
#
#     def optimize_network(self, cost_func, x0, **kwargs):
#
#         update_minimizer_options(**kwargs)
#
#         sol = sc.basinhopping(cost_func, x0, **self.options)
#
#         return sol
