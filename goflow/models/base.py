# @Author: Felix Kramer <felix>
# @Date:   2022-06-28T16:24:41+02:00
# @Email:  felixuwekramer@proton.me
# @Filename: base.py
# @Last modified by:   felix
# @Last modified time: 2022-07-01T17:21:12+02:00

import numpy as np
from dataclasses import dataclass, field


@dataclass
class model():
    """
    The basic model class, acting as a center piece for 'morph' simulations, as
    it defines the individual dynamic system.

    This class defines the ODE sytem, metabolic cost model and a diverse set of
    functionalities for effective evaluation of any simulation's proceeding.
    Based on scipy.integrate.solve_ivp (1.7.3) it defines specific event
    handlers for stiff systems and utility functions.

    Args:
        ivp_options (dict):\n
         Information to generate the internal solver_options. Providing t0,t1
         number of evaluation and x0.\n
        model_args (list):\n
         Model specific paramerets need to evaluate the update rules of the DS
         \n
        solver_options (dict):\n
         Specifying runtime and evaluation marks. \n
        events (dict):\n
         Events to consier, here in gernal flatlining events which allow for
         early termination of stiff simulations. \n
        null_decimals (in):\n
         description \n

    Returns:
        type: model

    """

    defVals = dict(init=False, repr=False)

    ivp_options: dict = field(default_factory=dict, **defVals)
    model_args: list = field(default_factory=list, **defVals)
    solver_options: dict = field(default_factory=dict, init=False, repr=True)
    events: dict = field(default_factory=dict, **defVals)
    null_decimal: int = field(default=6, **defVals)
    # jac: bool = field(default=False, **defVals)

    def __post_init__(self):

        self.init()

    def init(self):
        """
        The model post_init function.

        Setting proxy values for internal variables.

        Args:
            None

        Returns:
            type: None

        """

        self.ivp_options = {
            't0': 0.,
            't1': 1.,
            'x0': 1,
            'num': 100,
        }

        self.model_args = []
        self.solver_options = {
            't_eval': np.linspace(
                self.ivp_options['t0'],
                self.ivp_options['t1'],
                num=self.ivp_options['num']
                )
        }

        self.events = {
                'default': self.flatlining_default,
                'dynamic': self.flatlining_dynamic,
            }

    def update_event_func(self):
        """
        Update the event function and ensures that solver_options are set.

        Args:
            None

        Returns:
            None

        Raises:
            Exception: \\n
             Warning: Event handling got inadequadt event function,
             falling back to default

        """

        try:
            kw = self.solver_options['events']
            self.solver_options['events'] = self.events[kw]

        except Exception:
            print(
                """
                Warning: Event handling got inadequadt event function,
                falling back to default
                """
                )
            self.solver_options['events'] = self.events['default']

    def flatlining_default(self, t, x_0, *args):
        """
        The default flatlining function for determining the terminal event of
        the dynamical system.

        Based on the event handling schemes of scipy.integrate.solve_ivp
        (1.7.3), has the same call signature as calc_update_stimuli(). The
        method checks whether the system's Lyapunov function
        has decreased beneath a certain threshold.

        Args:
            t (float):\n
             Current time step in numeric ODE evaluation \n
            x_0 (array):\n
             Current state vector of the DS. \n
            args (iterable):\n
             Model specific tuple of parameters, same signature as needed for
             calc_update_stimuli. \n

        Returns:
            float: \n
             z, the signature of sign change, signaling the internal
             threshold being passed.

        """

        F, dF = self.calc_cost_stimuli(t, x_0, *args)
        dF_abs = np.linalg.norm(dF)
        quality = np.round(np.divide(dF_abs, F), self.null_decimal)

        z = quality - np.power(10., -(self.null_decimal-1))

        # print(f'ref: {z}')
        # print(f't: {t}')
        return z

    def flatlining_dynamic(self, t, x_0, *args):

        """
        The dynamic flatlining function for determining the terminal event of
        the dynamical system.

        Based on the event handling schemes of scipy.integrate.solve_ivp
        (1.7.3), has the same call signature as calc_update_stimuli(). The
        method checks whether the magnitude of relative change of the
        state vector x_0 has decreased beneath a certain threshold.

        Args:
            t (float):\n
             Current time step in numeric ODE evaluation \n
            x_0 (array):\n
             Current state vector of the DS. \n
            args (iterable):\n
             Model specific tuple of parameters, same signature as needed for
             calc_update_stimuli. \n

        Returns:
            float: \n
             z, the signature of sign change, signaling the internal
             threshold being passed.

        """

        dx = self.calc_update_stimuli(t, x_0, *args)
        dx_abs = np.absolute(dx)
        rel_r = np.divide(dx_abs, x_0)
        quality = np.round(np.linalg.norm(rel_r), self.null_decimal)

        z = quality-np.power(10., -(self.null_decimal-1))

        return z

    for f in [flatlining_default, flatlining_dynamic]:
        f.terminal = True
        f.direction = 1.

    def set_model_parameters(self, model_pars):
        """
        Set internal model arguments array.

        Args:
             model_pars (dict):\n
              Model argument dictionary generated on program initilization. \n

        Returns:
            None

        """

        for k, v in model_pars.items():

            self.model_args.append(v)

    def set_solver_options(self, solv_opt):
        """
        Set internal solver_options and update event function.

        Args:
             solv_opt (dict):\n
              Solver argument dictionary generated on program initilization. \n

        Returns:
            None

        """

        for k, v in solv_opt.items():

            self.solver_options[k] = v

        self.update_event_func()

    def calc_update_stimuli(self, t, x_0, *args):
        """
        The dynamic system's temporal update rule, computing the gradient -dF
        for dx/dt.

        Args:
            t (float):\n
             Current time step in numeric ODE evaluation \n
            x_0 (array):\n
             Current state vector of the DS. \n
            args (iterable):\n
             Model specific tuple of parameters, needed to evaluate stimulus ]
             functions \n

        Returns:
            array: \n
             The computed change dx to the state vector x_0.

        """

        dx = np.array()

        return dx

    def calc_cost_stimuli(self, t, x_0, *args):
        """
        Computes the dynamic system's Lyapunov function and its gradient.

        Args:
            t (float):\n
             Current time step in numeric ODE evaluation \n
            x_0 (array):\n
             Current state vector of the DS. \n
            args (iterable):\n
             Model specific tuple of parameters, needed to evaluate stimulus ]
             functions \n

        Returns:
            iterable: \n
             The computed gradient dF to the current cost function F, as well
             as F itself.

        """

        F = np.array()
        dF = np.array()

        return F, dF
