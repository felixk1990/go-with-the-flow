# @Author: Felix Kramer <kramer>
# @Date:   23-06-2021
# @Email:  kramer@mpi-cbg.de
# @Project: phd_network_remodelling
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-09-08T21:08:26+02:00
import numpy as np
from dataclasses import dataclass, field


@dataclass
class model():

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

        self.ivp_options = {
            't0': 0.,
            't1': 1.,
            'x0': 1,
            'num': 100,
        }

        self.model_args = []
        self.solver_options = {
            't_eval': np.linspace(self.ivp_options['t0'], self.ivp_options['t1'], num = self.ivp_options['num'])
        }

        self.events = {
                'default': self.flatlining_default,
                'dynamic': self.flatlining_dynamic,
            }

    def update_event_func(self):

        try:
            self.solver_options['events'] = self.events[self.solver_options['events']]

        except:
            print('Warning: Event handling got inadequadt event function, falling back to default')
            self.solver_options['events'] = self.events['default']

    def flatlining_default(self, t, x_0, *args):

        F, dF = self.calc_cost_stimuli(t, x_0, *args)
        dF_abs = np.linalg.norm(dF)
        quality = np.round(np.divide(dF_abs, F), self.null_decimal)

        z = quality - np.power(10., -(self.null_decimal-1))

        # print(f'ref: {z}')
        # print(f't: {t}')
        return z

    def flatlining_dynamic(self, t, x_0, *args):

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

        for k, v in model_pars.items():

            self.model_args.append(v)

    def set_solver_options(self, solv_opt):

        for k, v in solv_opt.items():

            self.solver_options[k] = v

        self.update_event_func()

    def calc_update_stimuli(self, t, x_0, *args):

        dx = np.array()

        return dx

    def calc_cost_stimuli(self, t, x_0, *args):

        F = np.array()
        dF = np.array()

        return F, dF
