# @Author: Felix Kramer <kramer>
# @Date:   23-06-2021
# @Email:  kramer@mpi-cbg.de
# @Project: phd_network_remodelling
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-09-08T21:08:26+02:00

import numpy as np
import kirchhoff.circuit_flow as kfc
import hailhydro.flow_init as hfi
import hailhydro.flow_random as hfr
# custom
from goflow.init_ivp import *


def init(mode = 'default', **kwargs):

    model_mode = {
        'default': murray,
        'murray': murray,
        'corson': corson,
        'katifori': katifori,
    }

    if mode in model_mode:
        model = model_mode[mode](**kwargs)
    else:
        model = model_mode['default'](**kwargs)
        print('Warning: Unknown model, set default: Murray')

    return model

def initialize_flow_on_crystal(dict_pars):

    circuit = kfc.initialize_flow_circuit_from_crystal(crystal_type = dict_pars['type'], periods = dict_pars['periods'])
    circuit.set_source_landscape(dict_pars['source'])
    circuit.set_plexus_landscape(dict_pars['plexus'])
    flow = hfi.initialize_flow_on_circuit(circuit)

    return flow

def initialize_randflow_on_crystal(dict_pars):

    circuit = kfc.initialize_flow_circuit_from_crystal(crystal_type = dict_pars['type'], periods = dict_pars['periods'])
    circuit.set_source_landscape(dict_pars['source'])
    circuit.set_plexus_landscape(dict_pars['plexus'])
    flow = hfr.initialize_random_flow_on_circuit(circuit)

    return flow

def initialize_reflow_on_crystal(dict_pars):

    circuit = kfc.initialize_flow_circuit_from_crystal(crystal_type = dict_pars['type'], periods = dict_pars['periods'])
    circuit.set_source_landscape(dict_pars['source'])
    circuit.set_plexus_landscape(dict_pars['plexus'])
    flow = hfr.initialize_rerouting_flow_on_circuit(circuit, flow_setting = {'p_broken': dict_pars['p_broken'], 'num_iter': dict_pars['iteration']})

    return flow

class model():

    def __init__(self):

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

    def flatlining_default(self, t, x_0, flow, alpha_0, alpha_1):

        self.jac = True
        F, dF = self.calc_cost_stimuli(t, x_0, flow, alpha_0, alpha_1)
        dF_abs = np.linalg.norm(dF)
        quality = np.round(np.divide(dF_abs, F), self.null_decimal)

        z = quality - np.power(10., -(self.null_decimal-1))

        return z

    flatlining_default.terminal = True
    flatlining_default.direction =-1

    def flatlining_dynamic(self, t, x_0, flow, alpha_0, alpha_1):

        self.jac = True
        dx = self.calc_update_stimuli(t, x_0, flow, alpha_0, alpha_1)
        dx_abs = np.absolute(dx)
        rel_r = np.divide(dx_abs, x_0)
        quality = np.round(np.linalg.norm(rel_r), self.null_decimal)

        z = quality-np.power(10., -(self.null_decimal-1))

        return z

    flatlining_dynamic.terminal = True
    flatlining_dynamic.direction =-1

    def update_event_func(self):

        try:
            self.solver_options['events'] = self.events[self.solver_options['events']]

        except:
            print('Warning: Event handling got inadequadt event functin, falling back to default')
            self.solver_options['events'] = self.events['default']

class murray(model, object):

    def __init__(self, **kwargs):

        self.null_decimal = 30
        self.jac = False
        super(murray, self).__init__()

        self.events = {
                'default': self.flatlining_default,
                'dynamic': self.flatlining_dynamic,
            }
        self.model_args = [1., 1.]
        self.solver_options.update({ 'events': 'default'})
        if 'pars' in kwargs:
            self.set_model_parameters(kwargs['pars'])

    def set_model_parameters(self, model_pars):

        for k, v in model_pars.items():

            if 'alpha_0' == k :
                self.model_args[0] = v

            if 'alpha_1' == k :
                self.model_args[1] = v

    def set_solver_options(self, solv_opt):

        for k, v in solv_opt.items():

            self.solver_options[k] = v

        self.update_event_func()

    def calc_update_stimuli(self, t, x_0, flow, alpha_0, alpha_1):

        x_sq, p_sq = self.get_stimuli_pars(flow, x_0)

        s1 = 2.*alpha_1*np.multiply( p_sq, x_sq)
        s2 = alpha_0*np.ones(len(x_0))
        ds = np.subtract(s1, s2)
        dx = 2*np.multiply(ds, x_0)

        return dx

    def calc_cost_stimuli(self, t, x_0, flow, alpha_0, alpha_1):

        x_sq, p_sq = self.get_stimuli_pars(flow, x_0)

        f1 = alpha_1*np.multiply(p_sq, np.power(x_sq, 2))
        f2 = alpha_0 *x_sq
        F = np.sum(np.add(f1, f2))

        if self.jac:

            dF = -self.calc_update_stimuli(t, x_0, flow, alpha_0, alpha_1)

            return F, dF

        return F

    def get_stimuli_pars(self, flow, x_0):

        k = flow.circuit.scales['conductance']
        src =flow.circuit.nodes['source']

        conductivity = flow.calc_conductivity_from_cross_section( np.power(x_0, 2), k)
        x_sq = flow.calc_cross_section_from_conductivity( conductivity, k)
        p_sq, q_sq =flow.calc_sq_flow( conductivity, src)

        return x_sq, p_sq, k

class corson(murray, object):

        def __init__(self, **kwargs):

            super(corson, self).__init__()

            self.model_args = [1., 1., 1, 0.]
            self.solver_options.update({ 'events': 'dynamic'})
            if 'pars' in kwargs:
                self.set_model_parameters(kwargs['pars'])

        def set_model_parameters(self, model_pars):

            for k, v in model_pars.items():

                if 'alpha_0' == k :
                    self.model_args[0] = v

                if 'alpha_1' == k :
                    self.model_args[1] = v

                if 'gamma' == k :
                    self.model_args[2] = v

                if 'noise' == k :
                    self.model_args[3] = v

        def calc_update_stimuli(self, t, x_0, flow, alpha_0, alpha_1, gamma, noise):

            flow.set_effective_source_matrix(noise)
            x_sq, p_sq = self.get_stimuli_pars(flow, x_0)

            s1 = 2.*alpha_1*np.multiply( p_sq, np.power(x_sq, gamma))
            s2 = alpha_0*np.ones(len(x_0))
            ds = np.subtract(s1, s2)
            dx = 2*np.multiply(ds, x_0)

            return dx

        def get_stimuli_pars(self, flow, x_0):

            k = flow.circuit.scales['conductance']

            conductivity = flow.calc_conductivity_from_cross_section( np.power(x_0, 2), k)
            x_sq = flow.calc_cross_section_from_conductivity( conductivity, k)
            p_sq, q_sq = flow.calc_sq_flow_effective( conductivity)

            return x_sq, p_sq

class katifori(corson, object):

    def __init__(self, **kwargs):

        super(katifori, self).__init__()

        self.model_args = [1., 1.]
        if 'pars' in kwargs:
            self.set_model_parameters(kwargs['pars'])

    def set_model_parameters(self, model_pars):

        for k, v in model_pars.items():

            if 'alpha_0' == k :
                self.model_args[0] = v

            if 'alpha_1' == k :
                self.model_args[1] = v

    def calc_update_stimuli(self, t, x_0, flow, alpha_0, alpha_1):

        avg_dP_sq, avg_F_sq, avg_R, avg_diss = self.get_stimuli_pars(flow, x_0)
        s1 = 2.*alpha_1*avg_diss
        s2 = alpha_0*avg_R
        dx = 2*np.subtract(s1, s2)

        return dx

    def get_stimuli_pars(self, flow, x_0):

        k = flow.circuit.scales['conductance']

        conductivity = flow.calc_conductivity_from_cross_section( np.power(x_0, 2), k)
        avg_dP_sq, avg_F_sq, avg_R, avg_diss = flow.calc_sq_flow_avg(conductivity)
        return avg_dP_sq, avg_F_sq, avg_R, avg_diss

# class meigel(model, object):
#
#     def __init__(self, **kwargs):
#
#         super(alim, self).__init__()
#
#         self.model_args = [1., 1.]
#         if 'pars' in kwargs:
#             self.set_model_parameters(kwargs['pars'])
#
#     def solve_absorption(self, t, R, K):
#
#         K.C = np.power(R, 4)*K.k
#         K.R = R
#
#         dr, PE = self.calc_absorption_dr(K)
#         self.save_dynamic_output([R, PE], ['radii', 'PE'], K)
#
#         return dr
#
#     def calc_absorption_dr(self, K):
#
#         dr = np.zeros(self.M)
#         c, B_new, K = self.calc_profile_concentration(K)
#
#         phi = self.calc_absorption(K.R, K)
#         J_phi = self.calc_absorption_jacobian(K.R, K)
#         phi0 = np.ones(self.M)*K.phi0
#
#         for i in range(self.M):
#             dr[i] = -2.*np.sum(np.multiply(np.subtract(phi, phi0), J_phi[i, :]))
#
#         return dr, K.PE
#
#     # calc and optimize networ costs with established sovers
#     def calc_absorption_cost(self, R, *args):
#
#         K = args[0]
#         K.C = np.power(R[:], 4)*K.k
#         K.R = R[:]
#         c, B_new, K = self, calc_profile_concentration(K)
#
#         phi = self.calc_absorption(R, dicts[0], K)
#         J_phi = self.calc_absorption_jacobian(R, B_new, K)
#
#         phi0 = np.ones(m)*K.phi0
#         F = np.sum(np.power(np.subtract(phi, phi0), 2))
#         DF = np.array([ 2.*np.sum(np.multiply(np.subtract(phi, phi0), J_phi[j, :])) for j in range(m) ])
#         # print(F)
#         return F, DF
#         # return F
#
# class kramer(meigel, object):
#
#     def __init__(self, **kwargs):
#
#         super(meigel, self).__init__()
#
#         self.model_args = [1., 1.]
#         if 'pars' in kwargs:
#             self.set_model_parameters(kwargs['pars'])
#
#     def calc_absorption_shear_dr(self, K):
#
#         dr = np.zeros(len(K.R))
#         c, B_new, K = self.calc_profile_concentration(K)
#         phi = self.calc_absorption(K.R, K)
#
#         J_phi = self.calc_absorption_jacobian(K.R, K)
#
#         Q, dP, X = self.calc_flows_pressures(K)
#         phi0 = np.ones(self.M)*K.phi0
#
#         for i in range(self.M):
#             dr[i] = -2.*np.sum(np.multiply(np.subtract(phi, phi0), J_phi[i, :]))
#
#         # DR = np.multiply(np.subtract(2.*K.alpha_1*np.power(np.multiply(dP, K.R), 2), K.alpha_0*np.ones(len(phi))), K.R)
#         DR = self.calc_shear_dr(K)
#         dr = np.add(dr, 2.*DR)
#
#         return dr, K.PE
#
#     def calc_shear_absorption_dr(self, K):
#
#         dr = np.zeros(len(K.R))
#         c, B_new, K = self.calc_profile_concentration(K)
#         phi = self.calc_absorption(K.R, K)
#         J_phi = self.calc_absorption_jacobian(K.R, K)
#
#         Q, dP, X = self.calc_flows_pressures(K)
#         phi0 = np.ones(self.M)*K.phi0
#
#         for i in range(self.M):
#             dr[i] = -2.*K.alpha_0*np.sum(np.multiply(np.subtract(phi, phi0), J_phi[i, :]))
#
#         DR = np.multiply(2.*np.subtract(np.power(np.multiply(dP, K.R), 2)*K.k, K.alpha_1*np.ones(len(phi))*np.sqrt(K.k)), K.R)
#         dr = np.add(dr, 2.*DR)
#
#         return dr, K.PE
#
#     def calc_absorption_dissipation_cost(self, R, *args):
#
#         K = args[0]
#         m = len(K.R)
#         K.C = np.power(R[:], 4)*K.k
#         c, B_new, S, K = self.calc_profile_concentration(K)
#         Q, dP, P = self.calc_flows_pressures(K)
#
#         phi = self.calc_absorption(R, dicts[0], K)
#         J_phi = self.calc_absorption_jacobian(R, B_new, K)
#
#         phi0 = np.ones(self.M)*K.phi0
#         F = np.sum(np.power(np.subtract(phi, phi0), 2))+ K.alpha_0*np.sum(np.multiply(np.power(dP, 2), np.power(R, 4)))
#         DF = np.array([ 2.*np.sum(np.multiply(np.subtract(phi, phi0), J_phi[j, :])) for j in range(self.M) ])
#         DF = np.subtract(DF, 4.*K.alpha_0*np.multiply(np.power(dP, 2), np.power(R, 3)))
#
#         return F, DF
#
#     def calc_absorption_volume_cost(self, R, *args):
#
#         K = args[0]
#         m = len(K.R)
#         K.C = np.power(R[:], 4)*K.k
#         c, B_new, S, K = self.calc_profile_concentration(K)
#
#         phi = self.calc_absorption(R, dicts[0], K)
#         J_phi = self.calc_absorption_jacobian(R, B_new, K)
#
#         phi0 = np.ones(m)*K.phi0
#         F = np.sum(np.power(np.subtract(phi, phi0), 2))+K.alpha*np.sum(np.power(R, 2))
#         DF = np.array([ 2.*np.sum(np.multiply(np.subtract(phi, phi0), J_phi[j, :])) for j in range(m) ])
#         DF = np.add(DF, K.alpha*R)
#
#         return F, DF
#
#     def calc_absorption_dissipation_volume_cost(self, R, *args):
#
#         K = args[0]
#         K.C = np.power(R[:], 4)*K.k
#         K.R = R[:]
#         c, B_new, S, K = self.calc_profile_concentration(K)
#
#         phi = self.calc_absorption(R, K)
#         J_phi = self.calc_absorption_jacobian(R, B_new, K)
#         phi0 = np.ones(self.M)*K.phi0
#
#         Q, dP, P = self.calc_flows_pressures(K)
#         sq_R = np.power(R, 2)
#         F = np.sum(np.power(np.subtract(phi, phi0), 2)) + K.alpha_1*np.sum(np.add(np.multiply(np.power(dP, 2), np.power(R, 4)), K.alpha_0*sq_R))
#         DF1 = np.array([ 2.*np.sum(np.multiply(np.subtract(phi, phi0), J_phi[j, :])) for j in range(m) ])
#         DF2 = 2.*K.alpha_1*np.multiply(np.subtract(np.ones(m)*K.alpha_0, 2.*np.multiply(np.power(dP, 2), sq_R)), R)
#
#         return F, np.add(DF1, DF2)
#         # return F
#
#     def calc_absorption_volumetric_shear_dr(self, K):
#
#         DR1 = self.calc_shear_dr(K)
#         DR2 = self.calc_absorption_volumetric_dr(K)
#         # ones = np.ones(len(K.dict_volumes.values()))
#         # c, B_new, K = self.calc_profile_concentration(K)
#         #
#         # phi = self.calc_absorption(K.R, K)
#         # J_phi = self.calc_absorption_jacobian(K.R, K)
#         # phi0 = ones*K.phi0
#         # dphi = ones
#         #
#         # for i, v in enumerate(K.dict_volumes.keys()):
#         #     dphi[v] = np.sum(phi[K.dict_volumes[v]]-phi0[v])
#         #
#         # for j, e in enumerate(self.list_e):
#         #     for i, v in enumerate(K.dict_volumes.keys()):
#         #         dr[j] -=2.*dphi[v]*np.sum(J_phi[j, K.dict_volumes[v]])
#         dr = np.add(DR1, DR2)
#         return dr
#
#     def calc_absorption_volumetric_dr(self, K):
#
#         dr = np.zeros(len(K.R))
#         ones = np.ones(len(K.dict_volumes.values()))
#         c, B_new, K = self.calc_profile_concentration(K)
#
#         phi = self.calc_absorption(K.R, K)
#         J_phi = self.calc_absorption_jacobian(K.R, K)
#         phi0 = ones*K.phi0
#         dphi = ones
#
#         for i, v in enumerate(K.dict_volumes.keys()):
#             dphi[i] = np.sum(phi[K.dict_volumes[v]])-phi0[i]
#
#         for j, e in enumerate(self.list_e):
#             for i, v in enumerate(K.dict_volumes.keys()):
#                 dr[j] -=2.*dphi[i]*np.sum(J_phi[j, K.dict_volumes[v]])
#
#         return dr
#
#     def solve_absorption_volumetric_shear(self, t, R, K):
#
#         K.C = np.power(R, 4)*K.k
#         K.R = R
#
#         dr = self.calc_absorption_volumetric_shear_dr(K)
#         self.save_dynamic_output([R, K.PE], ['radii', 'PE'], K)
#
#         return dr
#
#     def solve_absorption_volumetric(self, t, R, K):
#
#         K.C = np.power(R, 4)*K.k
#         K.R = R
#
#         dr = self.calc_absorption_volumetric_dr(K)
#         self.save_dynamic_output([R, K.PE], ['radii', 'PE'], K)
#
#         return dr
#
#     def solve_shear_absorption(self, t, R, K):
#
#         K.C = np.power(R, 4)*K.k
#         K.R = R
#
#         dr, PE = self.calc_shear_absorption_dr(K)
#         self.save_dynamic_output([R, PE], ['radii', 'PE'], K)
#
#         return dr
#
#     def solve_absorption_shear(self, t, R, K):
#
#         K.C = np.power(R, 4)*K.k
#         K.R = R
#
#         dr, PE = self.calc_absorption_shear_dr(K)
#         self.save_dynamic_output([R, PE], ['radii', 'PE'], K)
#
#         # x = 16
#         # z = np.round(np.linalg.norm(np.divide(dr, R)), x)-np.power(10., -(x-1))
#         # print('num_solve:'+str(z))
#         # print('time_calc:'+str(t))
#         return dr
#
#     def solve_absorption_shear_custom(self, K):
#
#         sol = []
#         for i in range(self.iterations):
#             try:
#                 dr, PE = self.calc_absorption_shear_dr(K)
#                 self.save_dynamic_output([R, PE], ['radii', 'PE'], K)
#                 K.R = np.add(K.R, dr*self.dt)
#                 K.C = np.power(K.R, 4)*K.k
#                 sol.append(K.R)
#             except:
#                 break
#
#         return sol, K
#
#     def solve_shear_absorption_custom(self, K):
#
#         sol = []
#         for i in range(self.iterations):
#             try:
#                 dr, PE = self.calc_shear_absorption_dr(K)
#                 self.save_dynamic_output([R, PE], ['radii', 'PE'], K)
#                 K.R = np.add(K.R, dr*self.dt)
#                 K.C = np.power(K.R, 4)*K.k
#                 sol.append(K.R)
#             except:
#                 break
#
#         return sol, K
#
#     def solve_volumetric_custom(self, K):
#
#         sol = []
#         for i in range(self.iterations):
#             try:
#                 dr, PE = self.calc_absorption_dr(K)
#                 self.save_dynamic_output([R, PE], ['radii', 'PE'], K)
#                 K.R = np.add(K.R, dr*self.dt)
#                 K.C = np.power(K.R, 4)*K.k
#                 sol.append(K.R)
#             except:
#                 break
#
#         return sol, K
#
#     def solve_absorption_volumetric_shear_custom(self, K):
#
#         sol = []
#         for i in range(self.iterations):
#             try:
#                 dr = self.calc_absorption_volumetric_shear_dr(K)
#                 self.save_dynamic_output([R, K.PE], ['radii', 'PE'], K)
#                 K.R = np.add(K.R, dr*self.dt)
#                 K.C = np.power(K.R, 4)*K.k
#                 sol.append(K.R)
#             except:
#                 break
#
#         return sol, K
#
#     def solve_shear_fluctuation_custom(self, K):
#
#         sol = []
#         for i in range(self.iterations):
#
#             # try:
#                 # dr, shear_sq, dV_sq, F_sq, avg_phi = self.calc_shear_fluctuation_dr(K)
#                 # self.save_dynamic_output([shear_sq, dV_sq, F_sq, avg_phi], ['shear_sq', 'dV_sq', 'F_sq', 'Phi'], K)
#                 dr, shear_sq, dV_sq, F_sq = self.calc_shear_fluctuation_dr(K)
#                 self.save_dynamic_output([shear_sq, dV_sq, F_sq], ['shear_sq', 'dV_sq', 'F_sq'], K)
#                 K.R = np.add(K.R, dr*self.dt)
#                 K.C = np.power(K.R, 4)*K.k
#                 sol.append(K.R)
#             # except:
#             #     break
#
#         return sol, K
