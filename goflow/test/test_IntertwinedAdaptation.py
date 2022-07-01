# # @Author: Felix Kramer <felixk1990>
# # @Date:   2022-06-29T17:34:40+02:00
# # @Email:  felixuwekramer@proton.me
# # @Filename: test_IntertwinedAdaptation.py
# @Last modified by:   felix
# @Last modified time: 2022-07-01T15:14:01+02:00
#
#
# import numpy as np
# import os
# import os.path as op
#
# import kirchhoff.circuit_dual as kid
# from goflow.adapter import init_ivp as gi
# from test_Solvers import tmpDirectory
# from aux import *
#
# locPath = './goflow/test/tmp'
# pars_src = dict(modesSRC='dipole_border')
# pars_plx = dict(modePLX='default')
# modes = ['meigel', 'link']
#
#
# def initEval(pars_model, pars_rnd):
#
#     #initialize circuit+flow pattern
#     D=kid.initialize_dual_from_minsurf(dual_type='laves',num_periods=2)
#
#     pars_src = {
#     'modeSRC': ['root_short','root_long'],
#     }
#     pars_plx = {
#         'modePLX': ['default','default'],
#     }
#
#     # # initialize dynamic system and set integration parameters
#     morpheus = gi.morph_dynamic(D, 'kramer', [pars_model, pars_src, pars_plx, pars_rnd])
#     # numerically evaluate the system
#     mfc = [morpheus.flow[i].circuit for i in range(2)]
#     x = [ np.power(c.edges['conductivity']/c.scales['conductance'],0.25) for c in mfc]
#     xscale = np.amin(morpheus.flow[0].dist_adj)
#     sp = {
#         't0': 0.,
#         't1': .01,
#         'x_0': 0.1 * np.concatenate((x[0], x[1]))/xscale,
#     }
#     nsol = morpheus.autoSolve((sp['t0'],sp['t1']), sp['x_0'])
#
#     return nsol, morpheus
#
# @tmpDirectory
# def test_nsol_Cyclic_Attr():
#
#     pars_rnd={
#         'mode': ['default', 'default'],
#         'noise': [.1, .1],
#     }
#     # set model and model parameters
#     pars_model = {
#         'p_0': [.001, .001],
#         'p_1': [10000., 10000.],
#         'p_2': [10000., 10000.],
#         'p_3': pars_rnd['noise'],
#         'coupling': 2.,
#     }
#
#     # set model and model parameters
#     nsol, morpheus = initEval(pars_model, pars_rnd)
#
#     # test for sucess and otimization
#     assert nsol.success
#
#
# @tmpDirectory
# def test_nsol_Tree_Attr():
#
#     pars_rnd={
#         'mode': ['default', 'default'],
#         'noise': [.1, .1],
#     }
#     # set model and model parameters
#     pars_model = {
#         'p_0': [.1, .1],
#         'p_1': [1., 1.],
#         'p_2': [10000., 10000.],
#         'p_3': pars_rnd['noise'],
#         'coupling': 2.,
#     }
#
#     # set model and model parameters
#     nsol, morpheus = initEval(pars_model, pars_rnd)
#
#     # test for sucess and otimization
#     assert nsol.success
#
#
# @tmpDirectory
# def test_nsol_Cyclic_Rep():
#
#     pars_rnd={
#         'mode': ['default', 'default'],
#         'noise': [100., 100.],
#     }
#     # set model and model parameters
#     pars_model = {
#         'p_0': [.01, .01],
#         'p_1': [1., 1.],
#         'p_2': [10000., 10000.],
#         'p_3': pars_rnd['noise'],
#         'coupling': -2.,
#     }
#
#     # set model and model parameters
#     nsol, morpheus = initEval(pars_model, pars_rnd)
#
#     # test for sucess and otimization
#     assert nsol.success
#
#
# @tmpDirectory
# def test_nsol_Tree_Rep():
#
#     pars_rnd={
#         'mode': ['default', 'default'],
#         'noise': [.1, .1],
#     }
#     # set model and model parameters
#     pars_model = {
#         'p_0': [.001, .001],
#         'p_1': [10000., 10000.],
#         'p_2': [10000., 10000.],
#         'p_3': pars_rnd['noise'],
#         'coupling': -2.,
#     }
#
#     # set model and model parameters
#     nsol, morpheus = initEval(pars_model, pars_rnd)
#
#     # test for sucess and otimization
#     assert nsol.success
