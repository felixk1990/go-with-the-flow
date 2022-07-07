# goflow

This repository is all about simulating flow driven pruning in biological flow networks. For full package documentation see:
<https://felixk1990.github.io/go-with-the-flow/>
##  Introduction
This module 'goflow' is the final of a series of pyton packages encompassing a set of class and method implementations for a kirchhoff network datatype, in order to to calculate flow/flux on lumped parameter model circuits and their corresponding adaptation. The flow/flux objects are embedded in the kirchhoff networks, and can be altered independently from the underlying graph structure. This is meant for fast(er) and efficient computation and dependends on the packages 'kirchhoff','hailhydro'.

What does it do: Modelling morphogenesis of capillary networks which can be modelled as Kirchhoff networks, and calculate its response given flow q/ pressure dp/flux j based stimuli functions. We generally assume Hagen-Poiseulle flow and first order solution transport phenomena Given the radii r of such vessel networks we simulate its adaptation as an ODE system with <br>

<img src="https://render.githubusercontent.com/render/math?math=\dot{r}_i (t) = f_i( \lbrace r \rbrace, \lbrace q \rbrace, \lbrace j \rbrace, ... ) ">

The dynamic system f is usually constructed for a Lyapunov function L with <br>

<img src="https://render.githubusercontent.com/render/math?math=L = \sum_i \alpha_1 p_i^2r_i^4 %2B \alpha_0 r_i^2 %2B+...">

such that we get <br>
<img src="https://render.githubusercontent.com/render/math?math=f_i( \lbrace r \rbrace, \lbrace q \rbrace, \lbrace j \rbrace, ... )= -\frac{dL}{dr_i} ">

The package not only includes premade Lyapunov functions and flow/flux models but further offers custom functions to be provided by the user.
##  Installation
pip install goflow
##  Usage
First you have to create your rudimentary circuit/ flow network which yu want to evolve later:
```
import numpy as np
import kirchhoff.circuit_init as kfi
from goflow.adapter import init_ivp as gi

# #initialize plexus+flow pattern
init_flow=dict(
    crystal_type='triagonal_planar',
    periods= 3,
)

C = kfi.initialize_circuit_from_crystal(**init_flow)

# plot initial network with data of choice
fig = C.plot_circuit()
fig.show()
```
![plexus](./gallery/plexus_murray.png)

Next you have to set the dynamical model (how are flows calculated, vessels adjusted during each adaptation step):
```
# set plexus parameters
pars_src = {
    'modesSRC': 'root_geometric'
}
pars_plx = {
    'modePLX':'default',
}
# set model and model parameters
pars_model = {
    'alpha_0':1.,
    'alpha_1':1.
}

# # initialize dynamic system and set integration parameters
morpheus = gi.morph_dynamic(C, 'murray', [pars_model, pars_src, pars_plx])
morpheus.evals = 200

# numerically evaluate the system
cnd = morpheus.flow.circuit.edges['conductivity']
cnd_scale = morpheus.flow.circuit.scales['conductance']

sp = {
    't0': 1e-05,
    't1': 4.,
    'x0': np.power(cnd/cnd_scale,0.25)*0.1,
}
nsol = morpheus.nlogSolve((sp['t0'],sp['t1']), sp['x0'])
# print(nsol)

dataPoints = zip(nsol.t,nsol.y.transpose())
murrayModel = morpheus.model
args = murrayModel.solver_options['args']
cost = [murrayModel.calc_cost_stimuli(t, y, *args)[0] for t, y in dataPoints]
```
When you are done, plot dynamics of vessel development:
```
# # plot dynamic data such as radii and costs
import matplotlib.pyplot as plt
fig,axs=plt.subplots(2,1,figsize=(12,6),sharex=True)
axs[0].plot(nsol.t,nsol.y.transpose(),alpha=0.1,color='b')
axs[1].plot(nsol.t,cost,alpha=0.2,color='b')

for i in range(2):
    axs[i].grid(True)
    axs[i].set_xscale('log')

axs[1].set_xlabel(r'time $t$')
axs[0].set_ylabel(r'radii $r$')
axs[1].set_ylabel(r'metabolic cost $\Gamma$')
axs[1].set_yscale('log')
plt.show()
```
![dynamics](./gallery/dynamics_murray.png)<br>
If you like you may generate interactive plots with plotly or animations just as easily, e.g. see the examples in the gallery section:

![updated1](./gallery/murray_triagonal_plexus.gif)<br>
![updated2](./gallery/murray_triagonal_dynm.gif)<br>

Further examples and recipes: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/felixk1990/go-with-the-flow/examples)
##  Requirements
``` pandas ```,``` networkx ```, ``` numpy ```, ``` scipy ```, ``` kirchhoff ```, ``` hailhydro ```, ```plotly```
##  Gallery

## Acknowledgement
* Pre-customized models presentend and implemented here as given by:
    *  Murray, The Physiological Principle of Minimum Work, 1926
    *  Katifori et al, Damage and Fluctuations Induce Loops in Optimal Transport Networks, 2010
    *  Corson, Fluctuations and Redundancy in Optimal Transport Networks, 2010
    *  Hu and Cai, Adaptation and Optimization of Biological Transport Networks, 2013
    *  Kramer and Modes, How to pare a pair: Topology control and pruning in intertwined complex networks, 2020

```goflow``` written by Felix Kramer
