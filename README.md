# goflow
This repository is all about simulating flow driven pruning in biological flow networks. 
##  Introduction
This module 'goflow' is the final of a series of pyton packages encompassing a set of class and method implementations for a kirchhoff network datatype, in order to to calculate flow/flux on lumped parameter model circuits and their corresponding adaptation. The flow/flux objects are embedded in the kirchhoff networks, and can be altered independently from the underlying graph structure. This is meant for fast(er) and efficient computation and dependends on the packages 'kirchhoff','hailhydro'.

What does it do: Modelling morphogenesis of capillary networks which can be modelled as Kirchhoff networks, and calculate its response given flow q/ pressure dp/flux j based stimuli functions. We generally assume Hagen-Poiseulle flow and first order solution transport phenomena Given the radii r of such vessel networks we simulate its adaptation as an ODE system with <br>

<img src="https://render.githubusercontent.com/render/math?math=\dot{r}_i (t) = f_i( \lbrace r \rbrace, \lbrace q \rbrace, \lbrace j \rbrace, ... ) ">

The dynamic system f is usually constructed for a Lyapunov function L with <br>

<img src="https://render.githubusercontent.com/render/math?math=L = \sum_i \alpha_1 dp_i^2r_i^4 %2B \alpha_0 r_i^2 %2B+...">

such that we get <br>
<img src="https://render.githubusercontent.com/render/math?math=f_i( \lbrace r \rbrace, \lbrace q \rbrace, \lbrace j \rbrace, ... )= -\frac{dL}{dr_i} ">

The package not only includes premade Lyapunov functions and flow/flux models but further offers custom functions to be provided by the user.
##  Installation
pip install goflow
##  Usage
```
import numpy as np
import goflow.init_ivp as gi
import goflow.models as gfm

#initialize circuit+flow pattern
cfp={
    'type': 'hexagonal',
    'periods': 3,
    'source':'dipole_border',
    'plexus':'default',
}
flow=gfm.initialize_flow_on_crystal(cfp)

#set model and model parameters
mp={
    'alpha_0':1,
    'alpha_1':1.
}
murray=gfm.init(model='murray',pars=mp)

#initialize dynamic system and set integration parameters
morpheus=gi.morph_dynamic(flow=flow,model=murray)   
sp={
    't0': 0.,
    't1': 5.5,
    'x0': np.power(morpheus.flow.circuit.edges['conductivity']/morpheus.flow.circuit.scales['conductance'],0.25),
}

#plot initial network with data of choice
flow.circuit.draw_weight_scaling=2.
fig=flow.circuit.plot_circuit({'width':'conductivity'})
fig.show()
```
![plexus](./gallery/plexus_murray.png)

```
#numerically evaluate the system
nsol=morpheus.nsolve(murray.calc_update_stimuli,(sp['t0'],sp['t1']),sp['x0'], **murray.solver_options)
murray.jac=False
cost=[murray.calc_cost_stimuli(t,y,*murray.solver_options['args']) for t,y in zip(nsol.t,nsol.y.transpose())]

#plot dynamic data such as radii and costs
import matplotlib.pyplot as plt
fig,axs=plt.subplots(2,1,figsize=(12,6),sharex=True)
axs[0].plot(nsol.t,nsol.y.transpose(),alpha=0.1,color='b')
axs[1].plot(nsol.t,cost,alpha=0.2,color='b')

for i in range(2):
    axs[i].grid(True)
    
axs[1].set_xlabel(r'time $t$')
axs[0].set_ylabel(r'radii $r$')
axs[1].set_ylabel(r'metabolic cost $\Gamma$')
plt.show()

#plot network with data of choice
flow.circuit.edges['conductivity']=nsol.y.transpose()[-1]
flow.circuit.edges['flow_rate']=flow.calc_configuration_flow()
flow.circuit.draw_weight_scaling=2.
fig=flow.circuit.plot_circuit({'width':'conductivity'})
fig.show()
```
![dynamics](./gallery/dynamics_murray.png)
![updated](./gallery/updated_murray.png)
##  Requirements
``` pandas ```,``` networkx ```, ``` numpy ```, ``` scipy ```, ``` kirchhoff ```, ``` hailhydro ```
##  Gallery

## Acknowledgement
* Pre-customized models presentend and implemented here as given by:
    *  Murray, The Physiological Principle of Minimum Work, 1926
    *  Katifori et al, Damage and Fluctuations Induce Loops in Optimal Transport Networks, 2010
    *  Corson, Fluctuations and Redundancy in Optimal Transport Networks, 2010
    *  Hu and Cai, Adaptation and Optimization of Biological Transport Networks, 2013
    *  Kramer and Modes, How to pare a pair: Topology control and pruning in intertwined complex networks, 2020

```goflow``` written by Felix Kramer
