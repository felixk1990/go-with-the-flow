# goflow
<iframe 
src="https://felixk1990.github.io/go-with-the-flow/test.html"
title="description">
</iframe>
This repository is all about simulating flow driven pruning in biological flow networks.
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
import goflow.init_ivp as gi
import goflow.models as gfm

# #initialize circuit+flow pattern
cfp={
    'type': 'hexagonal',
    'periods': 3,
    'source':'dipole_border',
    'plexus':'default',
}
flow=gfm.initialize_flow_on_crystal(cfp)
flow.circuit.nodes['label'] = [n for n in flow.circuit.G.nodes()]
flow.circuit.edges['label'] = [e for e in flow.circuit.G.edges()]

# plot initial network with data of choice
flow.circuit.draw_weight_scaling=2.
fig=flow.circuit.plot_circuit()
fig.show()
```
![plexus](https://raw.githubusercontent.com/felixk1990/go-with-the-flow/main/gallery/plexus_murray.png)

Next you have to set the dynamical model (how are flows calculated, vessels adjusted during each adaptation step):
```
#set model and its parameters, here for example go for the classic Murray model (1926)

# set model and model parameters
mp={
    'alpha_0':1,
    'alpha_1':1.
}
murray=gfm.init(model='murray',pars=mp)

# initialize dynamic system and set integration parameters
morpheus=gi.morph_dynamic(flow=flow,model=murray)   
sp={
    't0': 0.,
    't1': 5.5,
    'x0': np.power(morpheus.flow.circuit.edges['conductivity']/morpheus.flow.circuit.scales['conductance'],0.25),
}

#numerically evaluate the system
nsol=morpheus.nsolve(murray.calc_update_stimuli,(sp['t0'],sp['t1']),sp['x0'], **murray.solver_options)
cost=[murray.calc_cost_stimuli(t,y,*murray.solver_options['args']) for t,y in zip(nsol.t,nsol.y.transpose())]
```
When you are done, plot dynamics of vessel development and the final structures.
```
#plot dynamic data such as radii and costs
import plotly.graph_objects as go

cl='rgba(0,0,255,.1)'
names = [str(s) for s in list(flow.circuit.G.edges())]
fig = go.Figure()
for i, c in enumerate(nsol.y):
    fig.add_trace(go.Scatter(x=nsol.t, y=c, mode='lines', name=names[i], line={'color': cl}))
fig.show()
```
![dynamics](https://raw.githubusercontent.com/felixk1990/go-with-the-flow/main/gallery/dynamics_murray.png)<br>
You can customize what the interactive plots display:
```
# plot final network
flow.circuit.edges['conductivity']=nsol.y.transpose()[-1]*3.
fig=flow.circuit.plot_circuit(linewidth=[flow.circuit.edges['conductivity']])
fig.show()

# plot network with data of choice
Q, dP = flow.calc_configuration_flow()
flow.circuit.edges['flow_rate'], dP=flow.calc_configuration_flow()
fig=flow.circuit.plot_circuit('flow_rate', color_edges=[np.absolute(Q)], linewidth=[flow.circuit.edges['conductivity']])
fig.show()
```
![updated1](https://raw.githubusercontent.com/felixk1990/go-with-the-flow/main/gallery/updated_murray1.png)<br>
![updated2](https://raw.githubusercontent.com/felixk1990/go-with-the-flow/main/gallery/updated_murray2.png)<br>

##  Requirements
``` pandas ```,``` networkx ```, ``` numpy ```, ``` scipy ```, ``` kirchhoff ```, ``` hailhydro ```, ```plotly```
##  Gallery
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/felixk1990/go-with-the-flow/HEAD)
## Acknowledgement
* Pre-customized models presentend and implemented here as given by:
    *  Murray, The Physiological Principle of Minimum Work, 1926
    *  Katifori et al, Damage and Fluctuations Induce Loops in Optimal Transport Networks, 2010
    *  Corson, Fluctuations and Redundancy in Optimal Transport Networks, 2010
    *  Hu and Cai, Adaptation and Optimization of Biological Transport Networks, 2013
    *  Kramer and Modes, How to pare a pair: Topology control and pruning in intertwined complex networks, 2020

```goflow``` written by Felix Kramer
