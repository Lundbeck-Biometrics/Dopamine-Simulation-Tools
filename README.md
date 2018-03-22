# Dopamine-Simulation-Tool
This is a toolbox for making simulations of dopamine (DA) signaling in python. It is developed under Python 3.5

## Importing
For example:

from DopamineNeuronClass import DA, D1MSN, D2MSN

## DA
Class definition of a DA system. This is main class used to make simulations of two compartment DA signalling. 

### How to create an instance 
Create VTA system using DAsys = DA("vta").  This will establish equations that corresponds to mesolimbic signaling. Ie. VTA  > nucleus accumbens DA signaling.

Create SNc DA system using DA("SNC"). System will be set up for simulating nigrostriatal signalling.  Ie SNc -> dorsal striatum

The difference between these two systems is the amount of DA released and reuptake. 

Arguments are not case sensitive. All DA systems contain two instances of feedback-classes which represent somatodendritic and terminal autoreceptors respectively. 

### Attributes and methods in DA class: 

#### update(dt,  nu_in = 5, e_stim = False):
dt = float Euler timestep. 0.01 seconds is often OK. 
nu_in = float input cell firing rate in Hz. Default is 5 Hz. The actual firing rate will include negative somatodendritic feedback. Steady state with default input is ~4 Hz. 
e_stim = bool. If True somatodendritic feedback will be bypassed. Used to emulate experiments where electric or optogenetic simulus drives neurons. 

This method updates the status of feedback systems and the DA concetrations in each compartment. 



#### NNeurons (=100)
The number of DA neurons in the system. When NNeurons = 100 (default) the release and uptake will correspond to litterature parameters for intact. For denervation studies reduce, release and uptake will be reduced accordingly. 

#### area ( = "SNc" / "VTA")
Describes the anatomical location of the cell bodies. If other area that the two standard areas is given, the DA class issues a warning.
#### Vmax_pr_neuron
This is the Michalis menten reuptake Vmax pr neuron in the simulation.  
  = 40 nM/s if area = "SNc"
  = 15 nM/s if area = "VTA"
If other area is chosen these attributes are not set. 

#### Gamma_pr_neuron
= 40 nM/s if area = "SNc"  = 15 nM/s if area = "VTA"If other area is chosen these attributes are not set. 

## DAsystem class (obsolete)
Sets up a DA system. Import this class using 'from MouseClassV5 import DASimulation'.

For example see the 'DA_run1.py' script 

## MouseClass (obsolete)
This is an old project of economic decion making in rodents. The mouse class simulates a mouse that likes reward and will press levers to get it. It is motivated by dopamine concentrations. It contains a simple version of a DA-system which is used here as a starting point for a more generic DA simulation. 

See 'mouserun1.py' for an example... 

#### MarkovLeverTool
This is a list of functions used by the mouse class

