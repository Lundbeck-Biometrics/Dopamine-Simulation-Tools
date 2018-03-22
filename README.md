# Dopamine-Simulation-Tool
This is a toolbox for making simulations of dopamine (DA) signaling in python. It is developed under Python 3.5

## Importing
For example:

from DopamineNeuronClass import DA, D1MSN, D2MSN

## Dopamine Neuron Class DA
Class definition of a DA system. This is main class used to make simulations of two compartment DA signalling. 

### How to create an instance 
Create VTA system using DAsys = DA("vta").  This will establish equations that corresponds to mesolimbic signaling. Ie. VTA  > nucleus accumbens DA signaling.

Create SNc DA system using DA("SNC"). System will be set up for simulating nigrostriatal signalling.  Ie SNc -> dorsal striatum

The difference between these two systems is the amount of DA released and reuptake. 

Arguments are not case sensitive. All DA systems contain two instances of feedback-classes which represent somatodendritic and terminal autoreceptors respectively. 

## Attributes and methods in DA class: 

### update(dt,  nu_in = 5, e_stim = False):
dt = float Euler timestep. 0.01 seconds is often OK. 
nu_in = float input cell firing rate in Hz. Default is 5 Hz. The actual firing rate will include negative somatodendritic feedback. Steady state with default input is ~4 Hz. 
e_stim = bool. If True somatodendritic feedback will be bypassed. Used to emulate experiments where electric or optogenetic simulus drives neurons. 

This method updates the status of feedback systems and the DA concetrations in each compartment. 

Update the DA system in time step dt using DAsys.update(dt). 
Current version of the script creates a list of DAsystems and runs a simulation.  

## DAsystem class (obsolete)
Sets up a DA system. Import this class using 'from MouseClassV5 import DASimulation'.

For example see the 'DA_run1.py' script 

## MouseClass
This is an old project of economic decion making in rodents. The mouse class simulates a mouse that likes reward and will press levers to get it. It is motivated by dopamine concentrations. It contains a simple version of a DA-system which is used here as a starting point for a more generic DA simulation. 

See 'mouserun1.py' for an example... 

## MarkovLeverTool
This is a list of functions used by the mouse class

