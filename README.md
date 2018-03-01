# Dopamine-Simulation-Tool
This is a tool for making simulations of dopamine (DA) signaling in python

## DopamineNeuronClass
Class definition of a DA system. Create VTA system using DAsys = DA("vta"). Update the DA system in time step dt using DAsys.update(dt). 
Current version of the script creates a list of DAsystems and runs a simulation.  

## DAsystem class (obsolete)
Sets up a DA system. Import this class using 'from MouseClassV5 import DASimulation'.

For example see the 'DA_run1.py' script 

## MouseClass
This is an old project of economic decion making in rodents. The mouse class simulates a mouse that likes reward and will press levers to get it. It is motivated by dopamine concentrations. It contains a simple version of a DA-system which is used here as a starting point for a more generic DA simulation. 

## MarkovLeverTool
This is a list of functions used by the mouse class

