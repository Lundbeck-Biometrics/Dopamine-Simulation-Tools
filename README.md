# Dopamine-Simulation-Tool
This is a toolbox for making simulations of dopamine (DA) signaling in python. It is developed under Python 3. 

The goal of these simulations is to provide a toolbox that enables 
1. Functional interpretation of DA cell firing, 
2. how this is influenced by typical drugs and diseases. 
2. How long term changes in signalling by drugs or diseases influence post synaptic signalling.  

The current version focuses on how DA cell firing regulates cAMP levels in post synaptic neurons. 

Authored by Jakob Dreyer, Department of Bioinformatics, H Lundbeck A/S. 
Full documentation is given in documents/index.html

## Main classes and functions

### DA:
A dopamine system. Creates a representaion of 100 DA neurons of either mesolimbic or nirostriatal projections. 
The dopamine system generates extracellular dopamine levels in two compartments, somatodendritic and terminal areas. 
It has two D2-receptor feedback systems.

### PostSynapticNeuron
Can either be D1-MSN or D2-MSN. The main difference is how DA couples to AC5 and hence regulates intracellular cAMP. 


## Examples
Running simulations using user-defined firing rate and update D1 and D2  postsynaptic neurons:

```python
import numpy as np
from DopamineToolbox import DA, PostSynapticNeuron

#Create mesolimbic DA system:
da = DA("vta")
print(da)

d1 = PostSynapticNeuron('d1');
print(d1)

d2 = PostSynapticNeuron('d2');
print(d2)

#Decide how many times to update the systems:
niterations = 10

#Select timestep in integration. dt = 0.01 seems safe.... 
dt = 0.01;

#Create firing rate that we use to drive release from the DA neurons:
NU =  np.random.normal(loc = 5, size = niterations)
NU[NU < 0] = 0; #Make sure it is non-negative


for k in range(niterations):
    #Update DA with firing rate as input: 
    da.update(dt, NU[k]);  
    #Update post neurons with DA as input:
    d1.updateNeuron(dt, da.Conc_DA_term)
    d2.updateNeuron(dt, da.Conc_DA_term)
    #Print the results
    print('term DA =', da.Conc_DA_term, '\t Soma DA =', da.Conc_DA_soma, '\t D1-cAMP = ', d1.cAMP, ndigits, '\t D2-cAMP = ', d2.cAMP)
```
