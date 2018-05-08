# Dopamine-Simulation-Tool
This is a toolbox for making simulations of dopamine (DA) signaling in python. It is developed under Python 3. 

It is authored by Jakob Dreyer, Department of Bioinformatics, H Lundbeck A/S. 

Full documentation is given in documents/index.html

## Main classes and functions

### DA()
A dopamine system. Creates a representaion of 100 DA neurons of either mesolimbic or nirostriatal projections. The dopamine system generates extracellular dopamine levels in two compartments, somatodendritic and terminal areas. It has two D2-receptor feedback systems


## Importing
For example:

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


nprints = 10
ndigits = 5;
dt = 0.01;

NU =  np.random.normal(loc = 5, size = nprints)

for k in range(nprints):
    #Update DA with firing rate as input: 
    da.update(dt, NU[k]);  
    #Update post neurons with DA as input:
    d1.updateNeuron(dt, da.Conc_DA_term)
    d2.updateNeuron(dt, da.Conc_DA_term)
    #Print the results
    print('term DA =', np.round(da.Conc_DA_term, ndigits), '\t Soma DA =', np.round(da.Conc_DA_soma, ndigits), '\t D1-cAMP = ', np.round(d1.cAMP, ndigits), '\t D2-cAMP = ', np.round(d2.cAMP, ndigits))
```
