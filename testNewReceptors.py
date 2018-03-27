# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 10:10:46 2018

@author: jakd
"""

import matplotlib.pyplot as plt
import numpy as np
import time

from DopamineNeuronClass import receptor
#%%

"Test1: We create receptors with one ligand, the default"
D2R = receptor( )

#D2R = receptor(k_on = [2.0, 3], k_off = [0.10,0.10], occupancy = [0, 0] )



dt = 0.01;
nupdates = int(5e3);
t = dt*np.arange(0, nupdates)
occ = np.zeros([2, nupdates])
act = np.zeros(nupdates)


for k in range(nupdates) :
    D2R.updateOccpuancy(dt, 6)#, [1, 0])
    occ[:,k] = D2R.occupancy;
    act[k] = D2R.activity();
   

plt.close(2)
plt.figure(2)
plt.subplot(2,1,1)
plt.plot(t, occ[0,:], t, occ[1,:])

#%% "§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§"

D2R = receptor(k_on = [2.0, 3], k_off = [0.10,0.10], occupancy = [0, 0] )


"Test1: We create receptors with two ligands but nr2 is always 0"
dt = 0.01;
nupdates = int(5e3);
t = dt*np.arange(0, nupdates)
occ = np.zeros([2, nupdates])
act = np.zeros(nupdates)


for k in range(nupdates) :
    D2R.updateOccpuancy(dt, [1, 0])
    occ[:,k] = D2R.occupancy;
    act[k] = D2R.activity();
   

plt.close(2)
plt.figure(2)
plt.subplot(2,1,1)
plt.plot(t, occ[0,:], t, occ[1,:])


plt.subplot(2,1,2)
plt.plot(t, act)

