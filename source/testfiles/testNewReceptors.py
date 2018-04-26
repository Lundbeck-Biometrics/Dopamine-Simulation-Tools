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


D2R = receptor(2.0, 0.1 )


dt = 0.01;
nupdates = int(5e3);
t = dt*np.arange(0, nupdates)
occ = np.zeros(nupdates)
act = np.zeros(nupdates)


for k in range(nupdates) :
    D2R.updateOccpuancy(dt, 1)
    occ[k] = D2R.occupancy;
    act[k] = D2R.activity();
   

plt.close(1)
plt.figure(1)
plt.subplot(2,1,1)
plt.plot(t, occ)
plt.title("test1: default settings. One ligand")
plt.ylabel('occupancy')
plt.subplot(2,1,2)
plt.plot(t, act)
plt.ylabel('activity')
plt.xlabel('time (s)')

#%% "§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§"

D2R = receptor(k_on = [2.0, 3], k_off = [1,0.10], occupancy = [0, 0.9] )


tit = "Test2: We create receptors with two ligands but nr2 is decaying to 0"
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
plt.title(tit)
plt.ylabel('occupancy')
plt.subplot(2,1,2)
plt.plot(t, act)
plt.ylabel('activity')
plt.xlabel('time (s)')

#%%
tit = "Test3: We create receptors with two ligands\n but nr2 is decaying to 0 and is antagonist"
D2R = receptor(k_on = [2.0, 3], k_off = [1,0.10], occupancy = [0, 1] , efficacy = [1, 0])



dt = 0.01;
nupdates = int(5e3);
t = dt*np.arange(0, nupdates)
occ = np.zeros([2, nupdates])
act = np.zeros(nupdates)


for k in range(nupdates) :
    D2R.updateOccpuancy(dt, [1, 0])
    occ[:,k] = D2R.occupancy;
    act[k] = D2R.activity();
   

plt.close(3)
plt.figure(3)
plt.subplot(2,1,1)
plt.plot(t, occ[0,:], t, occ[1,:])
plt.title(tit)
plt.ylabel('occupancy')
plt.subplot(2,1,2)
plt.plot(t, act)
plt.ylabel('activity')
plt.xlabel('time (s)')

#%%
tit = "Test4: We create receptors with two ligands\n but nr2 is steady and is antagonist"
D2R = receptor(k_on = [2.0, 3], k_off = [1,0.10], occupancy = [0, 0] , efficacy = [1, 0])



dt = 0.01;
nupdates = int(5e3);
t = dt*np.arange(0, nupdates)
occ = np.zeros([2, nupdates])
act = np.zeros(nupdates)


for k in range(nupdates) :
    D2R.updateOccpuancy(dt, [1, 1])
    occ[:,k] = D2R.occupancy;
    act[k] = D2R.activity();
   

plt.close(4)
plt.figure(4)
plt.subplot(2,1,1)
plt.plot(t, occ[0,:], t, occ[1,:])
plt.title(tit)
plt.ylabel('occupancy')
plt.subplot(2,1,2)
plt.plot(t, act)
plt.ylabel('activity')
plt.xlabel('time (s)')

#%%

#%%
tit = "Test5: We create receptors with two ligands\n Nr2 is steady antagonist. Nr displaces"
D2R = receptor(k_on = [2.0, 3], k_off = [1,0.10], occupancy = [0, 0] , efficacy = [1, 0])



dt = 0.01;
nupdates = int(1e5);
t = dt*np.arange(0, nupdates)
occ = np.zeros([2, nupdates])
act = np.zeros(nupdates)

C1 = 1.0*np.ones(nupdates);

C1[t > 250] = 10


C1[t > 500] = 50

for k in range(nupdates) :
    D2R.updateOccpuancy(dt, [C1[k], 1])
    occ[:,k] = D2R.occupancy;
    act[k] = D2R.activity();
   

plt.close(5)
plt.figure(5)
plt.subplot(2,1,1)
plt.plot(t, occ[0,:], t, occ[1,:])
plt.title(tit)
plt.ylabel('occupancy')
plt.subplot(2,1,2)
plt.plot(t, act)
plt.ylabel('activity')
plt.xlabel('time (s)')


