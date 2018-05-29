# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 13:48:34 2018

@author: jakd
"""

import matplotlib.pyplot as plt
import numpy as np
import time

from DopamineToolbox import DA, PostSynapticNeuron, Drug

#%%

da = DA("vta")
print(DA())

d1 = PostSynapticNeuron('d1');

print(d1)

d2 = PostSynapticNeuron('d2');
print(d2)

"Set random number seed to 1234"
S = 1234;
np.random.seed(S);

nprints = 10
ndigits = 5;
dt = 0.01000;
olddump = np.load('M:/Python/Py-files/Dopamine-simulation-Tools/data/datadumptestdata.dat.npy')
newdump = np.zeros([nprints, 6]);


for k in range(nprints):
    da.update(dt, 10); 'Extra high firing rate is used here' 
    d1.updateNeuron(dt, da.Conc_DA_term)
    d2.updateNeuron(dt, da.Conc_DA_term)
    print('term DA =', np.round(da.Conc_DA_term, ndigits), '\t Soma DA =', np.round(da.Conc_DA_soma, ndigits), '\t D1-cAMP = ', np.round(d1.cAMP, ndigits), '\t D2-cAMP = ', np.round(d2.cAMP, ndigits))
    newdump[k] = [da.Conc_DA_term, da.Conc_DA_soma, da.D2term.gain(), da.D2soma.gain(), d1.cAMP, d2.cAMP]

#np.save('M:/Python/Py-files/Dopamine-simulation-Tools/data/datadumptestdata.dat.npy', newdump)

print('\n\n')
print('All elements in old dump are found:', np.all(olddump == newdump))
print('\n\n')

nupdates = int(1e4)
t1 = time.time()
for k in range(nupdates):
    da.update(dt); 'Extra high firing rate is used here' 
    d1.updateNeuron(dt, da.Conc_DA_term)
    d2.updateNeuron(dt, da.Conc_DA_term)
t2 = time.time();
print('Time for ' + str(nupdates) + ' updates = ' + str(t2 - t1))
print('\n\n')


#%% TESTING Haloperidol 



HAL = Drug('haloperidol', 'D2R', k_off = 0.65/60,  k_on = 2.13/60,  efficacy = 0)
da_hal = DA('vta', HAL)


d2_hal = PostSynapticNeuron('d2', HAL)

t1 = time.time()
Chal = [0 for k in range(nupdates)] + [10 for k in range(nupdates)];

da_hal.daout = np.zeros(2*nupdates)
da_hal.nuout = np.zeros(2*nupdates)
d2_hal.actout = np.zeros(2*nupdates)
d2_hal.campout = np.zeros(2*nupdates)


for k in range(2*nupdates):
    da_hal.update(dt, Conc = [1, Chal[k]])
    d2_hal.updateNeuron(dt, [da_hal.Conc_DA_term, Chal[k]])
    da_hal.daout[k] = da_hal.Conc_DA_term
    da_hal.nuout[k] = da_hal.nu;
    d2_hal.actout[k] = d2_hal.DA_receptor.activity();
    d2_hal.campout[k] = d2_hal.cAMP;
    
    
    
t2 = time.time();
print('Time for ' + str(nupdates) + ' updates = ' + str(t2 - t1))
print('\n\n')

print(da_hal)

#%%

timeax = np.arange(0, 2*nupdates*dt, dt)

plt.close('all')
plt.figure(1)
plt.plot(timeax, da_hal.daout)