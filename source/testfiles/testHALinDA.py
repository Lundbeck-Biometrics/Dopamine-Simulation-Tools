# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 10:12:02 2018

@author: jakd
"""

import matplotlib.pyplot as plt
import numpy as np
import time

from DopamineNeuronClass import DA, D1MSN, D2MSN, Drug


"Test1: DA before and after hal"

HAL = Drug();"Default creates preliminary haloperidol"

HAL.efficacy = 0;
da = DA("vta", HAL)


dt = 0.01;
nupdates = int(5e4);
t = dt*np.arange(0, nupdates)
d2occ = np.zeros([2, nupdates])
daconc = np.zeros(nupdates)
nu     = np.zeros(nupdates)
d2inhib = np.zeros(nupdates)

Chal = 0.0*np.ones(nupdates);

Chal[t > 100] = 1000


Chal[t > 500] = 2

for k in range(nupdates) :
    da.update(dt, Conc = [0, Chal[k]])
    daconc[k]  = da.Conc_DA_term;
    nu[k] = da.nu
    d2inhib[k] = da.D2term.gain();
    d2occ[:,k] = da.D2term.occupancy;
   

tit = "DA status"

plt.close(1)
plt.figure(1)
plt.subplot(2,1,1)
plt.plot(t, daconc)
plt.title(tit)
plt.ylabel('Terminal')
plt.subplot(2,1,2)
plt.plot(t, nu)
plt.ylabel('Firing rate')
plt.xlabel('time (s)')






tit = "D2 status"

plt.close(2)
plt.figure(2)
plt.subplot(2,1,1)
plt.plot(t, d2occ[0,:], t, d2occ[1,:])
plt.title(tit)
plt.ylabel('occupancy')
plt.subplot(2,1,2)
plt.plot(t, d2inhib)
plt.ylabel('D2 terminal activity')
plt.xlabel('time (s)')



