# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 10:14:20 2018

@author: jakd
"""

import matplotlib.pyplot as plt
import numpy as np
import time

from DopamineNeuronClass import  D1MSN, D2MSN, Drug

HAL = Drug();
d1 = D1MSN(1000, 30, 0.05, 0.01, HAL)
d2 = D2MSN(1000, 30, 0.05, 0.01, HAL)





dt = 0.01;
nupdates = 10000;
plott = dt*np.arange(nupdates);
d1.Occlist = np.zeros([2, nupdates])
d1.actlist = np.zeros(nupdates)
d1.cAMPlist = np.zeros(nupdates)

d2.Occlist = np.zeros([2, nupdates])
d2.actlist = np.zeros(nupdates)
d2.cAMPlist = np.zeros(nupdates)

for k in range(nupdates):
    d1.updateNeuron(dt, [123, 100])
    d2.updateNeuron(dt, [123, 0])
    d1.Occlist[:,k] = d1.DA_receptor.occupancy
    d1.actlist[k]   = d1.DA_receptor.activity();
    d1.cAMPlist[k]   = d1.cAMP;
    
    d2.Occlist[:,k] = d2.DA_receptor.occupancy
    d2.actlist[k]   = d2.DA_receptor.activity();
    d2.cAMPlist[k]   = d2.cAMP;




plt.close('all')
plt.figure(1)

plt.subplot(3,1,1)
line = plt.plot(plott, d1.Occlist[0], plott, d1.Occlist[1])
line[0].set_label('DA')
line[1].set_label('Drug')

plt.legend()

plt.title('D1 cell with antagonist')
plt.subplot(3,1,2)
plt.plot(plott, d1.actlist)

plt.subplot(3,1,3)
plt.plot(plott, d1.cAMPlist)
plt.xlabel('time (s)')

plt.figure(2)

plt.subplot(3,1,1)
plt.plot(plott, d2.Occlist[0], plott, d2.Occlist[1])
plt.title('D2 cell with antagonist')
plt.subplot(3,1,2)
plt.plot(plott, d2.actlist)

plt.subplot(3,1,3)
plt.plot(plott, d2.cAMPlist)
plt.xlabel('time (s)')
