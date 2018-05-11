# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 15:11:24 2018

@author: jakd
"""


import matplotlib.pyplot as plt
import numpy as np
import time

from DopamineToolbox import TerminalFeedback

print('Default')
d2 = TerminalFeedback( 3.0, 0.3e-2, 0.3)
print('Occ:', d2.occupancy)
print('gain:', d2.gain())
print('\n') 


print('\n With antagonist')
d2 = TerminalFeedback( 3.0, [0.3e-2, 0.1], [0.3, 0.4], occupancy = [0, 1], efficacy = [0,0])
print('Occ:', d2.occupancy)
print('gain:', d2.gain())
print('activation', d2.activity())

print('\n') 

print('\n With agonist')
d2 = TerminalFeedback( 3.0, [0.3e-2, 0.1], [0.3, 0.4], occupancy = [0, 1], efficacy = [1,1])
print('Occ:', d2.occupancy)
print('gain:', d2.gain())
print('activation', d2.activity())

#%%

tit = "Test1: Constant dopamine, and increasing agonist..."

d2 = TerminalFeedback( 3.0, [0.3e-2, 0.1], [0.3, 0.4], occupancy = [0, 0], efficacy = [1,1])

dt = 0.01;
nupdates = int(1e5);
t = dt*np.arange(0, nupdates)
occ = np.zeros([2, nupdates])
act = np.zeros(nupdates)
gay = np.zeros(nupdates)


C1 = 40.0*np.ones(nupdates);
C2 = .0*np.ones(nupdates);

C2[t > 250] = 2


C2[t > 500] = 4

for k in range(nupdates) :
    d2.updateOccpuancy(dt, [C1[k], C2[k]])
    occ[:,k] = d2.occupancy;
    act[k] = d2.activity();
    gay[k] = d2.gain();
   

plt.close(1)
plt.figure(1)
plt.subplot(2,1,1)
plt.plot(t, occ[0], t, occ[1])
plt.title(tit)
plt.ylabel('occpuancy')
plt.subplot(2,1,2)
plt.plot(t, gay)
plt.ylabel('gain')
plt.xlabel('time(s)')


#%%

tit = "Test2: Constant dopamine, and increasing ANTagonist..."

d2 = TerminalFeedback( 3.0, [0.3e-2, 0.1], [0.3, 0.4], occupancy = [0, 0], efficacy = [1,0])

dt = 0.01;
nupdates = int(1e5);
t = dt*np.arange(0, nupdates)
occ = np.zeros([2, nupdates])
act = np.zeros(nupdates)
gay = np.zeros(nupdates)


C1 = 40.0*np.ones(nupdates);
C2 = .0*np.ones(nupdates);

C2[t > 250] = 2


C2[t > 500] = 4

for k in range(nupdates) :
    d2.updateOccpuancy(dt, [C1[k], C2[k]])
    occ[:,k] = d2.occupancy;
    act[k] = d2.activity();
    gay[k] = d2.gain();
   

plt.close(2)
plt.figure(2)
plt.subplot(2,1,1)
plt.plot(t, occ[0], t, occ[1])
plt.title(tit)
plt.ylabel('occpuancy')
plt.subplot(2,1,2)
plt.plot(t, gay)
plt.ylabel('gain')
plt.xlabel('time(s)')