# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 12:16:37 2018

@author: jakd
"""
import matplotlib.pyplot as plt
from MouseClassV5 import Mouse, Experiment
#
M1 = Mouse();
#fir
#M1.Firing.nu_tonic_baseline = 4;
#E = Experiment(1000, [10, 40], [100, 0], [10, 10], [500])
##E = Experiment(5000, [15, 30, 60, 120], [200, 0, 0, 0], [15, 40, 2, 2], [500, 1000, 2000])
E = Experiment(500, [10], [100] )
#
M1.RunSchedule(E)


M1.CleanUpResults()
#M1.savedata('LowDA_FR20.txt')

NStates = len(E.crit)
plt.close('all')

plt.figure(1); plt.plot(M1.Results.time, M1.Results.NUeff);plt.show()
plt.figure(2); plt.plot(M1.Results.time, M1.Results.terminalDA);plt.show()
plt.figure(3); plt.plot(M1.Results.time, M1.Results.P_State + 1); plt.ylim(0.9, NStates +1.1);plt.show()



plt.figure(30); 
for k in range(0, len(E.crit)): 
    plt.plot(M1.Results.time, M1.Results.P_Epress[k]);
plt.title('E press')
plt.show()

plt.figure(31); 
for k in range(0, len(E.crit)): 
    plt.plot(M1.Results.time, M1.Results.P_Psurv[k]);
plt.title('Survival probability')
plt.show()

plt.figure(32); 
for k in range(0, len(E.crit)): 
    plt.plot(M1.Results.time, M1.Results.P_Eloss[k]);
plt.title('Loss pr trial')

plt.show()

plt.figure(33); 
for k in range(0, len(E.crit)): 
    plt.plot(M1.Results.time, M1.Results.P_Totalloss[k]);
plt.title('Total loss pr reward')

plt.show()


plt.figure(34); 
for k in range(0, len(E.crit)): 
    plt.plot(M1.Results.time, M1.Results.P_Leverutility[k]);
plt.title('Utility of lever')

plt.show()


plt.figure(35); 
plt.plot(M1.Results.time, M1.Results.P_cumReward)
plt.title('Cumulative Reward')
plt.show()

plt.figure(136); 
plt.plot(M1.Results.time, M1.Results.P_cumPress)

plt.title('Cumulative Presses')


M1.DisplayResults(E)
