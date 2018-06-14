# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 13:48:34 2018

@author: jakd
"""

import matplotlib.pyplot as plt
import numpy as np
import time

from DopamineToolbox import DA, PostSynapticNeuron, Drug, Cholinergic

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

nupdates = int(3e4)
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
"""Constants are derived from Sykes et al: *Extrapyramidal side effects of antipsychotics are linked to their association kinetics at dopamine D2 receptors*
	Nature Communicationsvolume 8, Article number: 763 (2017); doi:10.1038/s41467-017-00716-z """

da_hal = DA('vta', HAL)


d1_hal = PostSynapticNeuron('d1')
d2_hal = PostSynapticNeuron('d2', HAL)

haldose = 0.8;
"Concentration of Haloperidol in ext fluid. Selected to give 70% occupancy"


t1 = time.time()
Chal = [0 for k in range(nupdates)] + [haldose for k in range(nupdates)];

da_hal.daout = np.zeros(2*nupdates)
da_hal.nuout = np.zeros(2*nupdates)

d2_hal.actout = np.zeros(2*nupdates)
d2_hal.campout = np.zeros(2*nupdates)
d2_hal.occout = np.zeros([2*nupdates, 2])

d1_hal.actout = np.zeros(2*nupdates)
d1_hal.campout = np.zeros(2*nupdates)
d1_hal.occout = np.zeros(2*nupdates)


for k in range(2*nupdates):
    da_hal.update(dt, Conc = [1, Chal[k]])
    d1_hal.updateNeuron(dt,  da_hal.Conc_DA_term)
    d2_hal.updateNeuron(dt, [da_hal.Conc_DA_term, Chal[k]])
    da_hal.daout[k] = da_hal.Conc_DA_term
    da_hal.nuout[k] = da_hal.nu;
    
    d1_hal.occout[k] = d1_hal.DA_receptor.occupancy;
    d1_hal.actout[k] = d1_hal.DA_receptor.activity();
    d1_hal.campout[k] = d1_hal.cAMP;
    
    d2_hal.occout[k] = d2_hal.DA_receptor.occupancy;
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
plt.title('DA conc at terminals')

plt.figure(2)
plt.plot(timeax, da_hal.nuout)
plt.title('DA cell firing rate')


plt.figure(3)
plt.plot(timeax, d2_hal.campout)
plt.title('D2 msn cAMP')

plt.figure(4)
line = plt.plot(timeax, d2_hal.occout)
line[0].set_label('DA')
line[1].set_label('HAL')
plt.title('D2 receptor occupancy')
plt.legend()



plt.figure(31)
plt.plot(timeax, d1_hal.campout)
plt.title('D1 msn cAMP')

plt.figure(41)
line = plt.plot(timeax, d1_hal.occout)

plt.title('D1 receptor occupancy')
plt.legend()

#%%

Nit = int(25/dt);
Chol = Cholinergic(k_AChE=10, gamma = 100)
#Chol.D2soma.k_off = np.array([10])
#Chol.D2soma.alpha = 30
Chol.nuout = np.zeros(Nit)
Chol.AChout = np.zeros(Nit)
Chol.occ = np.zeros(Nit)




da = DA('VTA')
da.daout = np.zeros(Nit)
da.nuout = np.zeros(Nit)
#Create DA burst and pause:

d1 = PostSynapticNeuron('d1')
d1.campout = np.zeros(Nit)
d1.ac5out = np.zeros(Nit)
d1.Other_receptor.bmax = 20

d1_2 = PostSynapticNeuron('d1')
d1_2.campout = np.zeros(Nit)
d1_2.ac5out = np.zeros(Nit)
d1_2.Other_receptor.bmax = 20


NU_da = 5*np.ones(Nit);
tburstpause = np.arange(0, Nit)*dt;

tburst  = 7;
dtburst = 0.25;
bindx = np.logical_and(tburstpause > tburst, tburstpause < tburst+dtburst);
NU_da[bindx]=20;

tburst  = 15;
dtburst = 1;
bindx = np.logical_and(tburstpause > tburst, tburstpause < tburst+dtburst);
NU_da[bindx]=0;

plt.figure(100)
plt.plot(tburstpause, NU_da)


for k in range(Nit):
    da.update(dt, NU_da[k])
    Chol.update(dt, da.Conc_DA_term)
#    d1.updateNeuron(dt, 50, Chol.Conc_ACh)
#    d1.updateNeuron(dt, da.Conc_DA_term, 50)
    d1.updateNeuron(dt, da.Conc_DA_term, Chol.Conc_ACh)
    d1_2.updateNeuron(dt, da.Conc_DA_term, 45 )
    
    da.daout[k] = da.Conc_DA_term
    da.nuout[k]= da.nu;
    Chol.AChout[k] = Chol.Conc_ACh
    Chol.nuout[k] = Chol.nu
    Chol.occ[k] = Chol.D2soma.occupancy
    

    d1.campout[k] = d1.cAMP
    d1.ac5out[k] = d1.AC5()
    
    d1_2.campout[k] = d1_2.cAMP
    d1_2.ac5out[k]  = d1_2.AC5()
   


Nreps = 10;

da_hal = DA('VTA', HAL)
da_hal.daout = np.zeros(Nit*Nreps)
da_hal.nuout = np.zeros(Nit*Nreps)

Chol_hal = Cholinergic(10, 100, HAL)
Chol_hal.nuout = np.zeros(Nit*Nreps)
Chol_hal.AChout = np.zeros(Nit*Nreps)
Chol_hal.occ = np.zeros([Nit*Nreps, 2])

d1_hal = PostSynapticNeuron('d1') #<- no direct interaction between 
d1_hal.campout = np.zeros(Nit*Nreps)
d1_hal.ac5out = np.zeros(Nit*Nreps)
d1_hal.Other_receptor.bmax = 20
thal = np.arange(0, Nit*Nreps)*dt;

for k in range(Nit*Nreps):
  
    da_hal.update(dt, NU_da[k%Nit],  Conc = np.array([0, haldose])) #<- Firing rate repeats in a loop
  
    Chol_hal.update(dt, [da_hal.Conc_DA_term, haldose])
#    d1.updateNeuron(dt, 50, Chol.Conc_ACh)
#    d1.updateNeuron(dt, da.Conc_DA_term, 50)
    d1_hal.updateNeuron(dt, da_hal.Conc_DA_term, Chol_hal.Conc_ACh)
    
    da_hal.daout[k] = da_hal.Conc_DA_term
    da_hal.nuout[k]= da_hal.nu;
    
    Chol_hal.AChout[k] = Chol_hal.Conc_ACh
    Chol_hal.nuout[k] = Chol_hal.nu
    Chol_hal.occ[k] = Chol_hal.D2soma.occupancy
    d1_hal.campout[k] = d1_hal.cAMP
    d1_hal.ac5out[k] = d1_hal.AC5()
       
#%%
plt.close('all')
plt.figure(101)
plt.plot(tburstpause, da.daout)
plt.xlabel('time(s)')
plt.ylabel('DA (nM)')
plt.title('DA inputs to Chol Neurons')


plt.figure(102)
plt.plot(tburstpause, Chol.nuout)
plt.xlabel('time(s)')
plt.ylabel('ACh Firing (Hz)')
plt.title('Chol Neuron firing rate')


plt.figure(1020)
plt.plot(tburstpause, Chol.occ)
plt.xlabel('time(s)')
plt.ylabel('Chol-D2 occupancy')
plt.title('DA binding to chol')


plt.figure(103)
plt.plot(tburstpause, Chol.AChout)
plt.xlabel('time(s)')
plt.ylabel('ACh conc (nM)')
plt.title('Striatal ACh concentration')


plt.figure(200)
plt.plot(tburstpause, d1_2.ac5out, tburstpause, d1.ac5out, tburstpause, 0*tburstpause, '--k')
plt.title('D1-AC5')
plt.xlabel('time (s)')
plt.ylim([-0.2, 4])


#%%
    
plt.figure(1101)
plt.plot(thal, np.tile(da.daout, Nreps),thal,  da_hal.daout)
plt.xlabel('time(s)')
plt.ylabel('DA (nM)')
plt.title('DA inputs to Chol Neurons')

#%%

plt.figure(102)
plt.plot(thal, Chol_hal.nuout)
plt.xlabel('time(s)')
plt.ylabel('ACh Firing (Hz)')
plt.title('Chol Neuron firing rate')


plt.figure(1020)
plt.subplot(2,1,1)
plt.plot(thal, Chol_hal.occ[:,1])
plt.subplot(2,1,2)
plt.plot(thal, Chol_hal.occ[:,0])
plt.xlabel('time(s)')
plt.ylabel('Chol-D2 occupancy')
plt.title('DA binding to chol')

#%%
plt.figure(1103)
plt.plot( thal, np.tile(Chol.AChout, Nreps), thal, Chol_hal.AChout)
plt.xlabel('time(s)')
plt.ylabel('ACh conc (nM)')
plt.title('Striatal ACh concentration: effect of Hal')


print(np.mean(Chol_hal.AChout))

print(np.mean(Chol.AChout))

#%%

plt.figure(2000)
plt.plot(thal, np.tile(d1.ac5out, Nreps), thal, 0*thal, '--k', thal, d1_hal.ac5out)
plt.title('D1-AC5')
plt.xlabel('time (s)')
plt.ylim([-0.2, 5])
