# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 13:48:34 2018

@author: jakd
"""

import matplotlib as plt
import numpy as np
import time

from DopamineNeuronClass import DA, D1MSN, D2MSN

da = DA("vta")
print(DA())

d1 = D1MSN(1000);

print(d1)

d2 = D2MSN(1000);
print(d2)

"Set random number seed to 1234"
S = 1234;
np.random.seed(S);

nprints = 10
ndigits = 5;
dt = 0.01000;
olddump = np.load('datadumptestdata.dat.npy')
newdump = np.zeros([nprints, 6]);


for k in range(nprints):
    da.update(dt, 10); 'Extra high firing rate is used here' 
    d1.updateCAMP(dt, da.Conc_DA_term)
    d2.updateCAMP(dt, da.Conc_DA_term)
    print('term DA =', round(da.Conc_DA_term, ndigits), '\t Soma DA =', round(da.Conc_DA_soma, ndigits), '\t D1-cAMP = ', round(d1.cAMP, ndigits), '\t D2-cAMP = ', round(d2.cAMP, ndigits))
    newdump[k] = [da.Conc_DA_term, da.Conc_DA_soma, da.D2term.gain(), da.D2soma.gain(), d1.cAMP, d2.cAMP]

#np.save('datadumptestdata.dat', newdump)

print('\n\n')
print('All elements in old dump are found:', np.all(olddump == newdump))
print('\n\n')

nupdates = int(3.17e4)
t1 = time.time()
for k in range(nupdates):
    da.update(dt); 'Extra high firing rate is used here' 
    d1.updateCAMP(dt, da.Conc_DA_term)
    d2.updateCAMP(dt, da.Conc_DA_term)
t2 = time.time();
print('Time for ' + str(nupdates) + ' updates = ' + str(t2 - t1))

