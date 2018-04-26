# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 12:36:16 2018

@author: jakd
Load DA simulation class from MouseClassV5 and run a small simulation
"""

import matplotlib.pyplot as plt
import numpy as np
from MouseClassV5 import DASimulation


sim1 = DASimulation()

Nmax = 10000;

sim1.Results.terminalDA = np.zeros(Nmax)
#sim1.Terminal.DA = 50
#sim1.Soma.DA = 500

for k in range(0, Nmax):
    sim1.UpdateDA();
    sim1.Results.terminalDA[k] = sim1.Terminal.DA
    
plt.plot(sim1.Results.terminalDA)

