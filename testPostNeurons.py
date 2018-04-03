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
print(d1.DA_receptor.k_on)
print(d1.DA_receptor.k_off)

print("Occ:", d1.DA_receptor.occupancy)
print("Effi:", d1.DA_receptor.efficacy)
print("acti:", d1.DA_receptor.activity())


dt = 0.01;
nupdates = 100;

for k in range(nupdates):
    d1.updateNeuron(dt, [123, 239])

print("Occ:", d1.DA_receptor.occupancy)
print("Effi:", d1.DA_receptor.efficacy)
print("acti:", d1.DA_receptor.activity())
