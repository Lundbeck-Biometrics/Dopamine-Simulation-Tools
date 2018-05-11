# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 14:45:59 2018

@author: jakd
"""

from DopamineToolbox import Drug, DA, DrugReceptorInteraction, PostSynapticNeuron

hal = Drug()
hal.D2 = DrugReceptorInteraction('hal.d2', 'd2', 123, 11, 0.)
hal.D1 = DrugReceptorInteraction('hal.d1', 'd1', 123, 234, 0.)

da = DA('vta', hal.D2)

#d1 = PostSynapticNeuron('d1', 1000, 30, 0.04, 0.1, hal.D1)
#d2 = D2MSN(1000, 30, 0.04, 0.1, hal.D2)
