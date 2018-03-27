# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 15:11:24 2018

@author: jakd
"""


import matplotlib.pyplot as plt
import numpy as np
import time

from DopamineNeuronClass import TerminalFeedback

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
d2 = TerminalFeedback( 3.0, [0.3e-2, 0.1], [0.3, 0.4], occupancy = [0, 1], efficacy = [0,1])
print('Occ:', d2.occupancy)
print('gain:', d2.gain())
print('activation', d2.activity())
