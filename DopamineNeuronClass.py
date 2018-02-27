# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 09:32:34 2018

@author: jakd
"""

import numpy as np
import matplotlib.pyplot as plt

class feedback:
    "This is a feedback class"
    def __init__(self, alpha, k_on, k_off, occupancy = 0.5):
        self.alpha = alpha;
        self.k_on = k_on;
        self.k_off = k_off;
        self.occupancy = occupancy;
        self.update(0,0);#This call makes shure that the 'gain' attribute gets set
        
    def update(self, dt, C):
        self.occupancy += dt*( (1 - self.occupancy)*self.k_on*C - self.occupancy*self.k_off) 
        self.occupancy = np.max([0,self.occupancy])
        self.gain = 1/(1+self.alpha*self.occupancy)
      
    
class DA:
    """This is a dopamine class. Sets up a set of eqns that represent DA. Parameters depend on area. 
    Call like DA(""VTA"") or DA(""SNC""). Area argument is not case sensitive"""
    def __init__(self, area = "VTA"):
        self.D2term = feedback(2, 1e-2, 0.4)
        self.D2soma = feedback(0.8, 0.2, 0.6, 0)
        Original_NNeurons = 100;
        self.NNeurons = Original_NNeurons;
        self.NU_in = 5.0; #This is the parameter that determines DA levels
        
        if area.lower() == "vta":
            self.Vmax_pr_neuron = 1500.0/Original_NNeurons
            self.Gamma_pr_neuron = 2.0/Original_NNeurons #This value is deliberately high and anticipates a ~50% reduction by terminal feedback
        elif area.lower() == "snc":
            self.Vmax_pr_neuron = 4000.0/Original_NNeurons
            self.Gamma_pr_neuron = 4.0/Original_NNeurons; 
        else:
            print('You are in unknown territory')
        self.Vmax_pr_neuron_soma = 200.0/Original_NNeurons;
        self.Gamma_pr_neuron_soma = 20.0/Original_NNeurons;
        self.Conc_DA_soma = 0.0;
        self.Conc_DA_term = 0.0;
        
    """Parameters defined here are shared between all instances of the class. """
    Km = 160.0; #MicMen reuptake parameter. Affected by cocaine or methylphenidate
    k_nonDAT = 0.0; #First order reuptake constant.  
    Precurser = 1.0; # Change this to simulate L-dopa
    
    def update(self, dt, nu_in):
        "This is the update function that increments DA concentraions. Argumet is 'dt'. " 
        self.D2soma.update(dt, self.Conc_DA_soma)
        self.D2term.update(dt, self.Conc_DA_term)
        
        nu = nu_in*self.D2soma.gain;
#        print('input: ', self.NU_in, '. Effective:' , nu)
        rel = np.random.poisson(self.NNeurons*nu*dt);
        "first calculate somato dendritic DA:"
        self.Conc_DA_soma += rel*self.Gamma_pr_neuron_soma - dt*self.Vmax_pr_neuron_soma*self.NNeurons*self.Conc_DA_soma/(self.Km + self.Conc_DA_soma) - dt*self.k_nonDAT;
        "...then terminal DA"
        self.Conc_DA_term += rel*self.Gamma_pr_neuron*self.D2term.gain - dt*self.Vmax_pr_neuron*self.NNeurons*self.Conc_DA_term/(self.Km + self.Conc_DA_term)  - dt*self.k_nonDAT;
         
        
    
#test the classes:
test = DA('vta')
#create a list of class instances: 
nlist = 3;
DAlist = [DA('SNC') for i in range(nlist)]
DAlevels = [10, 30, 2309]
Nupdates = int(1e5);
Occ = np.zeros([nlist, Nupdates]);
dt = 0.001;
timeax = dt*np.arange(0,Nupdates);
NU = 5*np.ones(Nupdates);
burstindx = np.logical_and(20 <= timeax, timeax < 25);
NU[burstindx]= 20

burstindx = np.logical_and(40 <= timeax, timeax < 45);
NU[burstindx]= 0


for k in range(nlist):
    DAlist[k].NNeurons = 100 - k*40
for L in range(Nupdates):
    for k in range(nlist):
        DAlist[k].update(dt, NU[L])
        Occ[k,L]= DAlist[k].Conc_DA_term



fignum = 1;
plt.close(fignum)
plt.figure(fignum);
for k in range(nlist):
    plt.plot(timeax, Occ[k])



