# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 09:32:34 2018

@author: jakd
"""

import numpy as np



class feedback:
    "This is a feedback class"
    def __init__(self, alpha, k_on, k_off, Gmin = 0, occupancy = 0.5):
        "The feedback. "
        self.alpha = alpha;
        self.k_on = k_on;
        self.k_off = k_off;
        self.occupancy = occupancy;
        self.Gmin = Gmin;
        self.update(0,0);#This call makes shure that the 'gain' attribute gets set
        
    def update(self, dt, C):
        self.occupancy += dt*( (1 - self.occupancy)*self.k_on*C - self.occupancy*self.k_off) 
        self.occupancy = np.max([0, self.occupancy])
        "Gain is defined so that occupancy = 0, means gain = 1. "
        self.gain =  (self.Gmin + 1)/(1+self.alpha*self.occupancy) - self.Gmin
        "Make sure gain is bigger than 0. "
        self.gain = np.max([0, self.gain]);
    
#class feedback_keep:
#    "This is a feedback class"
#    def __init__(self, alpha, k_on, k_off, occupancy = 0.5):
#        self.alpha = alpha;
#        self.k_on = k_on;
#        self.k_off = k_off;
#        self.occupancy = occupancy;
#        self.update(0,0);#This call makes shure that the 'gain' attribute gets set
#        
#    def update(self, dt, C):
#        self.occupancy += dt*( (1 - self.occupancy)*self.k_on*C - self.occupancy*self.k_off) 
#        self.occupancy = np.max([0, self.occupancy])
#        self.gain =  1/(1+self.alpha*self.occupancy)
#    
 
      
    
class DA:
    """This is a dopamine class. Sets up a set of eqns that represent DA. Parameters depend on area. 
    Create instances by calling DA(""VTA"") or DA(""SNC""). Area argument is not case sensitive.
    Update method uses forward euler steps. Works for dt < 0.005 s"""
    def __init__(self, area = "VTA"):
        feedrate = 0.3;
        self.D2term = feedback(3, feedrate*1e-2, feedrate*1, 0)
        self.D2soma = feedback(0.5, 1e-2, 10, 4, 0)
        Original_NNeurons = 100;
        self.NNeurons = Original_NNeurons;
#        self.NU_in = 5.0; #This is the parameter that determines DA levels
        self.nu = 0; "This attribute reports the current firing rate after sd feedback"
        self.area = area
        if area.lower() == "vta":
            self.Vmax_pr_neuron = 1500.0/Original_NNeurons
            self.Gamma_pr_neuron = 200.0/Original_NNeurons #This value is deliberately high and anticipates a ~50% reduction by terminal feedback
        elif area.lower() == "snc":
            self.Vmax_pr_neuron = 4000.0/Original_NNeurons
            self.Gamma_pr_neuron = 400.0/Original_NNeurons; 
        else:
            print('You are in unknown territory')
        self.Vmax_pr_neuron_soma = 200.0/Original_NNeurons;
        self.Gamma_pr_neuron_soma = 20.0/Original_NNeurons;
        self.Conc_DA_soma = 0.0;
        self.Conc_DA_term = 0.0;
        
   
    Km = 160.0; #MicMen reuptake parameter. Affected by cocaine or methylphenidate
    k_nonDAT = 0.0; #First order reuptake constant.  
    Precurser = 1.0; # Change this to simulate L-dopa
    
    def update(self, dt, nu_in = 5, e_stim = False):
        "This is the update function that increments DA concentraions. Argumet is 'dt'. " 
        self.D2soma.update(dt, self.Conc_DA_soma)
        self.D2term.update(dt, self.Conc_DA_term)
        "If e_stim -> True, then the firing rate overrules s.d. inhibition"
        self.nu = nu_in*self.D2soma.gain*int(not(e_stim)) + nu_in*int(e_stim);
#        print('input: ', self.NU_in, '. Effective:' , nu)
        rel = np.random.poisson(self.NNeurons*self.nu*dt);
        "first calculate somato dendritic DA:"
        self.Conc_DA_soma += self.Precurser*rel*self.Gamma_pr_neuron_soma - dt*self.Vmax_pr_neuron_soma*self.NNeurons*self.Conc_DA_soma/(self.Km + self.Conc_DA_soma) - dt*self.k_nonDAT;
        "...then terminal DA"
        self.Conc_DA_term += self.Precurser*rel*self.Gamma_pr_neuron*self.D2term.gain - dt*self.Vmax_pr_neuron*self.NNeurons*self.Conc_DA_term/(self.Km + self.Conc_DA_term)  - dt*self.k_nonDAT;
         
