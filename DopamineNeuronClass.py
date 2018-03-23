# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 09:32:34 2018

@author: jakd
"""

import numpy as np

class PostSynapticNeuron:
    """This is going to represent D1- or D2 MSN's. They respond to DA conc and generate cAMP. They 
    have gain and threshold for activating cascades.  
    Future versions may have agonist and antagonist neurotransmitters"""
    def __init__(self, EC50, Gain , Threshold , kPDE ):
        self.EC50 = EC50;
        self.Gain = Gain;
        self.Threshold = Threshold;
        self.kPDE = kPDE;
        self.cAMP = 0;
    
    cAMPlow = 0.1;
    cAMPhigh = 10;
    def occupancy(self, C_DA):
        return C_DA/(self.EC50 + C_DA);
    def AC5(self, C):
        "Placeholder function. Must overridden in D1 or D2MSN -classes where AC is controlled differently"
        return 0
    def updateCAMP(self, dt, C_DA):
        self.cAMP += dt*(self.AC5(C_DA) - self.kPDE*self.cAMP)
    
        
#class receptor:
#    """This is a basic class for all receptors"""
#    def __itit__(self, k_on, k_off, occupancy = 0):
#        self.k_on = k_on;
#        self.k_off = k_off;
#        self.occupancy  = occupancy; 
#    def stepupdate(self, dt, )
    
    
class D1MSN(PostSynapticNeuron):
    
    def __init__(self, EC50, Gain = 30, Threshold = 0.04, kPDE = 0.10):
        PostSynapticNeuron.__init__(self, EC50, Gain, Threshold, kPDE)
    def AC5(self, C_DA):
        return self.Gain*(self.occupancy(C_DA) - self.Threshold)*(self.occupancy(C_DA) > self.Threshold)
    def updateG_and_T(self, dt, cAMP_vector):
        "batch updating gain and threshold. Use a vector of cAMP values. dt is time step in update vector"
        dT = np.heaviside(cAMP_vector - self.cAMPlow, 0.5) - 0.99;
        self.Threshold += np.sum(dT)*dt/cAMP_vector.size
        #print("T=" , self.Threshold)
        dT = - np.heaviside(cAMP_vector - self.cAMPhigh, 0.5) + 0.01;
        self.Gain += 10*np.sum(dT)*dt/cAMP_vector.size
        #print("G=" , self.Gain)
    
class D2MSN(PostSynapticNeuron):
    "Almost like D1 MSNs but cDA regulates differently and threshold is also updated differently"
  
    def __init__(self, EC50, Gain = 50, Threshold = 0.06, kPDE = 0.10):
        PostSynapticNeuron.__init__(self, EC50, Gain, Threshold, kPDE)
    def AC5(self, C_DA):
        return self.Gain*(self.Threshold - self.occupancy(C_DA))*(self.occupancy(C_DA) < self.Threshold)
    def updateG_and_T(self, dt, cAMP_vector):
        "batch updating gain and threshold. Use a vector of cAMP values. dt is time step in update vector"
        dT = np.heaviside(cAMP_vector - self.cAMPlow, 0.5) - 0.99;
        "NOTE '-' SIGN BELOW"
        self.Threshold -= np.sum(dT)*dt/cAMP_vector.size; 
        #print("T=" , self.Threshold)
        dT = - np.heaviside(cAMP_vector - self.cAMPhigh, 0.5) + 0.01;
        self.Gain += 10*np.sum(dT)*dt/cAMP_vector.size
        #print("G=" , self.Gain)
     
    

class feedback:
    """This is a feedback class. Takes arguments: 
        alpha: list of two numbers
        k_on: onrate, typically 1e-2
        k_off: offrate, 1 or 10 or whatever. 
        """
    def __init__(self, alpha, k_on, k_off, occupancy = 0.5):
        "The feedback. "
        self.alpha = alpha;
        self.k_on = k_on;
        self.k_off = k_off;
        self.occupancy = occupancy;
        self.update(0,0);#This call makes shure that the 'gain' attribute gets set
        
    def update(self, dt, C):
        self.occupancy += dt*( (1 - self.occupancy)*self.k_on*C - self.occupancy*self.k_off) 
        self.occupancy = np.maximum(0, self.occupancy)

class TerminalFeedback(feedback):
    def __init__(self, alpha, k_on, k_off, occupancy = 0.5):
        feedback.__init__(self, alpha, k_on, k_off, occupancy)
    
    def gain(self):
        "Make sure gain is bigger than 0. "
        return 1/(1 + self.alpha*self.occupancy)    
      
class SomaFeedback(feedback):
    def __init__(self, alpha, k_on, k_off, occupancy = 0.5):
        feedback.__init__(self, alpha, k_on, k_off, occupancy)
    
    def gain(self):
        "Make sure gain is bigger than 0. "
        return np.maximum(0, self.alpha*self.occupancy);
    
 
class DA:
    """This is a dopamine class. Sets up a set of eqns that represent DA. Parameters depend on area. 
    Create instances by calling DA(""VTA"") or DA(""SNC""). Area argument is not case sensitive.
    Update method uses forward euler steps. Works for dt < 0.005 s"""
    def __init__(self, area = "VTA"):
        self.D2term = TerminalFeedback(3.0, 0.3e-2, 0.3)
        self.D2soma = SomaFeedback(10.0, 1e-2, 10, 0)
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
        
        self.nu = np.maximum(nu_in - self.D2soma.gain()*(1 - e_stim), 0);
#       self.nu = nu_in*self.D2soma.gain*int(not(e_stim)) + nu_in*int(e_stim);
#        print('input: ', self.NU_in, '. Effective:' , nu)
        rel = np.random.poisson(self.NNeurons*self.nu*dt);
        "first calculate somato dendritic DA:"
        self.Conc_DA_soma += self.Precurser*rel*self.Gamma_pr_neuron_soma - dt*self.Vmax_pr_neuron_soma*self.NNeurons*self.Conc_DA_soma/(self.Km + self.Conc_DA_soma) - dt*self.k_nonDAT;
        "...then terminal DA"
        self.Conc_DA_term += self.Precurser*rel*self.Gamma_pr_neuron*self.D2term.gain() - dt*self.Vmax_pr_neuron*self.NNeurons*self.Conc_DA_term/(self.Km + self.Conc_DA_term)  - dt*self.k_nonDAT;
         
