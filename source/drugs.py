# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 15:20:17 2021

@author: JAKD
"""
import numpy as np


class DrugReceptorInteraction:
    """
    Representing interaction between a drug and a receptor. 
    
    :param name: Name of drug
    :type name: str
    :param target: Name of receptor target for this interaction
    :type target: str
    :param k_on: onrate of drug to *target* in s\:sup:`-1`
    :type k_on: float
    :param k_off: off-rate of drug to *target* in s\ :sup:`-1`
    :type k_off: float
    :param efficacy: Efficacy of drug to ativate *target*. See :class:`receptor` for more information. 
    :type efficacy: 0<= float <= 1
   
    """
    def __init__(self, name, target, k_on, k_off, efficacy):
        self.name = name;
        self.target = target;
        self.k_on = k_on;
        self.k_off = k_off;
        self.efficacy = efficacy;
        
    def __str__(self):
        class_string = \
        'Drug-receptor interaction. \n'\
        'Name:\t\t' + self.name + '\n' +\
        'Efficacy:\t' + str(self.efficacy) + '\n' +\
        'Onrate:\t\t' + str(self.k_on) + ' (s nM)^{-1}' + '\n' \
        'Offrate:\t' + str(self.k_off) + ' s^{-1}' + '\n'
        
        return class_string


class Drug(DrugReceptorInteraction):
    """
    Class that is used to simulate presence of other ligands. Default has only one type of receptor to interact with. 
    See below in case you want to simulate a drug with several interactions. 
    
    Examples::
        
       >>#create drug instance with default on and off rates:
       >>mydrug = Drug('secret DA agonist', 'D2R', efficacy = 1)
        
    """
    
    def __init__(self, name = 'Default Agonist', target = 'D2R', k_on = 0.01, k_off = 1.0, efficacy = 1.0):
        DrugReceptorInteraction.__init__(self, name, target, k_on, k_off, efficacy)
        print("Creating Drug-receptor class. More receptor interactions can be added manually! \nUse <name>.<target> = DrugReceptorInteraction(\'name.tagret\', target, kon, koff, efficacy)")
    
    def Concentration(self, t,  dose = 1, t_infusion = 0, k12 = 0.3/60, k21 = 0.0033, k_elimination =0.0078):
        """
        Calculates drug concentration in brain using two-compartment PK. Default values are for cocaine PK as estimated 
        in `Pan, Menacherry, Justice; J Neurochem, 1991 <https://doi.org/10.1111/j.1471-4159.1991.tb11425.x>`_. 
        But here the variaables are transformed from minues to seconds. 
        
        :param t: timepoint to calculate drug concentration. Can be scalar or array. If *t< t_infusion* the concentration will be 0. 
        :type t: scalar or array
        :param dose: Dose multiplier. 
        :type dose: float
        :param t_infusion: Time of infusion. Default *t_infusion* = 0. 
        :type t_infusion: float
        :param k12: rate constant in s\ :sup:`-1` for passage from bloodstream into brain (compartment 2)
        :type k12: float
        :param k21: rate constant in s\ :sup:`-1` for passage from brain to bloodstream (compartment 1)
        :type k21: float
        :param k_elimination: rate constant in\ :sup:`-1` for drug elimination from bloodstream
        :type k_elimination: float
        :return: Drug concentration at time t. numpy array of same size as input parameter *t*. If *t< t_infusion* the concentration will be 0. 
        
        
        .. figure:: Pan_1991_Fig1.jpg
           :scale: 50 %
           :alt: Pan et al, 1991, Figure 1. 
           :align: center

           Figure 1 from `Pan et al, J Neurochem, 1991 <https://doi.org/10.1111/j.1471-4159.1991.tb11425.x>`_ showing compartments. The first compartment 'BODY CAVITY' is not used. 
        
        .. Note:: If *t* < *t_infusion* the concentraion is 0. If *t* is an array of times, the output concentration for *t* < *t_infusion* are 0. 
        """

        sumk = k12+k21+k_elimination
        D = np.sqrt(sumk**2 - 4*k21*k_elimination);
        a = 0.5*(sumk + D);
        b = 0.5*(sumk - D);
        
        T = np.array(t-t_infusion)
        Tdrug = T[T>0]
 
        C = dose*k12/(a-b)*( np.exp(-b*(Tdrug) ) - np.exp(-a*(Tdrug)) );
        pre_padding = np.zeros(T.size - Tdrug.size)
        
        c2 = np.concatenate( (pre_padding, C) )

        return c2
 
