# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 15:11:41 2021

@author: JAKD
"""
import numpy as np


class receptor:
    """
    This is a basic class for all receptors. It is used by feedbackloops in :class:`DA`, :class:`D1MSN` and  :class:`D1MSN`. It can handle several ligands, and *activity* is not the same as *occupancy*. 
    If there are several ligands competing for the receptors all input parameters must arrays where the first entry is dopamine.  Default inputs will be converted into arrays of length 1. 
    The *activity* is always a scalar. 
    
    :param k_on: On-rate for ligands in units of nM\ :sup:`-1` s\ :sup:`-1`. Default is 1 (nM s)\ :sup:`-1`. 
    :type k_on: numpy array 
    :param k_off: Off-rate for ligands in units of s\ :sup:`-1`. Default is 1 s\ :sup:`-1`.
    :type k_off: numpy array
    :param occupancy: Initial occupancy of ligands. np.sum(occupany) must be <= 1. 
    :type occupancy: numpy array
    :param efficacy: Efficacy of activating receptor. See below
    
    The efficacy relates the *occupancy* of the ligand with the activation of the receptor. 
    
    ==============   ===========
    Efficacy value   Interaction
    ==============   ===========
    1                Full agonist
    0                Full antagonist
    0.-.99           Partial agonist
    ==============   ===========

    """
    def __init__(self, k_on = 1, k_off = 1, occupancy = 0, efficacy = 1):
        self.k_on = 1.0*np.array(k_on);
        self.k_off = 1.0*np.array(k_off);
        #"Two steps to assure that occupancy is seen as a vector"
        self.occupancy  = 1.0*np.array(occupancy)
        self.occupancy  *= np.ones(self.occupancy.size)
        #"Make sure that efficacy has same size as occupancy. Default efficacy is 1. "
        tempeff = 1.0*np.array(efficacy);
        if tempeff.size == 1:
            self.efficacy = tempeff*np.ones(self.occupancy.size)
        else:
            self.efficacy = tempeff
       
        
    def updateOccpuancy(self, dt, Cvec ):
        """
        This is the method that increments the occupancy. It takes a timestep, *dt*, 
        and a ligand concentration, *Cvec*, and updates the occupancy-property off the class. 
        
        :param dt: Timestep in seconds
        :type dt: float
        :param Cvec: Concentration of ligands in nM
        :type Cvec: numpy array
        """
        free = 1 - np.sum(self.occupancy);
        d_occ = free*self.k_on*Cvec - self.k_off*self.occupancy
#        print(type(d_occ), d_occ.size)
#        print(type(self.occupancy), self.occupancy.size)
        self.occupancy += dt*d_occ;
        
    def activity(self):
        """
        This method calculates the activity of the receptor given its occupancy and the efficacy of ligands.
        
        :return: Activity of the receptor.
        :rtype: float
        """
        
        return np.dot(self.efficacy, self.occupancy)
    
    def __str__(self):
        pristr = 'This is a receptor object.\n On-rate: ' + str(self.k_on) + ' nM^-1 s^-1' + '\n'\
        'Offrate: ' + str(self.k_off) + ' s^-1' + '\n' \
        'Current occupancy:' + str(self.occupancy) + '\n' \
        'Current activity:' + str(self.activity()) + '\n' \
        
        
        return pristr


class TerminalFeedback(receptor):
    """ 
    DA terminal Feedback loop, special case of :class:`receptor`. Uses receptor activation to regulate multiplicative gain of DA release from terminals,
    is evoked by the :class:`DA`-class to control DA release. 
    The feedback is similar to `Dreyer et al, J Neurosci, 2016 <http://www.jneurosci.org/content/36/1/98.long>`_ and is regulated by parameter alpfa
    
    :param alpha: Gain paramter. *alpha* = 0 means no feedback. The larger the value, the stronger the feedback.
    :type alpha: float >= 0
    
    .. todo::
        -move defaults here, instead of :class:`DA`-class
    """
    def __init__(self, alpha, k_on, k_off, occupancy = 0.5, efficacy = 1):
        receptor.__init__(self, k_on, k_off, occupancy, efficacy)
        self.alpha = alpha;
    
    def gain(self):
        """
        :return: Gain is 1/(1 + *alpha*x*activity*) and must be multiplied to release probability. 
        :rtype: float
        """
        return 1/(1 + self.alpha*self.activity())    
      


class SomaFeedback(receptor):
    """ 
    DA somatodendritic Feedback loop, special case of :class:`receptor`. Uses receptor activation to regulate additive gain of DA firing rate,
    is evoked by the :class:`DA`-class.
    
    The feedback is similar to `Dreyer and Hounsgaard, J Neurophys, 2013 <https://www.physiology.org/doi/10.1152/jn.00502.2012>`_ and is regulated by parameter alpfa
    
    :param alpha: Gain paramter. *alpha* = 0 means no feedback. The larger the value, the stronger the feedback.
    :type alpha: float >= 0
    
    .. todo::
        -move defaults here, instead of :class:`DA`-class
    """
    def __init__(self, alpha, k_on, k_off, occupancy = 0., efficacy = 1):
        receptor.__init__(self, k_on, k_off, occupancy, efficacy)
        self.alpha = alpha;
    
    def gain(self):
        """
        :return: Gain is *alpha*x*activity* and must be subtracted from the firing rate. To be on the safe side it is truncated to be >= 0
        :rtype: float
        """
        return np.maximum(0, self.alpha*self.activity());
    