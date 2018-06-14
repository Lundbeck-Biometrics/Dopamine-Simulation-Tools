# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 09:32:34 2018

@author: jakd@lundbeck.com 

This is collection of classes and functions that are used to create the framework for simulations of dopamine signaling and adaptations in post synaptic interpretation of dopamine signals. 
Most important classes and functions are 

   - :class:`DA` which represents a full dopamine system with somatodendritic and terminal DA release. 
   - :class:`PostSynapticNeuron` which can be configured to represent the post synaptic readout of DA. 
   - :func:`AnalyzeSpikesFromFile` which is used to analyze data from experimental recordings. 
   
.. todo::
   - Create a simulation-class that takes firing rate vector as input. And that collects DA, D1, D2 systems at once.
   - Implement exact solutions of all linear systems. This system should work if we know DA conc as a timeseries? 
   - Can we rephrase post synaptic adaptations as number D2R's and A2aR' instead of Gain & Threshold?
 
**Copyright (C) 2018  Jakob Kisbye Dreyer, Department of Bioinformatics, H Lundbeck A/S.** 

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

===========================================================================
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

class PostSynapticNeuron:
    """
    This is the base class going to represent D1-MSN's and D2-MSN's. It will respond to external lignads such as dopamine
    and generate cAMP. This class uses the :class:`receptor`-class and add further attributes like gain and threshold for activating cascades.
    
    If the class is evoked with instances of :class:`Drug`-class receptors will be prepared for more ligands.
    
    :param neurontype: Write the particular type of medium spiny neuron. 'D2' or 'D1'. Not case sensitive. 
    :type neurontype: str
    :param k_on: On-rate of ligands
    :type k_on: float
    :param k_off: Off-rate of ligands
    :type k_off: float
    :param Gain: Gain parameter that links activation of receptors to rate of cAMP generation. Default Gain = 10. 
    :type Gain: float
    :param Threshold:  represents the threshold fraction of receptors that needs to be activated for initiating AC5. Default threshold = 0.05. Note the threshold is an *upper* threshold in :class:`D2MSN` and a *lower* threshold in :class:`D1MSN`  
    :type Threshold: float
    :param kPDE: Decay constant of phosphodiestase in s\ :sup:`-1`. Default is *kPDE* = 0.1\ :sup:`-1` (see `Yapo et al, J Neurophys, 2017 <http://dx.doi.org/10.1113/JP274475>`_  )
    :type kPDE: float
    :param efficacy: array of efficacies of the ligands. Default is 1, for dopamine. 
    :type efficacy: numpy array
    :param drugs: instance of :class:`Drug`-class. If none are given it is assumed that dopamine is the only ligand. More than one drug can be in the list.
    :type drugs: :class:`Drug` - object
    
    Example::
        
        >>> d1 = PostSynapticNeuron('d1');
        >>> print(d1)
        This is a D1-MSN. AC5 is *activated* by DA.

        Receptor on-rate = [0.01] nM^-1 s^-1 
        Receptor off-rate= [10.] s^-1
        Receptor EC50 = [1000.] nM 
        Current input mapping: 
           D1  Bmax  = 18
           M4 Bmax   = 10
        Current cAMP level: 
           cAMP     = 0.198                

    """
    #: Low limit for cAMP that initiates change in Bmax'es 
    cAMPlow = 0.1; 
    #: High limit for cAMP that initiates change in Bmax'es.
    cAMPhigh = 10; 
    
    #: *cAMPoffset* sets the bias in updating. This parameter determines how fast to update Bmax if everythin seems fine. Must be low, for example 0.1 or 0.01 
    cAMPoffset = 0.1;

    
    kPDE = 0.1
    
    k_on = 1e-2;
    k_off = 10;
    
    
    #Initial value of cAMP is 2*lower limit. 
    cAMP = 2*cAMPlow;

    def __init__(self, neurontype,  *drugs):
        k_on = np.array([self.k_on])
        k_off = np.array([self.k_off])
        efficacy = np.array([1])

    
        tempoccupancy = np.array([0]);
        
        for drug in drugs:
            print('\n\nAdding DA acting drug to post synaptic neuron:')
            print('  DA-competing drug: ' + drug.name)
            
            print('  on-rate: ', drug.k_on)
            k_on  = np.concatenate( ( k_on , [drug.k_on] )  )
           
            print('  off-rate: ',  drug.k_off)
            k_off = np.concatenate( ( k_off , [drug.k_off] )  )
  
            print('  inital occupancy: ',  0)
            tempoccupancy = np.concatenate( ( tempoccupancy, [0])  )
            print('  efficacy: ', drug.efficacy)
             
            efficacy = np.concatenate( (efficacy, [drug.efficacy]) )
            
        self.DA_receptor = receptor(k_on, k_off, tempoccupancy, efficacy)
        self.DA_receptor.ec50 = k_off/k_on
        
        
        #: The 'other receptor' represents the competing receptor. If D1MSN it is M4R and in D2-MSN it is A2A.
        self.Other_receptor = receptor(self.k_on, self.k_off, tempoccupancy[0])#Use default efficacy = 1;
        self.Other_receptor.ec50 = self.k_off/self.k_on
        OtherligandC = 100;
        self.Other_receptor.occupancy = np.array([OtherligandC /(self.Other_receptor.ec50 + OtherligandC)])
         
        if neurontype.lower() == 'd1' :
            print('Setting type = D1-MSN. DA *activates* AC5.')
            self.type = 'D1-MSN'
            self.ac5sign = 1;
            
            self.DA_receptor.bmax = 18
            self.Other_receptor.bmax = 10
    
         
        elif neurontype.lower() == 'd2':
            print('Setting type = D2-MSN. DA *inhibits* AC5.')
            self.type = 'D2-MSN'
            self.ac5sign = -1
            
            self.DA_receptor.bmax = 50
            self.Other_receptor.bmax = 25
  
        else:
            print('Unknow neuron type. No interaction link wtih AC5')
            self.type = neurontype + '(unknown)'
            self.ac5sign = 0;

    def AC5(self):
        """
        Method that calculates AC5 activity based on current values of receptor activity, gain and threshold. 
        This method uses the ac5-sign attribute to get the sign right for linking receptor activity and AC5.
        
        
        :return: AC5 activity that can be used to update cAMP.
        :rtype: float
        """
        act_da    = self.DA_receptor.activity()*self.DA_receptor.bmax
        act_other = self.Other_receptor.activity()*self.Other_receptor.bmax
        return np.maximum(self.ac5sign*(act_da - act_other), 0)    
  

    def updateCAMP(self, dt):
        """Function that increments the value of cAMP. Uses the current occupancy and calls the :func:`AC5`-method
        
        :param dt: Timestep
        :type dt: float
        """
        self.cAMP += dt*(self.AC5() - self.kPDE*self.cAMP)
        
    def updateNeuron(self, dt, C_DA_ligands, C_other_ligand = 100):
        """
        This is a method that  opdates occupancy *and* cAMP in one go. 
        
        :param dt: Timestep
        :type dt: float
        :param C_DA_ligands: Concentration of DA receptor binding ligands at time *t*. 
            If the Postsynaptic neuron is initialized with drugs then this shold be a list of concentrations.
        :type C_DA_ligands: float or array
        :param C_other_ligand: Concentration of opponent ligand. If D1 neuron this is acetylcholine, if D2 neuron this is adenosine. 
        :type C_other_ligand: float
        """
        self.DA_receptor.updateOccpuancy(dt, C_DA_ligands)
        self.Other_receptor.updateOccpuancy(dt, C_other_ligand)
        self.updateCAMP(dt)
        
  
    def updateBmax(self, dt, cAMP_vector):
        """
        This method updates gain and threshold in a biologically realistic fashion. G&T is incremented based on current *caMP* and *cAMPlow* and *cAMPhigh*. 
        Use a vector of cAMP values to batchupdate.
        
        :param dt: time step in update vector
        :type dt: float
        :param cAMP_vector: vector of recorded caMP values. 
        :type cAMP_vector: numpy array
        
        .. Note:: This is a very slow method and is mainly used to illustrate which adaptatios are faster than others and to investigate non-adapted systems. Use the :func:`Fast_updateG_and_T`-method if you just want to know the end-stage of the adaptaions.
        
        .. seealso:: :func:`Fast_updateG_and_T`
        """
        
        #First get the lower boundary. Everytime cAMP was *below* cAMPlow we reduce/increase opponent bamx in D1/D2 neurons 
        '              This term is 1 if cAMP is lower than limit         This term is around 0.1'
        LowLimErr  =   np.heaviside(self.cAMPlow - cAMP_vector, 0.5)     -   self.cAMPoffset;       
        'If camp is everywhere above camplow we have a small negative LowlimERR. If everythwere below camplow we have a large positive term'
 
        '              This term is 1 if cAMP is higher than high-limit         This term is around 0.1'
        HighLimErr =   np.heaviside(cAMP_vector - self.cAMPhigh, 0.5)             - self.cAMPoffset; 
        'If camp is everywhere above camplow we have a small negative LowlimERR. If everythwere below camplow we have a large positive term'

        "Receptors are regulated differently in D1 and D2 msns:"
        if self.type == 'D1-MSN':
            self.DA_receptor.bmax    -= dt*np.sum(HighLimErr)
            self.Other_receptor.bmax -= dt*np.sum(LowLimErr)
            
        elif self.type == 'D2-MSN':
            self.DA_receptor.bmax    -= dt*np.sum(LowLimErr)
            self.Other_receptor.bmax -= dt*np.sum(HighLimErr)
        else:
            print('no valid neuron')
            return
          
        
        
    def Fast_updateG_and_T(self, cAMP_vector, Gain_guess = 0, Thold_guess = 0):
        """
        Here we provide a fast method to increment values of Gain and Threshold variables. The method uses a good guess 
        of what the next value should be rather than simply incrementing. The guess is based on assuming a linear transformation between input, Gain and threshold variables and cAMP. So we try to use Gain and Threshold to 'rescale* the cAMP axis. 
        Nature will not do it this way, but this method will converge faster. 
        
        
        
        :param cAMP_vector: vector of recorded caMP values
        :type cAMP_vector: numpy array
        :param Gain_guess: Initial value of gain. If *Gain_guess* == 0, the current *Gain* will be used. 
        :type Gain_guess: float
        :param Thold_guess: Initial value of Threshold. If *Thold_guess* == 0, the current *Threshold* will be used. 
        :type Thold_guess: float
        
        .. Warning:: **Does not work and updates nothing** 

        .. Warning:: Initial guess of *threshold* must be within the range of receptor occupancies visited. Otherwise we get *cAMP* = NaN. 
        
        .. seealso:: :func:`updateBmax`
        
        """
        
        
        if Gain_guess == 0:
            Gain_guess = self.DA_receptor.bmax;
                
        if Thold_guess == 0:
            Thold_guess = self.Other_receptor.bmax/self.DA_receptor.bmax*self.Other_receptor.activity();
            
        Flow = np.percentile(cAMP_vector, self.cAMPoffset*100)
        Fhigh = np.percentile(cAMP_vector, 100*(1 - self.cAMPoffset))#Note thold-offset is normally negative
        
        print('DAbmax:', self.DA_receptor.bmax)
        print('O-bmax:', self.Other_receptor.bmax)
        print('G:', Gain_guess)
        print('T:', Thold_guess)
  
        print('Flow:', Flow)
        print('Fhigh:', Fhigh)
  
    
        An = (self.cAMPhigh - self.cAMPlow)/(Fhigh - Flow)
        print('An:', An)
        Bn = (self.cAMPlow - An*Flow)/(Gain_guess*An)
        print('Bn:', Bn)

        NewThreshold =  Thold_guess - (self.ac5sign)*Bn
        NewGain = An*Gain_guess
        
        print('nG:', NewGain)
        print('nT:', NewThreshold)
        
        DA_receptor_bmax = NewGain;
        Other_receptor_bmax = self.DA_receptor.bmax*NewThreshold/self.Other_receptor.activity();
        
        print('new DAbmax:', DA_receptor_bmax)
        print('new O-bmax:', Other_receptor_bmax)
        
        print('\n*********************************************************\n'\
              + '** This function is not implemented with bmax paradigm **\n'\
              + '*********************************************************\n')

    def __str__(self):
        
        
        retstr = \
         '\n This is a ' + self.type + '. AC5 is ' + (self.ac5sign == 1)*'*activated*' + (self.ac5sign == -1)*'*inhibited*' + ' by DA and '\
         'AC5 is '+ (self.ac5sign == -1)*'*activated*' + (self.ac5sign == 1)*'*inhibited*' + ' by '\
         + (self.ac5sign == 1)*'M4R' + (self.ac5sign == -1)*'*A2AR*' + '.\n\n'\
        'DA Receptor on-rate = ' + str(self.DA_receptor.k_on) + ' nM^-1 s^-1 \n'\
        'DA Receptor off-rate= ' + str(self.DA_receptor.k_off) + ' s^-1\n'\
        'DA Receptor EC50 = ' + str(self.DA_receptor.k_off/self.DA_receptor.k_on) + ' nM \n'\
        'Current input mapping: \n' \
        '  ' + self.type[0:2] + '  Bmax  = ' + str(self.DA_receptor.bmax) + '\n'\
        '  ' + (self.ac5sign == 1)*'M4' + (self.ac5sign == -1)*'A2A' + ' Bmax  = ' + str(self.Other_receptor.bmax) + '\n'\
        'Current cAMP level: \n'\
        '  cAMP     = ' + str(self.cAMP) 
        
        return retstr

    
    

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
    
 
class DA:
    """
    This is a dopamine class. Sets up a set of eqns that represents dopamine levels in midbrain and Nucleus accumbens / Dorsal striatum. Parameters depend on area. 
    The somatodendritic compartment uses the same parameters. 
    
    :param area: The location of cell bodies can be either 'vta' for mesolimbic projections or 'SNc' for nigrostriatal projections. Default is 'vta'. Not case sensitive. 
    :type area: str
    :param drug: Optional Drug. Include only drugs that interact with the D2-receptor.
    :type drug: :class:`Drug`-object. 
    
    **Examples:**
   
    Simple run::
    
        >>> from DopamineToolbox import DA
        >>> #create mesolimbic dopamine system instance
        >>> mesolimb = DA('VTA')
        >>> #Update 3 times with timestep 0.01.(Use default input firing rate 5 Hz)
        >>> for k in range(3):
            ...     mesolimb.update(0.01)
            ...     print(mesolimb.Conc_DA_term)
            
        48.02961272573888
        49.374838967187856
        49.045908794874855
            
    Running with drug (see notes below)::
        
        >>> from DopamineToolbox import DA, Drug
        >>> #create default drug instance 
        >>> mydrug = Drug()
        Creating Drug-receptor class. More receptor interactions can be added manually! 
        Use <name>.<target> = DrugReceptorInteraction('name.tagret', target, kon, koff, efficacy)
        >>> #create nigrostriatal DA system with drug:
        >>> nigro = DA('SNC', mydrug)
        Adding to Dopamine system:
        D2-competing drug: Default Agonist
        on-rate:  0.01
        off-rate:  1.0
        inital occupancy:  0
        efficacy:  1.0
        >>> #Update with mydrug concentration = 1000 and print presynaptic D2 receptor occupany and activity
        >>> for k in range(5):
            ...    nigro.update(0.01, Conc = np.array([0, 1000])) 
            ...    print('Occ:', nigro.D2term.occupancy, '\t Act:', nigro.D2term.activity())

        Occ: [0.49925    0.05   ]        Act: 0.54925
        Occ: [0.49836076 0.094575  ]     Act: 0.5929357625
        Occ: [0.49735416 0.13433567]     Act: 0.6316898310475001
        Occ: [0.49621567 0.16982333]     Act: 0.6660390064955519
        Occ: [0.4949775  0.2015212]      Act: 0.6964986962342842
  
    
    .. Note::
           
        - In the above example *Conc* is a two element array. 2nd element is drug concentraion. First element is a placeholder replaced with current DA concentraion by the DA.update-method
        - If all receptor occupancies of the DA-cllass will be vectors if the instance is created with drug argument. Note that occupancy is two element: First is DA occupany, second is drug occupancy. *activity* is scalar. 
         
        
    """
    
    "Class attributes:"
    "*****************"
    
    Km = 160.0; 
    """ MicMen reuptake parameter. Affected by cocaine or methylphenidate. If running multiple systems, they can all be updated using ``DA.Km = xxx`` """
    
    k_nonDAT = 0.04; 
    """First order reuptake constant from `Budygin et al, J. Neurosci, 2002 <http://www.jneurosci.org/content/22/10/RC222>`_.
    Represents other sources of dopamine removal that are not mediated by DAT's. For example MAO's and NET-mediated uptake. """
    
    Precurser = 1.0; 
    """This is a parameter that is used to represent the DA release capacity. If *precurser* = 1, then DA release is normal. Use ``myDA.precurser = 3.0`` to simulate effect of L-dopa"""
    
    NNeurons = 100; 
    """The number of neurons in the intact system. Change this *after* creation of the class if simulating PD::
        
        >>> myDAsys = DA('SNC')
        >>> #Simulate 90% cell loss:
        >>> myDAsys.NNeurons = 10
        
    Loss of *V*\ :sub:`max` is automatically included.
    """
    def __init__(self, area = "VTA", *drugs):
        k_on_term = np.array([0.3e-2])
        k_off_term = np.array([0.3])
        k_on_soma = np.array([1e-2])
        k_off_soma = np.array([10.0])
        efficacy = np.array([1]);
        D2occupancyTerm = np.array([0.5])
        D2occupancySoma = np.array([0.])
        
        for drug in drugs:
            print('\n\nAdding to Dopamine system:')
            print('  D2-competing drug: ' + drug.name)
            print('  on-rate: ', drug.k_on)
            k_on_term = np.concatenate( ( k_on_term , [drug.k_on] ))
            k_on_soma = np.concatenate( ( k_on_soma , [drug.k_on] ))
            print('  off-rate: ',  drug.k_off)
            k_off_term = np.concatenate( ( k_off_term , [drug.k_off] ))
            k_off_soma = np.concatenate( ( k_off_soma , [drug.k_off] ))
            print('  inital occupancy: ',  0)
            D2occupancyTerm = np.concatenate( ( D2occupancyTerm, [0]))
            D2occupancySoma = np.concatenate( (D2occupancySoma, [0]))
            print('  efficacy: ', drug.efficacy)
            efficacy = np.concatenate( (efficacy, [drug.efficacy]))
        self._drugs = drugs   
        self.D2term = TerminalFeedback(3.0, k_on_term, k_off_term, D2occupancyTerm, efficacy)
        self.D2soma = SomaFeedback(10.0, k_on_soma, k_off_soma, D2occupancySoma, efficacy)
       
        Original_NNeurons = 100;
        
        
       
        self.nu = 0; 
        """This is an instance-attribute that reports the *actual* firing rate (*NU* minus sd feedback). It is only used to monitoring the system. 
        For example::
            
            >>> #update myDA 100 times with input firing rate 5 Hz
            >>> [myDA.update(dt, nu_in = 5) for k in range(100)]
            >>> #Check that the real firing rate is lower than nu_in
            >>> print('Actual firing rate:', myDA.nu, 'Hz')
            
            Actual firing rate: 4.1948 Hz
        
        
        """
        
        self.area = area
        if area.lower() == "vta":
            self.Vmax_pr_neuron = 1500.0/Original_NNeurons
            self.Gamma_pr_neuron = 200.0/Original_NNeurons #This value is deliberately high and anticipates a ~50% reduction by terminal feedback
        elif area.lower() == "snc":
            self.Vmax_pr_neuron = 4000.0/Original_NNeurons
            self.Gamma_pr_neuron = 400.0/Original_NNeurons; #This value is deliberately high and anticipates a ~50% reduction by terminal feedback
        else:
            print('You are in unknown territory.\n Vmax_pr_neuron and Gamma_pr_neuron are not set.')
            self.Vmax_pr_neuron = 0
            self.Gamma_pr_neuron = 0; 
        self.Vmax_pr_neuron_soma = 200.0/Original_NNeurons;
        self.Gamma_pr_neuron_soma = 20.0/Original_NNeurons;
        self.Conc_DA_soma = 50.0;
        self.Conc_DA_term = 50.0;
        
   
  
    
    def update(self, dt, nu_in = 5, e_stim = False, Conc = np.array([0.0])):
        """
        This is the update function that increments DA concentrations in somatodendritic and terminal compartments and also updates autoreceptors. 
        
        :param dt: time step in forward Euler integration. *dt* = 0.01 usually works
        :type dt: float
        :param nu_in: Input firing rate used in the update in Hz. Default is 5 Hz. This is *before* somadodendritic autoinhibition. When updated many times with default the firingrate will be around 4 Hz. 
        :type nu_in: float
        :param e_stim: Determines if the stimulus is part of an evoked stimulus train. Evoked stimuli do not respond to somatodendritic inhibition. So if *e_stim* == True, then *nu* = *nu_in*. Also used when analyzing experimental firing rates (because they already have had autoinhibition)
        :type e_stim: bool
        :param Conc: Concentration of ligands. **Un-necessary unless simulating drugs**. First element is a placeholder, always overwritten by current concentration of dopamine in the relevant compartment. When simulating presence of *N* drugs, *Conc* must be length *N+1*
        :type Conc: array
        
        
        """ 
        Conc[0] = self.Conc_DA_soma
#        print(Conc)
        self.D2soma.updateOccpuancy(dt, Conc)
        Conc[0] = self.Conc_DA_term
        self.D2term.updateOccpuancy(dt, Conc)
        
        "If e_stim -> True, then the firing rate overrules s.d. inhibition"
        self.nu = np.maximum(nu_in - self.D2soma.gain()*(1 - e_stim), 0);

        rel = np.random.poisson(self.NNeurons*self.nu*dt);
        "first calculate somato dendritic DA:"
        self.Conc_DA_soma += self.Precurser*rel*self.Gamma_pr_neuron_soma - dt*self.Vmax_pr_neuron_soma*self.NNeurons*self.Conc_DA_soma/(self.Km + self.Conc_DA_soma) - dt*self.k_nonDAT;
        "...then terminal DA"
        self.Conc_DA_term += self.Precurser*rel*self.Gamma_pr_neuron*self.D2term.gain() - dt*self.Vmax_pr_neuron*self.NNeurons*self.Conc_DA_term/(self.Km + self.Conc_DA_term)  - dt*self.k_nonDAT;

    def AnalyticalSteadyState(self, mNU = 4):
        """
        This is a function that calculates analytical values of steady-state DA concentrations and standard deviation of baseline based on the system parameters. 
        Includes the effect of terminal autoreceptors and assumes a constant firing. Uses linear approximation of DA release and uptake to get standard deviation. 
        
        :param mNU: Mean firing rate in Hz. **Assumes no somatodendritic autoinhibition on cell firing**. So *mNU* = 4 Hz corresponds to the steady state of *nu_in* = 5 Hz. 
        :type mNU: float
        :return: Mean of baseline and approximate standard deviation of baseline. 
        :rtype: float. Tuple with (mean, std) if called with one output variable, or call with ``m,s = myDA.AnalyticalSteadyState()`` to get separate values.     
        """
        g = self.Gamma_pr_neuron; 
        km = self.Km; 
        V = self.Vmax_pr_neuron;
        beta = self.D2term.alpha;
        e50 = self.D2term.k_off[0] / self.D2term.k_on[0]
        
        "Solution is root of a quadratic function. First we get poly-coeffs:" 
        A = mNU*g - V- V*beta
        B = mNU*g*(e50 + km) - e50*V
        C = mNU*g*km*e50;
        "Then the discrininant is found:"
        D = B**2 - 4*A*C
        
        "Only the positivie root makes sense:" 
        mDA = (-B - np.sqrt(D))/(2*A)
        
        "Now calculate the std dev:"
        mD2 = 1/(mDA + e50);
        g1 = g/(1+beta*mD2);
        sDA = g1*np.sqrt(mNU*km/(2*V))

        return mDA, sDA
    
    def CreatePhasicFiringRate(self, dt,  Tmax,  Nuaverage = 5,  Nuburst = 20, Tperiod = 1,  Tpre = 0):
        """
        This method creates a NU-time series with a repetivite pattern of bursts and pauses - Grace-bunney-style. The NU time series can be given as input to :func:`update`.
        
        :param dt: time step in output array
        :type dt: float
        :param Tmax: Total time length of output array in seconds, including any tonic presimulation. Actual lengt is floor integer. 
        :type Tmax: float. 
        :param Nuaverage: Average firing rate in Hz in phasic and tonic blocks. 
        :type Nuaverage: float 
        :param Nuburst: Firing rate in Hz in burst epoch in phasic block
        :type Nuburst: float
        :param Tperiod: Duration of a full burst-pause cycles
        :type Tperiod: float
        :param Tpre: Length of constant firing rate pre-simulation in seconds. Defualt 0. 
        :type Tpre: float
        :return: numpy array of firingrates. 
        
        .. todo:: 
            - Implement more types of firing patterns. 
            - Use markov chains to get burst and pauses. 
            - Create more random-like firing pattens
         
        """
       
        "## First Generate single burstpause pattern:"
        "Number of spikes in a signle burst:"
        spburst = Nuaverage*Tperiod;
        "Number of timesteps for a single burst:"
        burstN = int(np.ceil(spburst/Nuburst/dt));
        "Number of timesteps in single period. "
        Npat   = int(np.ceil(Tperiod/dt));
        pattern = np.zeros(Npat);
        "First part is burst, the rest remains 0 - its a pause :o)"
        pattern[0:burstN] = Nuburst;
        
        
        "Expand until many burst-pause patterns"
        dtph = Tmax - Tpre; #total duration of phasic pattern.
        Nph = np.ceil(dtph/dt);
        Nto = int(np.ceil(Tpre/dt));
        #Now the pattern gets expanded
        NUphasic = np.tile(pattern, int(np.ceil(Nph/Npat)));
        #The tonic segment is X times as long as the phasic
        NUtonic = Nuaverage*np.ones(Nto);
        #This is one segment containing a tonic and a phasic segment
        NUstep = np.hstack( (NUtonic, NUphasic) );
        #Now this is expanded into a repetivive pattern. 
        
        #Trunctate if actual number of elements exceeds Tmax  
        realN = int(np.floor(Tmax/dt))
        
        return NUstep[0:realN]
    

    
    def __str__(self):
        mda, sda = self.AnalyticalSteadyState()
        drugstr = '';
        
        for drug in self._drugs:
            drugstr += '\n' + 'Includes: \n ' + drug.__str__()
        
        retstr = \
        '\n' + 'Dopamine system including terminal and somatodendritic feedbacks. Cell body located at ' + self.area + '\n'\
        + drugstr + '\n' \
        'DA Settings: \n' \
        '   Vmax = ' + str(self.Vmax_pr_neuron * self.NNeurons) + ' nM/s \n'\
        '   Km = ' + str(self.Km) + ' nM \n'\
        '   Neurons = ' + str(self.NNeurons) + ' \n'\
        'Current DA concentrations: \n' \
        '   Terminal = ' + str(self.Conc_DA_term) + ' nM \n'\
        '   somatodenritic = ' + str(self.Conc_DA_soma) + ' nM \n'\
        'Analytic 4 Hz steady state: \n' \
        '   Terminal mean DA: ' + str(mda) + ' nM \n'\
        '   Terminal std DA: '  + str(sda) + ' nM \n'\
        'Current feedbacks: \n' \
        '   Terminal = ' + str(self.D2term.gain()) + ' \n'\
        '   Somatodenritic = ' + str(self.D2soma.gain()) + ' Hz \n'\
              
        return retstr

class Cholinergic:
    """
    This is an object that calculates acetyl choline concentrations in striatum.  Based on DA concentration inhibition of TAN firing rate.  
    
    Characteristics are taken from <https://www.sciencedirect.com/science/article/pii/S0014488613001118?via%3Dihub> 
    and https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3864134/
    
    """
    def __init__(self, k_AChE = 1.2, gamma = 24, *drugs):
        print('Creating TAN-interneuron release and AChE decay')
        self.k_AChE = k_AChE;
        self.NNeurons = 100;
        self.gamma1 = gamma/self.NNeurons;
        self.nu = 5; "Initial firing rate. Will be updated and report the actual firing rate of the TAN's"
        
        self.Conc_ACh = gamma*self.nu/k_AChE;
        
        k_on = np.array([1e-2])
        k_off = np.array([10.0])
        efficacy = np.array([1])
        D2occupancy = np.array([0.05])
        
        
        for drug in drugs:
            print('Adding D2active drugs to the TAN')
            print('  D2-competing drug: ' + drug.name)
            print('  on-rate: ', drug.k_on)
            k_on = np.concatenate( ( k_on , [drug.k_on] ))
           
            print('  off-rate: ',  drug.k_off)
            k_off = np.concatenate( ( k_off , [drug.k_off] ))
           
            print('  inital occupancy: ',  0)
            D2occupancy = np.concatenate( ( D2occupancy, [0]))
            
            print('  efficacy: ', drug.efficacy)
            efficacy = np.concatenate( (efficacy, [drug.efficacy]))
        self.D2soma = SomaFeedback(30.0, k_on, k_off, D2occupancy, efficacy)
        
    def update(self, dt,  Conc, nu_in = 6):
        """
        Here we update ACh concentrations. We use the Mic-Men data provided when initializing the neuron.
        
        :param dt: Time step in seconds
        :type dt: float
        :param nu_in: Input firing rate of the TAN in Hz. Default is 6 Hz, which corresponds to the *uninhibited* firing rate. (Prosperitti et al, Exp Neurology, 2013)
        :param Conc: Input concentration of D2 binding ligands. Conc[0] is DA concentraion in nM
        :param Conc: numpy array
        """
        
        "Update the Chol's D2 receptors:"
        self.D2soma.updateOccpuancy(dt, Conc) 
        self.nu = np.maximum(nu_in - self.D2soma.gain(), 0)
        R = np.random.poisson(self.NNeurons*self.nu*dt)
        self.Conc_ACh += np.maximum(self.gamma1*R - dt*self.k_AChE*self.Conc_ACh, -self.Conc_ACh)
        

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
       >>mydrug = Drug('secret DA agonist', 'DA-D2', efficacy = 1)
        
    """
    
    def __init__(self, name = 'Default Agonist', target = 'D2R', k_on = 0.01, k_off = 1.0, efficacy = 1.0):
        DrugReceptorInteraction.__init__(self, name, target, k_on, k_off, efficacy)
        print("Creating Drug-receptor class. More receptor interactions can be added manually! \nUse <name>.<target> = DrugReceptorInteraction(\'name.tagret\', target, kon, koff, efficacy)")
    
    def Concentration(self, t,  dose, t_infusion = 0, k12 = 0.3/60, k21 = 0.0033, k_elimination =0.0078):
        """
        Calculates drug concentration in brain using two-compartment PK. Default values are for cocaine PK as estimated 
        in `Pan, Menacherry, Justice; J Neurochem, 1991 <https://doi.org/10.1111/j.1471-4159.1991.tb11425.x>`_. But here the variaables are transformed from minues to seconds. 
        
        :param t: time to calculate dose. Can be scalar or array
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
        
        .. figure:: Pan_1991_Fig1.jpg
           :scale: 50 %
           :alt: Pan et al, 1991, Figure 1. 
           :align: center

           Figure 1 from `Pan et al, J Neurochem, 1991 <https://doi.org/10.1111/j.1471-4159.1991.tb11425.x>`_ showing compartments. The first compartment 'BODY CAVITY' is not used. 
        
        .. Note:: If *t* < *t_infusion* the concentraion is 0. If *t* is an array of times, the outpur concentraion for *t* < *t_infusion* are 0. 
        """
        sumk = k12+k21+k_elimination
        D = np.sqrt(sumk**2 - 4*k21*k_elimination);
        a = 0.5*(sumk + D);
        b = 0.5*(sumk - D);
        C= dose*k12/(a-b)*( np.exp(-b*(t-t_infusion) ) - np.exp(-a*(t-t_infusion)) );
        tdrug = t >= t_infusion;
        c2 = C*tdrug;

        return c2
 
        
def AnalyzeSpikesFromFile(FN, dt = 0.01, area = 'vta', synch = 'auto', pre_run = 0, tmax = 0, process = True, adjust_t = False):
    """
    This is a function that uses :class:`DA`, :class:`D1MSN` and :class:`D2MSN`-classes to analyze spikes from experimental recordings. 
    It is based on similar methods as used in `Dodson et al, PNAS, 2016 <https://doi.org/10.1073/pnas.1515941113>`_.
    It also includes the option to make 'clever' choice of synchrony, described below. 
    
    :param FN: *Filename* including full path to experimental data file. 
    :type FN: string
    :param dt: Timestep in simulation (dt = 0.01s by default)
    :type dt: float
    :param area: Area in which to set up the dopamine simualtion. Choose between 'VTA' or 'SNc'. Default is 'VTA'
    :type area: string
    :param synch: FWHM in s for other neurons in ensemble. If set to 'auto', the synch is decided based on average firing rate of the input cell. Default is 'auto'
    :type synch: float
    :param pre_run: Number of seconds to run a totally tonic simulation before recorded firing pattern kicks in. Default is 0.
    :type pre_run: float
    :param tmax: Length of simulation in seconds. If *tmax* == 0, tmax will be equal to the last spike in the recording. 
    :type tmax: float
    :param process: Indicate if the function should process the timestamps or not. If *True* the method will also include a DA simulation. If *False* the output will only contain the timestams and firing rates of the file. Default is *True*
    :type process: bool
    :param adjust_t: adjust off of time series?  Default is *False*
    :type adjust_t: bool
    :return: Result from a DA simulation using *file* as input. 
    :rtype: :class:`Res`-object. The attributes of the output depends on the *process* parameter. 
    
    .. note:: 
        Total length of the file is tmax + pre_run.    
    
    The input must be formatted as timestamps in ascii text.  
    Current upported formats are Spike2 files with WAVMK timestamps (exported .txt files from preselected (spikesorted) channels in Spike2), or just a single column of timestapms (with or without single line of header).)    
    
    The synch = 'auto'-option creates a synch value equal to 0.3012 x *mean interspike interval*. For eaxmple if mean firing rate is 4 Hz, the synch will be 0.3012x0.25s = 0.0753s. 
    This option ensures that perfect pacemaker firing at any firing rate will be smoothed so that peak firing rate is 2*minimum firing rate. 
    
    .. todo::
        - This could be a method of the :class:`DA` -class? To allow user control of parameters. 
        - Better documentation of result class output. Perhaps move into main?
        - Include link to example python script that uses this function. 
        
    """
    from scipy.ndimage.filters import gaussian_filter1d as gsmooth
    
    #create empty array for the spikes:
    spikes = np.array([]); 
        
    print("½½½½½½½½½½½½½½½½½½½½½½½½½½½")
    print("Opening " + FN)

    with open(FN, 'rt') as fp: 
        
        line = fp.readline()
        while line:
            line = fp.readline()
            if line.find("WAVMK") > -1:
                #get first '\t'
                i1 = line.find('\t');
                #get next '\t':
                i2 = line.find('\t', i1+1)
                #The number we seek is between the two locations:
                N = float(line[i1:i2])
                spikes = np.append(spikes, N)
    if spikes.size == 0:
        print('No WAVMK in file...')
        print('try to open as a simple list of timestamps')
        try:
            spikes = np.loadtxt(FN)
        except ValueError:
            spikes = np.loadtxt(FN, skiprows = 1)
        
            
        
    print("Finished reading... " +'\n')     
    nspikes = spikes.size;
    DT = spikes[-1] - spikes[0]
    mNU = (nspikes - 1)/DT;
    print("Found " + str(nspikes) + " spikes")
    if nspikes > 0:  
        print("First spike at ", spikes[0], ' s')        
        print("Last spike at ", spikes[-1], ' s')    
        print("Mean firing rate: ", mNU, 'Hz')
        print('\n')
    else:
        print('NO spikes in file. Returning')
        return
    
    mISI = 1/mNU
    
    if synch == 'auto':
        print("Using automatic smoothing\n")

        synch = 0.3012*mISI; #smoothing in magic window
        W = synch/dt;
    else:
        W = synch/dt;
    
    
    
    if adjust_t:
        DTtrans = mISI - spikes[0]
        print("Adjusting start-gab by forward-translating" , DTtrans , ' s')
        spikes += DTtrans; 
    
    if tmax == 0:
        tmax = spikes[-1];
    
    binedge = np.arange(0, tmax, dt);
    tfile = binedge[:-1] + 0.5*dt
    sphist = np.histogram(spikes, binedge)[0]
    
    NUfile = gsmooth(sphist.astype(float), W)/dt;
    
    lastspike = spikes[-1] + mISI;
    if lastspike < tfile[-1]:
        print("Padding the end of firing rates with tonic firing...", tfile[-1] - lastspike , ' s\n')
    
    endindx = tfile > lastspike;
    NUfile[endindx] = mNU;
    
    NUpre  = mNU*np.ones(round(pre_run/dt))
    NUall = np.concatenate((NUpre, NUfile))
    tall = dt*np.arange(NUall.size) + 0.5*dt;
    
    "Get indices for switch betwen tonic and phasic for later use"
    iphasic_on = np.where(tall > pre_run)[0][0] #pre_run >= 0, and tall[0] = dt/2. So guaranteed to get at least one 
    
    iphasic_off = np.where(tall > pre_run + lastspike)[0]
    "if pre_run + lastspike > than tall[-1] we need to set the end-point manually:"
    if iphasic_off.size == 0:
        iphasic_off = tall.size - 1;
    else:
        iphasic_off = iphasic_off[0];
    
    print('File time on: ', tall[iphasic_on] - dt/2, ' s')    
    print('File time off: ', tall[iphasic_off] + dt/2, ' s')    
    
    "Setting up output from the function and populating attributes from the DA D1MSN and D2MSN classes:"
    
    class Res: 
        """
        This is an container for the results from the simulation
        """  
        def __init__(self, process):
            self.__process = process
        def __str__(self):
            "Note that we refer to future attributes being set below"
            
            if self.__process:
                class_str = "\n Results from running " + FN + ".\n\n"\
                "DA system parameters:\n" + \
                "   Area:" + self.da.area         
            else:
                class_str = "\n Firing rate information from " + FN + ".\n\n"
                
            return class_str
        
    Result = Res(process);   

    Result.File = FN;
    Result.InputFiringrate = NUall;
    Result.MeanInputFiringrate = mNU;
    Result.timeax = tall;
    Result.timestamps = spikes + pre_run;
    Result.synch = synch
   
    if process == False:
        print('Returning just firingrate and timestamps and exit')
        return Result
    Result.DAfromFile = np.zeros(NUall.size);        
    
    Result.da = DA(area);
    Result.d1 = PostSynapticNeuron('d1');
    Result.d2 = PostSynapticNeuron('d2');
    
    
    Result.d1.AC5fromFile = np.zeros(NUall.size)
    Result.d1.cAMPfromFile = np.zeros(NUall.size) 
    
    Result.d2.AC5fromFile = np.zeros(NUall.size)
    Result.d2.cAMPfromFile = np.zeros(NUall.size)
    
    print("Adjusting post synaptic thresholds and initial DA concentration to this file:")
    
    mda, sda = Result.da.AnalyticalSteadyState(mNU);
    mdar = mda/(mda + Result.d1.DA_receptor.ec50);
    
    
    Result.d1.Other_receptor.bmax = Result.d1.DA_receptor.bmax*mdar/Result.d1.Other_receptor.occupancy
    Result.d2.Other_receptor.bmax = Result.d2.DA_receptor.bmax*mdar/Result.d2.Other_receptor.occupancy

    
    "Setting inital value for DAconc:"
    Result.da.Conc_DA_term = mda;
    
    
    
    print("Analyzing the Firing rate")
    for k in range(NUall.size):
        Result.da.update(dt, NUall[k], e_stim = True );
        Result.d1.updateNeuron(dt, Result.da.Conc_DA_term)
        Result.d2.updateNeuron(dt, Result.da.Conc_DA_term)
        Result.DAfromFile[k] = Result.da.Conc_DA_term
        Result.d1.AC5fromFile[k]  = Result.d1.AC5()
        Result.d1.cAMPfromFile[k] = Result.d1.cAMP
        Result.d2.AC5fromFile[k]  = Result.d2.AC5()
        Result.d2.cAMPfromFile[k] = Result.d2.cAMP
    print('... done')
    
    Result.tonic_meanDA = mda;
    Result.meanDAfromFile  = np.mean(Result.DAfromFile[iphasic_on:iphasic_off])
    Result.d1.meanAC5fromFile = np.mean(Result.d1.AC5fromFile[iphasic_on:iphasic_off])
    Result.d2.meanAC5fromFile = np.mean(Result.d2.AC5fromFile[iphasic_on:iphasic_off])
    Result.d1.meancAMPfromFile = np.mean(Result.d1.cAMPfromFile[iphasic_on:iphasic_off])
    Result.d2.meancAMPfromFile = np.mean(Result.d2.cAMPfromFile[iphasic_on:iphasic_off])
    
    if iphasic_on > 0:
        Result.d1.tonic_meanAC5 = np.mean(Result.d1.AC5fromFile[:iphasic_on])
        Result.d2.tonic_meanAC5 = np.mean(Result.d2.AC5fromFile[:iphasic_on])
        Result.d1.tonic_meancAMP = np.mean(Result.d1.cAMPfromFile[:iphasic_on])
        Result.d2.tonic_meancAMP = np.mean(Result.d2.cAMPfromFile[:iphasic_on])
        
        Result.dAC5 = [Result.d1.meanAC5fromFile/Result.d1.tonic_meanAC5 , Result.d2.meanAC5fromFile/Result.d2.tonic_meanAC5 ]
    
    
    return Result
    
