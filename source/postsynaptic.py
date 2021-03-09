# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 15:41:14 2021

@author: JAKD
"""
import numpy as np
import receptors_and_feedbacks as raf


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
            
        self.DA_receptor = raf.receptor(k_on, k_off, tempoccupancy, efficacy)
        self.DA_receptor.ec50 = k_off/k_on
        
        
        #: The 'other receptor' represents the competing receptor. If D1MSN it is M4R and in D2-MSN it is A2A.
        self.Other_receptor = raf.receptor(self.k_on, self.k_off, tempoccupancy[0])#Use default efficacy = 1;
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
            print('Unknow neuron type. No interaction link with AC5')
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
        """Function that increments the value of cAMP. Uses the current occupancy and calls the :func:`AC5`-method.
        This method updates the cAMP attribute using the current state of AC5 activity. It is much slower than :func:`Get_the_cAMP`.
        
        :param dt: Timestep
        :type dt: float
        
        .. seealso:: 
            :func:`Get_the_cAMP`
            :func:`updateNeuron`
        """
        self.cAMP += dt*(self.AC5() - self.kPDE*self.cAMP)
    
    def Get_the_cAMP(self, timeax, DARact, OtherRact):
        """This method calculates a timeseries of cAMP-values based on a timeseries of receptor *activation*.
        
        
        :param timeax: Timevector. Must start at t = 0, and be equal spaced, and length more than 2. 
        :type timeax: numpy array
        :param DARact: Activation timeseries of DA receptor. Same length as *timeax*
        :type DARact: numpy array
        :param OtherRact: Activation timeseries of Other receptor. This is M4 receptor is object is D1MSN, A2A receptor if object is D2MSN. 
        :type OtherRact: numpy array
        
        This way is often much faster than single step updating. It requires that you collect a vector of receptor-activations both for DA-receptors and the *other* receptor. 
        
        """
        
        dt = timeax[1]-timeax[0];
        rx = self.DA_receptor.bmax*DARact - self.Other_receptor.bmax*OtherRact
        rx *= self.ac5sign 
        rx *= (rx >= 0)
        
        cAMP = np.exp(-self.kPDE*timeax)*np.cumsum(np.exp(self.kPDE*timeax)*rx*dt)
        
        return cAMP

    def updateNeuron(self, dt, C_DA_ligands, C_other_ligand = 100):
        """
        This is a method that opdates occupancy *and* cAMP in one go. 
        
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
        
  

    def updateBmax(self, Kbmax, cAMP_vector = None, dtsim = 1):
        """
        This method updates Bmax either D1 or D2 MSNs. Bmax is incremented based on current *caMP* compared to the limits *cAMPlow* and *cAMPhigh*. *cAMPoffset* also determines the final rate.  
        We use fast and slow decay methods for 
        Use a vector of cAMP values to batchupdate. For example using :func:`Get_the_cAMP`. 
        
        :param Kbmax: Rate of bmax-updates. Can be different for D1 and D2 MSN's. The real decay rate will also depend on self.cAMPoffset which is set by the neuron. 
        :type Kbmax: float
        :param cAMP_vector: vector of recorded caMP values. If omitted the current cAMP level is used
        :type cAMP_vector: numpy array
        :param dtsim: time step in the cAMPvector, that is the same as in the dopamine simulations. If omitted it will be 1
        :type dtsim: float
 
        
       
        
        """
        
        
        if cAMP_vector is None:
            cAMP_vector = self.cAMP
        #First get the lower boundary. Everytime cAMP was *below* cAMPlow we reduce/increase opponent bamx in D1/D2 neurons 
        '              This term is 1 if cAMP is lower than limit         This term is around 0.1'
        LowLimErr  =   np.heaviside(self.cAMPlow - cAMP_vector, 0.5)     -   self.cAMPoffset;       
        'If camp is everywhere above camplow we have a small negative LowlimERR. If everythwere below camplow we have a large positive term'
 
        '              This term is 1 if cAMP is higher than high-limit         This term is around 0.1'
        HighLimErr =   np.heaviside(cAMP_vector - self.cAMPhigh, 0.5)             - self.cAMPoffset; 
        'If camp is everywhere above camplow we have a small negative LowlimERR. If everythwere below camplow we have a large positive term'

        "Receptors are regulated differently in D1 and D2 msns:"
        if self.type == 'D1-MSN':
            "Note '-=' assignment!!!"
            self.DA_receptor.bmax    -= Kbmax*dtsim*np.sum(HighLimErr)*self.DA_receptor.bmax 
            self.Other_receptor.bmax -= Kbmax*dtsim*np.sum(LowLimErr)*self.Other_receptor.bmax
            
        elif self.type == 'D2-MSN':
            "Note '-=' assignment!!!"
            self.DA_receptor.bmax    -= Kbmax*dtsim*np.sum(LowLimErr)*self.DA_receptor.bmax
            self.Other_receptor.bmax -= Kbmax*dtsim*np.sum(HighLimErr)*self.Other_receptor.bmax
        else:
            print('no valid neuron')
            return
          
        
        

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

    
class PostSynapticNeuronAdapt(PostSynapticNeuron):
    """
    This is a version of the post synaptic neuron that uses a more sophisticated adaptation method. 
    We will assign a SI vector for each receptor (DA_receptor and Other_receptor). Here the first element 
    corresponds to the number of surface receptors and the second element is the number of internalized elements. 
    
    WORK IN PROGRESS!
    """
    def __init__(self, neurontype,  *drugs):
        PostSynapticNeuron.__init__(self, neurontype, *drugs)
        
        
        self.DA_receptor.SIvec = np.array([20, 3])
            
        "Bmax and internalized are maintained for compatability with base class:"
        self.DA_receptor.bmax = self.DA_receptor.SIvec[0]
        self.DA_receptor.internalized = self.DA_receptor.SIvec[1]
        "Transport from surface to internal store will be dependent on AC5 activity"
        
        self.DA_receptor.k_synthesis = 0.01;
        "Total degradation will be dependent on cAMP levels"
        self.DA_receptor.k_degradation = 0.01;
        
        self.Other_receptor.k_synthesis = 0.01;
        "Total degradation will be dependent on cAMP levels"
        self.Other_receptor.k_degradation = 0.01;
        
        
        self.Other_receptor.SIvec = np.array([40, 3])
        "Bmax and internalized are maintained for compatability with base class:"
        self.Other_receptor.bmax = self.Other_receptor.SIvec[0]
        self.Other_receptor.internalized = self.Other_receptor.SIvec[1]
        "The rates  in-and-out of internal store is depending on neuron-type"
        
        if self.type == 'D1-MSN':
           self.DA_receptor.k_surface_to_internal = 0.1;
           self.DA_receptor.k_internal_to_surface = 0.1;
           self.Other_receptor.k_surface_to_internal = 0.1;
           self.Other_receptor.k_internal_to_surface = 0.1;
        elif self.type == 'D2-MSN':
          
            self.DA_receptor.k_surface_to_internal = 0.1;
            self.DA_receptor.k_internal_to_surface = 0.1;
            self.Other_receptor.k_surface_to_internal = 0.1;
            self.Other_receptor.k_internal_to_surface = 0.1;
     
       
    
    def updateBmax(self, dt):
        """
        This method updates Bmax values for DA-receptors and the 'other receptors' in a biologically realistic fashion. 
        Bmax is incremented based on current *caMP* and *cAMPlow* and *cAMPhigh*. 
        Use a time series vector of cAMP values to batchupdate. A time series of cAMP is obtained  using :func:`Get_the_cAMP`. 
        
        :param dt: time step in update vector. This method is supposed to use the same update timestep as in simulations.  
        :type dt: float
        
        """
        
        P = 0
        'LowlimErr  is 0 or 1 depending whether we are above or below lower cAMPlimit. '
        LowLimErr  =   np.heaviside(self.cAMPlow - self.cAMP, 0.5)    
        
 
        '              This term is 1 if cAMP is higher than high-limit'
        HighLimErr =   np.heaviside(self.cAMP - self.cAMPhigh, 0.5)
        'HighLimErr is 0 or 1 depending whether we are above or below upper cAMPlimit. '
        
        
        "Update DA receptors:"
        "Decay depends on cell type:"
        if self.type == 'D1-MSN':    
            K_deg = HighLimErr*self.DA_receptor.k_degradation
        elif self.type == 'D2-MSN':            
            K_deg = LowLimErr*self.DA_receptor.k_degradation
        else:
            print('not known neuron type')
            return
        
        K_S_to_I = self.DA_receptor.k_surface_to_internal*self.DA_receptor.activity()
        "Get the constants into a transition matrix between states: "
        transmat = np.array(
                [ [-K_S_to_I, self.DA_receptor.k_internal_to_surface], 
                [ K_S_to_I, -self.DA_receptor.k_internal_to_surface - K_deg] ] )
        
        "Constant synthesis:"
        offset = np.array( [self.DA_receptor.k_synthesis, 0] )
        
        "Get the change in surface and internalized populations:"
        dSI = np.matmul(transmat, self.DA_receptor.SIvec) + offset
        
        "Increament surface and internalized:"
        self.DA_receptor.SIvec = self.DA_receptor.SIvec + dt*dSI
        
        "½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½"
        
        "Do the same for 'Other_receptor:"

        "Decay depends on cell type: (it is opposite than the DA receptor)"
        if self.type == 'D1-MSN':    
            K_deg = LowLimErr*self.Other_receptor.k_degradation
        elif self.type == 'D2-MSN':            
            K_deg = HighLimErr*self.Other_receptor.k_degradation
        else:
            print('not known neuron type')
            return
        
        K_S_to_I = self.Other_receptor.k_surface_to_internal*self.Other_receptor.activity()
        "Get the constants into a transition matrix between states: "
        transmat = np.array(
                [ [-K_S_to_I, self.Other_receptor.k_internal_to_surface], 
                [ K_S_to_I, -self.Other_receptor.k_internal_to_surface - K_deg] ] )
        
        "Constant synthesis:"
        offset = np.array( [self.Other_receptor.k_synthesis, 0] )
        
        "Get the change in surface and internalized populations:"
        dSI = np.matmul(transmat, self.Other_receptor.SIvec) + offset
        
        "Increament surface and internalized:"
        self.Other_receptor.SIvec = self.Other_receptor.SIvec + dt*dSI

        "½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½"
        
        "Copy from SI to human readable format:"
        self.DA_receptor.bmax = self.DA_receptor.SIvec[0]
        self.DA_receptor.internalized = self.DA_receptor.SIvec[1]


        "Copy from SI to human readable format:"
        self.Other_receptor.bmax = self.Other_receptor.SIvec[0]
        self.Other_receptor.internalized = self.Other_receptor.SIvec[1]
    
          
        

