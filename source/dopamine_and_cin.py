# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 15:49:46 2021

@author: JAKD
"""


import numpy as np
from scipy.ndimage.filters import gaussian_filter1d as gsmooth
import receptors_and_feedbacks as raf



 
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
            if drug.target.lower() in ['d2', 'd2r', 'dad2', 'da-d2']:
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
            else:
                print('This drug is not added:\n name:', drug.name,'\n target:', drug.target)
                print('target is not D2 receptors...')
        self._drugs = drugs   
        self.D2term = raf.TerminalFeedback(3.0, k_on_term, k_off_term, D2occupancyTerm, efficacy)
        self.D2soma = raf.SomaFeedback(10.0, k_on_soma, k_off_soma, D2occupancySoma, efficacy)
       
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
        "first calculate somatodendritic DA:"
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
    
    def CreatePhasicFiringRate(self, dt,  Tmax, Generator = 'GraceBunney', Nuaverage = 5,  Nuburst = 20, Tperiod = 1,  Tpre = 0, dtsmooth = 0.1, CV = 1.5):
        """
        This method creates a NU-time series with a repetivite pattern of bursts and pauses - Grace-bunney-style. The NU time series can be given as input to :func:`update`.
        
        :param dt: time step in output array
        :type dt: float
        :param Tmax: Total time length of output array in seconds, including any tonic presimulation. Actual lengt is floor integer. 
        :type Tmax: float. 
        :param Generator: Selects the type of firing pattern used. Not case sensitive. 'GraceBunney' leads to repetitive pattern of burst and pauses. 'Gamma' leads to randomly generated bursts and pauses
        :type Generator: string
        :param Nuaverage: Average firing rate in Hz in phasic and tonic blocks. 
        :type Nuaverage: float 
        :param Nuburst: Firing rate in Hz in burst epoch in phasic block
        :type Nuburst: float
        :param Tperiod: Duration of a full burst-pause cycles in seconds
        :type Tperiod: float
        :param Tpre: Length of constant firing rate pre-simulation in seconds. Defaults to 0. 
        :type Tpre: float
        :return: numpy array representing firingrate as function of time 
        
        .. todo:: 
            - Implement more types of firing patterns. 
            - Use markov chains to get burst and pauses. 
            - Create more random-like firing pattens
         
        """
        #Switch to lower case
        Generator = Generator.lower()
        assert Generator in ['gracebunney', 'gamma'], 'unknown generator ''%s'' ' %Generator
        
        Tmaxph = Tmax - Tpre
        
        if Generator == 'gracebunney':
            
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
            Nph = np.ceil(Tmaxph/dt);
            #Now the pattern gets expanded
            NUphasic = np.tile(pattern, int(np.ceil(Nph/Npat)));
            
       
            
        elif Generator == 'gamma':
            "Here we generate a firing pattern which is based on gamma distribution. See Dreyer, J Neurosci, 2014. "
            W = dtsmooth/dt;

            meanisi = 1/Nuaverage;

            k = (1/CV)**2;
            th = meanisi/k;
            

            "We make room for extra spikes so that it is unlikiely to run out of data. "
            "Duration of phasic firing:"
            
            
            timeax = np.arange(0, 2*Tmaxph, step = dt);
            Nspikes = int(Tmaxph*Nuaverage*2);

            mother_isi = np.random.gamma(k, th, size = Nspikes);
#            print(mother_isi)
            mothertrain = np.cumsum(mother_isi);
#            print(mothertrain)
#            print(type(mothertrain))
            spikehist = np.histogram(mothertrain, timeax)[0]
#            print(spikehist)
            NUphasic = gsmooth(spikehist.astype(float), W)/dt;
#            NUall = gsmooth(spikehist.astype(float), W);
    
        Nto = int(np.ceil(Tpre/dt));
        NUtonic = Nuaverage*np.ones(Nto);
        #This is one segment containing a tonic and a phasic segment
        NUall = np.hstack( (NUtonic, NUphasic) );
            #Now this is expanded into a repetivive pattern. 
        #Trunctate if actual number of elements exceeds Tmax  
        realN = int(np.floor(Tmax/dt))
        
        return NUall[0:realN]
    

    
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
    
    Firing rate is taken from <https://www.sciencedirect.com/science/article/pii/S0014488613001118?via%3Dihub> 
    and the amount of inhibition by D2 receptors is from     https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3864134/
    
    """
    def __init__(self, *drugs):
        print('Creating TAN-interneuron release and AChE decay')
        self.k_AChE = 1.2;
        self.NNeurons = 100;
        self.gamma = 30
        self.gamma1 = self.gamma/self.NNeurons;
        self.nu = 5; "Initial firing rate. Will be updated and report the actual firing rate of the TAN's"
        
        self.Conc_ACh = self.gamma*self.nu/self.k_AChE;
        
        k_on_d2 = np.array([1e-2])
        k_off_d2 = np.array([10.0])
        efficacy_d2 = np.array([1])
        
        k_on_m4 = np.array([0.003])
        k_off_m4 = np.array([0.3])
        efficacy_m4 = np.array([1])
        
        D2Somareceptor_init_occupancy = np.array([0.05])
        M4receptor_init_occupancy = np.array([0.35])
        
        for drug in drugs:
            if drug.target.lower() in ['d2', 'd2r', 'dad2', 'da-d2']:
                print('Adding D2active drugs to the CholN')
                print('  D2-competing drug: ' + drug.name)
                print('  on-rate: ', drug.k_on)
                k_on_d2 = np.concatenate( ( k_on_d2 , [drug.k_on] ))
               
                print('  off-rate: ',  drug.k_off)
                k_off_d2 = np.concatenate( ( k_off_d2 , [drug.k_off] ))
               
                print('  inital occupancy: ',  0)
                D2Somareceptor_init_occupancy = np.concatenate( ( D2Somareceptor_init_occupancy, [0]))
                
                print('  efficacy: ', drug.efficacy)
                efficacy_d2 = np.concatenate( (efficacy_d2, [drug.efficacy]))
            elif drug.target.lower() in ['m4', 'm4r', 'achm4', 'ach-m4']:
                print('Adding M4active drugs to the CholN')
                print('  M4-competing drug: ' + drug.name)
                print('  on-rate: ', drug.k_on)
                k_on_m4 = np.concatenate( ( k_on_m4 , [drug.k_on] ))
               
                print('  off-rate: ',  drug.k_off)
                k_off_m4 = np.concatenate( ( k_off_m4 , [drug.k_off] ))
               
                print('  inital occupancy: ',  0)
                M4receptor_init_occupancy = np.concatenate( ( M4receptor_init_occupancy, [0]))
                
                print('  efficacy: ', drug.efficacy)
                efficacy_m4 = np.concatenate( (efficacy_m4, [drug.efficacy]))
            else:
                print('This drug is not added:\n name:', drug.name,'\n target:', drug.target)
                print('target is not D2 or M4 receptors...')
                
        self.D2soma = raf.SomaFeedback(30.0, k_on_d2, k_off_d2, D2Somareceptor_init_occupancy, efficacy_d2)
        
        self.M4terminal = raf.TerminalFeedback(3, k_on_m4, k_off_m4, M4receptor_init_occupancy, efficacy_m4)
        
    def update(self, dt,  DAD2_Conc, nu_in = 6):
        """
        Here we update ACh concentrations. We use the Mic-Men data provided when initializing the neuron.
        
        :param dt: Time step in seconds
        :type dt: float
        :param nu_in: Input firing rate of the TAN in Hz. Default is 6 Hz, which corresponds to the *uninhibited* firing rate. (Prosperitti et al, Exp Neurology, 2013)
        :param DAD2_Conc: Input concentration of D2 binding ligands. Conc[0] is DA concentraion in nM
        :param DAD2_Conc: numpy array
        """
        
        "Update the Chol's D2 receptors:"
        self.D2soma.updateOccpuancy(dt, DAD2_Conc) 
        self.M4terminal.updateOccpuancy(dt, self.Conc_ACh)
        self.nu = np.maximum(nu_in - self.D2soma.gain(), 0)
        R = np.random.poisson(self.NNeurons*self.nu*dt)*self.M4terminal.gain()
        self.Conc_ACh += np.maximum(self.gamma1*R - dt*self.k_AChE*self.Conc_ACh, -self.Conc_ACh)
        
        
class DA_CIN():
    """
    Class of DA system with Cholinergic interneuron interactions
    """
    def __init__(self, area='VTA', *drugs):
        self.DA = DA(area, *drugs)
        self.CIN = Cholinergic(*drugs)
        self.DA.NAchR = raf.receptor(k_on=0.003, k_off=0.3, occupancy=0.35)
        "This constant determines how much of direct DA release there is by ACh"
        self.dopaminemodulation = 0
        "This constant determines how much girk channels are de-coupled by ACh"
        self.girkmodulation = 10
        "We model CIN release as non-synchronized release from DA terminals:"
        self.single_group_fraction = 100
        self.Nterminals = self.DA.NNeurons*self.single_group_fraction
        self.gamma_da_cin = self.DA.Gamma_pr_neuron/self.single_group_fraction
        
    def update(self, dt, NU_da=5, NU_cin=6, e_stim_da = False, e_stim_cin= False, 
               Conc_DAN_receptor_ligands = np.array([0]), Conc_CIN_receptor_ligands = np.array( [0])):
        
        self.DA.update( dt, nu_in = NU_da, e_stim = e_stim_da, Conc=Conc_DAN_receptor_ligands)
        Conc_CIN_receptor_ligands[0] = self.DA.Conc_DA_term
        self.CIN.update(dt,  Conc_CIN_receptor_ligands, nu_in = NU_cin)
        self.DA.NAchR.updateOccpuancy(dt,  self.CIN.Conc_ACh)
        NAchR = self.DA.NAchR.activity()
        self.DA.D2term.alpha = self.girkmodulation*NAchR/(NAchR + 0.7)
        self.DA.Conc_DA_term += dt*self.gamma_da_cin*np.random.poisson(self.single_group_fraction*self.dopaminemodulation*NAchR)

