# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 09:32:34 2018

@author: jakd
"""

import numpy as np

class receptor:
    """
    This is a basic class for all receptors. It can handle several ligands. 
    """
    def __init__(self, k_on = 1, k_off = 1, occupancy = 0, efficacy = 1):
        self.k_on = 1.0*np.array(k_on);
        self.k_off = 1.0*np.array(k_off);
        self.occupancy  = 1.0*np.array(occupancy);
        self.occupancy  *= np.ones(self.occupancy.size)
        "Make sure that efficacy has same size as occupancy. Default efficacy is 1. "
        tempeff = 1.0*np.array(efficacy);
        if tempeff.size == 1:
            self.efficacy = tempeff*np.ones(self.occupancy.size)
        else:
            self.efficacy = tempeff
       
        
    def updateOccpuancy(self, dt, Cvec ):
        free = 1 - np.sum(self.occupancy);
        d_occ = free*self.k_on*Cvec - self.k_off*self.occupancy
#        print(type(d_occ), d_occ.size)
#        print(type(self.occupancy), self.occupancy.size)
        self.occupancy += dt*d_occ;
        
    def activity(self):
        return np.dot(self.efficacy, self.occupancy)

class PostSynapticNeuron:
    """
    This is going to represent D1- or D2 MSN's. They respond to DA conc and generate cAMP. They 
    have DA receptors and attributes like gain and threshold for activating cascades.  
    Future versions may have agonist and antagonist neurotransmitters
    """
    def __init__(self, k_on = np.array([1e-2]), k_off = np.array([10.0]), Gain = 10, Threshold = 0.05,  kPDE = 0.1, efficacy = np.array([1]), *drugs):
    
        self.Gain = Gain;
        self.Threshold = Threshold;
        self.kPDE = kPDE;
        self.cAMP = 2*self.cAMPlow;
    
        tempoccupancy = np.array([0]);
        for drug in drugs:
            print('\n\nAdding drug to post synaptic neuron:')
            print('  DA-competing drug: ' + drug.name)
            print('  on-rate: ', drug.k_on)
            k_on = np.concatenate( ( k_on , [drug.k_on] ))
           
            print('  off-rate: ',  drug.k_off)
            k_off = np.concatenate( ( k_off , [drug.k_off] ))
  
            print('  inital occupancy: ',  0)
            tempoccupancy = np.concatenate( ( tempoccupancy, [0]))
            print('  efficacy: ', drug.efficacy)
            efficacy = np.concatenate( (efficacy, [drug.efficacy]))
            
        self.DA_receptor = receptor(k_on, k_off, tempoccupancy, efficacy)
        self.DA_receptor.ec50 = k_off/k_on
    
    "High - and low limits for 'normal' cAMP"
    cAMPlow = 0.1;
    cAMPhigh = 10;
    "Offsets that regulate how much gain and treshhold are regulated"
    Gainoffset = 0.01;
    Tholdoffset = - 0.99;


    def updateCAMP(self, dt):
        self.cAMP += dt*(self.AC5() - self.kPDE*self.cAMP)
        
    def updateNeuron(self, dt, C_ligands):
        self.DA_receptor.updateOccpuancy(dt, C_ligands)
        self.updateCAMP(dt)
        
    def updateG_and_T(self, dt, cAMP_vector):
        """
        This method updates gain and threshold in a biologically realistic fashion. G&T is incremented slowly based on curren caMP.
        batch updating gain and threshold. Use a vector of cAMP values. dt is time step in update vector
        """
        dT = np.heaviside(cAMP_vector - self.cAMPlow, 0.5) + self.Tholdoffset;
        "NOTE SIGN BELOW: In D2MSN's Tholdspeed must be negative!"
        self.Threshold += self.Tholdspeed*np.sum(dT)*dt/cAMP_vector.size; 
        self.Threshold = np.maximum(0, self.Threshold)
        
        dT = - np.heaviside(cAMP_vector - self.cAMPhigh, 0.5) + self.Gainoffset;
        self.Gain += self.Gainspeed*np.sum(dT)*dt/cAMP_vector.size
        self.Gain = np.maximum(0, self.Gain)
        
    def Fast_updateG_and_T(self, cAMP_vector, Gain_guess = 0, Thold_guess = 0):
        """
        Here we provide a fast method to reach end-point G & T. Nature will not do it this way, but this is faster
        """
        if Gain_guess == 0:
            Gain_guess = self.Gain;
        if Thold_guess == 0:
            Thold_guess = self.Threshold;
            
        Flow = np.percentile(cAMP_vector, self.Gainoffset*100)
        Fhigh = np.percentile(cAMP_vector, -100*self.Tholdoffset)#Note thold-offset is normally negative
  
        An = (self.cAMPhigh - self.cAMPlow)/(Fhigh - Flow)
        Bn = (self.cAMPlow - An*Flow)/(Gain_guess*An)

        self.Threshold =  Thold_guess - np.sign(self.Tholdspeed)*Bn
        self.Gain = An*Gain_guess
        
    def __str__(self):
        
        retstr = \
        'Receptor on-rate = ' + str(self.DA_receptor.k_on) + ' nM^-1 s^-1 \n'\
        'Receptor off-rate= ' + str(self.DA_receptor.k_off) + ' s^-1\n'\
        'Receptor EC50 = ' + str(self.DA_receptor.k_off/self.DA_receptor.k_on) + ' nM \n'\
        'Current input mapping: \n' \
        '  Gain     = ' + str(self.Gain) + '\n'\
        '  Treshold = ' + str(self.Threshold) + '\n'\
        'Current cAMP level: \n'\
        '  cAMP     = ' + str(self.cAMP) 
        
        return retstr

    
    
class D1MSN(PostSynapticNeuron):
    """
    D1-MSN
    """
    def __init__(self, EC50 = np.array([1000]), Gain = 30, Threshold = 0.04, kPDE = 0.10, *drugs):
        k_on = np.array([ 1e-2]);
        k_off = k_on*EC50;
        DAefficacy = np.array([1]);
        PostSynapticNeuron.__init__(self, k_on, k_off , Gain, Threshold, kPDE, DAefficacy,  *drugs);
        
    Gainspeed = 20;
    Tholdspeed = 2;
        
    def AC5(self):
        return self.Gain*(self.DA_receptor.activity() - self.Threshold)*(self.DA_receptor.activity() > self.Threshold)
#    def updateG_and_T(self, dt, cAMP_vector):
#        "batch updating gain and threshold. Use a vector of cAMP values. dt is time step in update vector"
#        dT = np.heaviside(cAMP_vector - self.cAMPlow, 0.5) - 0.99;
#        self.Threshold += np.sum(dT)*dt/cAMP_vector.size
#        #print("T=" , self.Threshold)
#        dT = - np.heaviside(cAMP_vector - self.cAMPhigh, 0.5) + 0.01;
#        self.Gain += 10*np.sum(dT)*dt/cAMP_vector.size
#        #print("G=" , self.Gain)
    def __str__(self):
        retstr = '\n This is a D1-MSN. AC5 is *activated* by DA.\n\n'\
        + PostSynapticNeuron.__str__(self);
        
        return retstr
    
    
    
    
class D2MSN(PostSynapticNeuron):
    """
    Almost like D1 MSNs but cDA regulates differently and threshold is also updated differently
    """
    def __init__(self, EC50 = np.array([1000]), Gain = 30, Threshold = 0.04, kPDE = 0.10, *drugs):
        k_on = np.array([ 1e-2]);
        k_off = k_on*EC50;
        DAefficacy = np.array([1]);
        PostSynapticNeuron.__init__(self, k_on, k_off , Gain, Threshold, kPDE, DAefficacy,  *drugs);
    Gainspeed = 20;
    Tholdspeed = -2;
  
    
      
    def AC5(self):
        return self.Gain*(self.Threshold - self.DA_receptor.activity())*(self.DA_receptor.activity() < self.Threshold)

    def __str__(self):
        retstr = 'This is a D2-MSN. AC5 is *inhibited* by DA.\n'\
        + PostSynapticNeuron.__str__(self)
        
        return retstr   
    

 
class TerminalFeedback(receptor):
    """ DA terminal Feedback loop """
    def __init__(self, alpha, k_on, k_off, occupancy = 0.5, efficacy = 1):
        receptor.__init__(self, k_on, k_off, occupancy, efficacy)
        self.alpha = alpha;
    
    def gain(self):
        return 1/(1 + self.alpha*self.activity())    
      
class SomaFeedback(receptor):
    def __init__(self, alpha, k_on, k_off, occupancy = 0., efficacy = 1):
        receptor.__init__(self, k_on, k_off, occupancy, efficacy)
        self.alpha = alpha;
    
    def gain(self):
        "Make sure gain is bigger than 0. "
        return np.maximum(0, self.alpha*self.activity());
    
 
class DA:
    """
    This is a dopamine class. Sets up a set of eqns that represent DA. Parameters depend on area. 
    Create instances by calling DA(""VTA"") or DA(""SNC""). Area argument is not case sensitive.
    Update method uses forward euler steps. Works for dt <= 0.01 s
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
            
        self.D2term = TerminalFeedback(3.0, k_on_term, k_off_term, D2occupancyTerm, efficacy)
        self.D2soma = SomaFeedback(10.0, k_on_soma, k_off_soma, D2occupancySoma, efficacy)
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
            self.Gamma_pr_neuron = 400.0/Original_NNeurons; #This value is deliberately high and anticipates a ~50% reduction by terminal feedback
        else:
            print('You are in unknown territory.\n Vmax_pr_neuron and Gamma_pr_neuron are not set.')
            self.Vmax_pr_neuron = 0
            self.Gamma_pr_neuron = 0; 
        self.Vmax_pr_neuron_soma = 200.0/Original_NNeurons;
        self.Gamma_pr_neuron_soma = 20.0/Original_NNeurons;
        self.Conc_DA_soma = 50.0;
        self.Conc_DA_term = 50.0;
        
   
    Km = 160.0; #MicMen reuptake parameter. Affected by cocaine or methylphenidate
    k_nonDAT = 0.04; #First order reuptake constant.  Budygin et al, J. Neurosci, 2002
    Precurser = 1.0; # Change this to simulate L-dopa
    
    def update(self, dt, nu_in = 5, e_stim = False, Conc = np.array([0.0])):
        """
        This is the update function that increments DA concentraions. Argumet is 'dt'. 
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
        This is a function that calculates analytical values of steady-state DA concentrations and standard deviaion of baseline.
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
    def CreatePhasicFiringRate(self, dt,  Tmax, Tpre = 0, Tperiod = 1, Nuburst = 20,  Nutonic = 5):
        """
        This function creates a NU-time series with a repetivite pattern of bursts and pauses - Grace-bunney-style.
        Inputs are time step dt, Max time of total time NU-series, Time of single repetition, firing rate in bursts and average firing rate, and a pre-time with constant cell firing. 
        """
       
        "## First Generate single burstpause pattern:"
        "Number of spikes in a signle burst:"
        spburst = Nutonic*Tperiod;
        "Number of timesteps for a single burst:"
        burstN = int(spburst/Nuburst/dt);
        "Number of timesteps in single period. "
        Npat   = int(Tperiod/dt);
        pattern = np.zeros(Npat);
        "First part is burst, the rest remains 0 - its a pause :o)"
        pattern[0:burstN] = Nuburst;
        
        
        "Expand until many burst-pause patterns"
        dtph = Tmax - Tpre; #total duration of phasic pattern.
        Nph = int(dtph/dt);
        Nto = int(Tpre/dt);
        #Now the pattern gets expanded
        NUphasic = np.tile(pattern, int(Nph/Npat));
        #The tonic segment is X times as long as the phasic
        NUtonic = Nutonic*np.ones(Nto);
        #This is one segment containing a tonic and a phasic segment
        NUstep = np.hstack( (NUtonic, NUphasic) );
        #Now this is expanded into a repetivive pattern. 
        return NUstep
    

    
    def __str__(self):
        mda, sda = self.AnalyticalSteadyState()
        retstr = \
        'Dopamine neuron. Cell body located at ' + self.area + '\n'\
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


class DrugReceptorInteraction:
    """
    Returning interaction between a drug and a receptor
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
        'Name:\t' + self.name + '\n' +\
        'This efficacy:\t' + str(self.efficacy) + '\n' + \
        '1: full agonist\n'  +\
        '0: full antagonist\n' + \
        'between 0 and 1: partial agonist\n'
        
        return class_string


class Drug(DrugReceptorInteraction):
    def __init__(self, name = 'Default Agonist', target = 'D2R', k_on = 0.01, k_off = 1.0, efficacy = 1.0):
        DrugReceptorInteraction.__init__(self, name, target, k_on, k_off, efficacy)
        print("Creating Drug-receptor class. More receptor interactions can be added manually! \nUse <name>.<target> = DrugReceptorInteraction(\'name.tagret\', target, kon, koff, efficacy)")
    def Concentration(self, t,  dose, t_infusion = 0, k12 = 0.3/60, k21 = 0.2/60, k_elimination = 0.468/60):
        "Calculates drug concentration using two-compartment PK"
        sumk = k12+k21+k_elimination
        D = np.sqrt(sumk**2 - 4*k21*k_elimination);
        a = 0.5*(sumk + D);
        b = 0.5*(sumk - D);
        C= dose*k12/(a-b)*( np.exp(-b*(t-t_infusion) ) - np.exp(-a*(t-t_infusion)) );
        tdrug = t >= t_infusion;
        c2 = C*tdrug;

        return c2
 
        
def AnalyzeSpikesFromFile(FN, dt = 0.01, area = 'vta', synch = 'auto', pre_run = 0, tmax = 600):
    """
    This is a function that utilized DA-classes to analyze spikes from experimental recordings
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
        fp.close()
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

        smW = 0.3012*mISI; #smoothing in magic window
        W = smW/dt;
    else:
        W = synch/dt;
    
    DTtrans =  mISI - spikes[0]
    print("Adjusting start-gab by forward-translating" , DTtrans , ' s')
    spikes += DTtrans; 
    
     
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
        def __str__(self):
            "Note that we refer to future attributes being set below"
            
            class_str = "\n Results from running " + FN + ".\n\n"\
            "DA system parameters:\n" + \
            "   Area:" + self.da.area         
            return class_str
        
    Result = Res();           
    Result.da = DA(area);
    Result.d1 = D1MSN();
    Result.d2 = D2MSN();
    
    Result.File = FN;
    Result.InputFiringrate = NUall;
    Result.MeanInputFiringrate = mNU;
    Result.timeax = tall;
    Result.DAfromFile = np.zeros(NUall.size)
    
    Result.d1.AC5fromFile = np.zeros(NUall.size)
    Result.d1.cAMPfromFile = np.zeros(NUall.size) 
    
    Result.d2.AC5fromFile = np.zeros(NUall.size)
    Result.d2.cAMPfromFile = np.zeros(NUall.size)
    
    print("Adjusting post synaptic thresholds and initial DA concentration to this file:")
    
    mda, sda = Result.da.AnalyticalSteadyState(mNU);
    mdar = mda/(mda + Result.d1.DA_receptor.ec50);
    
    Result.d1.Threshold = mdar;
    Result.d2.Threshold = mdar;

    
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
    
