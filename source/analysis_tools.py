# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 15:32:40 2021

@author: JAKD
"""
import copy
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d as gsmooth
import postsynaptic as post 

class Simulation: 
    """
    This is an container handling results and main functions used in :func:`AnalyzeSpikesFromFile`. 
    It has methods for reading spikes from a filename. In future versions we should clean up the information flow here...  
    
    In the current implementation, :func:`AnalyzeSpikesFromFile` will add a lot of extra attributes to this object.
    for exampl the :class:`DA` used in the simulation and the simulation outputs. 
    
    :param process: Flag that controls whether :func:`AnalyzeSpikesFromFile` will run a simulation or just keep the spikes
    :type process: Bool
    """  
    def __init__(self, process):
        self.__process = process
        self.File = ''

    def __str__(self):
        "Note that we refer to future attributes being set below"
        
        if self.__process:
            class_str = "\n Dopamine simulation results from running " + self.File + ".\n\n"\
            "DA system parameters:\n" + \
            self.da.__str__()     
        else:
            class_str = "\n Dopamine simulation results from " + self.File + ".\n\n"
            
        return class_str
    
    def read_spikes_from_file(self, FN):
        """Reading spikes from a file. 
        """
        #create empty array for the spikes:
        spikes = np.array([]); 
        
        #print("½½½½½½½½½½½½½½½½½½½½½½½½½½½")
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
        
        return spikes
    
    
    def spikes_to_rate(self, dt, spikes, synch = 'auto', adjust_t = False, tmax = None):
        """Reading spike-timestamps into a firing rate vector. 
        
        
        :param dt: timestep of firing rate vector
        :type dt: float
        :param spikes: Timestamps of action potentials
        :type spikes: numpy array
        :param synch: FWHM in s for other neurons in ensemble. If set to 'auto', the synch is decided based on average firing rate of the input cell. Default is 'auto'
        :type synch: float or 'auto'
        :param adjust_t: adjust offset of time series? Use this if *spikes* has a gap at the beginning.  Default is *False*
        :type adjust_t: bool
        :param tmax: Length of simulation in seconds, default None. If *tmax* is None, tmax will be equal to the last spike in the recording. 
        :type tmax: float
        
        """
        
       
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
            DTtrans = spikes[0]
            print("Moving spikes by" , DTtrans , ' s')
            spikes += - DTtrans; 
        
        if tmax is None:
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
        
        return spikes, NUfile, mNU, synch
       
def AnalyzeSpikesFromFile(ToBeAnalyzed, DAsyst, dt = 0.01, synch = 'auto', pre_run = 0, tmax = None, process = True, adjust_t = False):
    """
    This is a function that uses :class:`DA` and :class:`PostSynapticNeuron`-classes to predict cAMP time series using spikes from experimental recordings. Output is a :class:`Simulation`-object and is further described below
    
    
    * It is based on similar methods as used in `Dodson et al, PNAS, 2016 <https://doi.org/10.1073/pnas.1515941113>`_.
    * It also includes the option to make 'clever' choice of synchrony, described below. 
    * The method decides the optimal values for Bmax for D1 and D2 neuronal systems. Thus pacemaker firing will always translate into low cAMP in D1 and D2 post neurons 

    :param ToBeAnalyzed: Input firing pattern. Two types of valid inputs: *ToBeAnalyzed* can be *Filename* including full path to experimental data file with time-stamps, as described more below. *ToBeAnalyzed* can also be a numpy array of time-stamps.
    :type ToBeAnalyzed: string or numpy array
    :param DAsyst: Instance of :class:`DA`-class that will be used to simulate the file inputs.
    :type DAsyst: Instance of :class:`DA`-class.
    :param dt: Timestep in simulation (dt = 0.01s by default)
    :type dt: float
    :param synch: FWHM in s for other neurons in ensemble. If set to 'auto', the synch is decided based on average firing rate of the input cell. Default is 'auto'
    :type synch: float
    :param pre_run: Number of seconds to run a totally tonic simulation before input firing pattern kicks in. Very useful to detect changes in the firing pattern. Default is 0.
    :type pre_run: float
    :param tmax: Length of simulation in seconds, default None. If *tmax* is None, tmax will be equal to the last spike in the recording. 
    :type tmax: float
    :param process: Indicate if the function should process the timestamps or not. If *True* the method will also include a DA simulation. If *False* the output will only contain the timestams and firing rates of the file. Default is *True*
    :type process: bool
    :param adjust_t: adjust offset of time series? Use this if your data has a gap at the beginning.  Default is *False*
    :type adjust_t: bool
    :return: Result from a DA simulation using *file* as input. 
    :rtype: :class:`Simulation`-object. The attributes of the output depends on the *process* parameter. 
    
    .. note:: 
        Total length of the simulation is tmax + pre_run.    
    
    If *ToBeAnalyzed* is a file: 
        The input must be formatted as timestamps in ascii text.  
        Current upported formats are Spike2 files with WAVMK timestamps (exported .txt files from preselected (spikesorted) channels in Spike2), 
        or just a single column of timestapms (with or without single line of header).)    

    If *ToBeAnalyzed* is a numpy array: 
        The input must be formatted as a single numpy array of timestamps in units of seconds    
    
    About the *synch*-option:
        The synch = 'auto'-option creates a synch value equal to 0.3012 x *mean interspike interval*. 
        For eaxmple if mean firing rate is 4 Hz, the synch will be 0.3012x0.25s = 0.0753s. 
        With this option a perfect pacemaker firing neuron will be translated into a firing rate where the maximum firing rate is 2 times the minimum firing rate. 
    
    The output: 
        The output is modified :class:`Simulation` object. It has attached the DA object, D1 and D2 post synaptic neuron objects. 
        
        The most important attributes are the DA time series, the input firing rate (a smoothed version of the spikes), and the post synaptic cAMP time series
        
    Examples::
        
       >>import DopamineToolbox
       >>import matplotlib.pyplot as plt
       >>import numpy as np
       >>#simulation object for analyzing VTA neurons:
       >>dasys = DA('vta')
       >># Create 4hz pacemaker spike-pattern  
       >>spikes = np.arange(0, 10, step = 0.25)
       >>#The firing pattern could also be the timestapms from a recoding
       >>#Alternatively 'spikes' could be a filename of a file with timestapms
       >>#Run simulation and add 30 seconds constant firing rate:
       >>result = AnalyzeSpikesFromFile(spikes, dasys, pre_run = 30)
       >>print(result)
       >>#plot main outputs:
       >>f, ax = plt.subplots(facecolor = 'w', nrows = 4, sharex = True)
       >>ax[0].plot(result.timeax, result.InputFiringrate)
       >>ax[0].set_title('Firing rate')
       >>ax[1].plot(result.timeax, result.DAfromFile)
       >>ax[1].set_title('DA levels')
       >>ax[2].plot(result.timeax, result.d1.cAMPfromFile)
       >>ax[2].set_title('D1 cAMP response')
       >>ax[3].plot(result.timeax, result.d2.cAMPfromFile)
       >>ax[3].set_title('D2 cAMP response')
    """
    
   
    
    
    Result = Simulation(process);
    
    "If input, *ToBeAnalyzed*, is a string we open the data as a file:"    
    if isinstance(ToBeAnalyzed, str):
        spikes = Result.read_spikes_from_file(ToBeAnalyzed)
        
    elif isinstance(ToBeAnalyzed, np.ndarray):
        print('Using input as spike time-stamps')
        spikes = ToBeAnalyzed
        #Reassign ToBeAnalyzed to be a default string because we need it later:
        ToBeAnalyzed = '<User Spike Times>'
        
    else:
        
        raise TypeError("ToBeAnalyzed mustbe  a string containing a file-name or a numpy array of timestamps. Not %s" % type(ToBeAnalyzed))
        
            
    
    "Note: The spike to rate method may tranlate spikes if adjust_t is True"
    "Note: The paramteter 'synch' will be a float from here on even if it was 'auto'.  "
    spikes, NUfile, mNU, synch = Result.spikes_to_rate(dt, spikes, synch, adjust_t, tmax )
    
    NUpre  = mNU*np.ones(round(pre_run/dt))
    
    
    NUall = np.concatenate((NUpre, NUfile))
    tall = dt*np.arange(NUall.size) + 0.5*dt;
    
    "Get indices for switch betwen tonic and phasic for later use"
    iphasic_on = np.where(tall > pre_run)[0][0] #pre_run >= 0, and tall[0] = dt/2. So guaranteed to get at least one 
    lastspike = spikes[-1] + 1/mNU;
    iphasic_off = np.where(tall > pre_run + lastspike)[0]
    "if pre_run + lastspike > than tall[-1] we need to set the end-point manually:"
    if iphasic_off.size == 0:
        iphasic_off = tall.size - 1;
    else:
        iphasic_off = iphasic_off[0];
    
    print('File time on: ',  tall[iphasic_on] - dt/2, ' s')    
    print('File time off: ', tall[iphasic_off] + dt/2, ' s')    
    
    "Setting up output from the function and populating attributes from the DA D1MSN and D2MSN classes:"
    
    
        
       

    Result.File = ToBeAnalyzed;
    Result.InputFiringrate = NUall;
    Result.MeanInputFiringrate = mNU;
    Result.timeax = tall;
    Result.timestamps = spikes + pre_run;
    Result.synch = synch
   
    if process == False:
        print('Returning just firingrate and timestamps and exit')
        return Result
    Result.DAfromFile = np.zeros(NUall.size);        
    
    #We copy the DA model to the class so that it will remain a static image of the DA model used. 
    Result.da = copy.deepcopy(DAsyst);
    
    Result.d1 = post.PostSynapticNeuron('d1');
    Result.d2 = post.PostSynapticNeuron('d2');
    
    
    Result.d1.AC5fromFile = np.zeros(NUall.size)
    Result.d1.cAMPfromFile = np.zeros(NUall.size) 
    
    Result.d2.AC5fromFile = np.zeros(NUall.size)
    Result.d2.cAMPfromFile = np.zeros(NUall.size)
    
    print("Adjusting post synaptic thresholds and initial DA concentration to this file:")
    
    mda, sda = Result.da.AnalyticalSteadyState(mNU);
    md1 = mda/(mda + Result.d1.DA_receptor.ec50);
    md2 = mda/(mda + Result.d2.DA_receptor.ec50);
    
    
    Result.d1.Other_receptor.bmax = Result.d1.DA_receptor.bmax*md1/Result.d1.Other_receptor.occupancy
    Result.d2.Other_receptor.bmax = Result.d2.DA_receptor.bmax*md2/Result.d2.Other_receptor.occupancy

    
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
    
    Result.analytical_meanDA = mda;
    Result.analytical_stdDA = sda;
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
