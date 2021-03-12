# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 09:32:34 2018

@author: jakd@lundbeck.com 

This is collection of classes and functions that are used to create the framework for simulations of dopamine signaling and adaptations in post synaptic interpretation of dopamine signals. 
Most important classes and functions are 

   - :class:`DA` which represents a full dopamine system with somatodendritic and terminal DA release. 
   - :class:`PostSynapticNeuron` which can be configured to represent the post synaptic readout of DA. 
   - :func:`AnalyzeSpikesFromFile` which is used to analyze data from experimental recordings. 
   
If the file is run as a script it will generate a set of simulations and plot the results. Otherwise the file can be imported into a new simulation. 
For example as::
        
        >>> import DopamineToolbox as DATB
        >>> #Create a VTA-based dopamine system called mesolimb:
        >>> mesolimd = DATB.DA('VTA')
   
.. todo::
   - Create a simulation-class that takes firing rate vector as input. And that collects DA, D1, D2 systems at once.


 
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




        
if __name__ == "__main__":
    """
    This part will be run if the file is being run as a script (in other words not imported as a plugin)
    We create two example runs and plot the results: 
        1. A simulation using a construted firing rate. The example has 1 part tonic cell firing and one part phasic cell firing.
           This simulation uses the default values of D1 and D2bmax. 
        2 Another simulation uses the cell firing of a single neuron to create an estimate of the DA concentraions 
          and post synaptic activation. This part is used to analyze 
    """
    import matplotlib.pyplot as plt 
    import numpy as np
    
    import analysis_tools as tools
    import receptors_and_feedbacks as raf
    import drugs
    import dopamine_and_cin as dancin   
    import postsynaptic as post
    
    print('Running simple simulation:')
    dt = 0.01;
    Tmax = 200.0
    "Create objects"
    da = dancin.DA('VTA')
    d1 = post.PostSynapticNeuron('D1')
    d2 = post.PostSynapticNeuron('D2')
    
    
    "Create firing rate"
    NU = da.CreatePhasicFiringRate(dt, Tmax, Tpre=0.5*Tmax, Generator='Gamma', CV = 2)
    Nit = len(NU)
    
    NU2 = da.CreatePhasicFiringRate(dt, Tmax, Tpre = 0.25*Tmax, 
                                    Nuaverage=2, Nuburst=20, Tperiod=10)
    
    
    timeax = dt*np.arange(0, Nit)
    "Allocate output-arrays"
    DAout   = np.zeros(Nit)
    D1_cAMP = np.zeros(Nit)
    D2_cAMP = np.zeros(Nit)
    
    
    "Run simulation"
    for k in range(Nit):
        da.update(dt, NU[k])
        d1.updateNeuron(dt, da.Conc_DA_term)
        d2.updateNeuron(dt, da.Conc_DA_term)
        DAout[k] = da.Conc_DA_term
        D1_cAMP[k] = d1.cAMP
        D2_cAMP[k] = d2.cAMP
        
    
        
    "plot results"
    f, ax = plt.subplots(dpi = 150, facecolor = 'w', nrows = 2, sharex = True)
    line = ax[0].plot(timeax, DAout, [0, Tmax], [0,0], 'k--')
    line[0].set_linewidth(1)
    line[1].set_linewidth(0.5)
    ax[0].set_title('Simulation output: Tonic and Phasic DA firing')
    ax[0].set_ylabel('DA (nM)')
    line = ax[1].plot(timeax, D1_cAMP, timeax, D2_cAMP, linewidth=1)
    line[0].set_label('D1-MSN')
    line[1].set_label('D2-MSN')
    
    ax[1].set_ylabel('cAMP')
    ax[1].legend()
    ax[1].set_ylim([0, 20])
    ax[-1].set_xlabel('Time (s)')
    
    "Run with drug:"
    
    drug = drugs.Drug()
    Cdrug = 100# nM
    
    da_with_drug = dancin.DA('SNC', drug)
    d1_with_drug = post.PostSynapticNeuron('D1')
    d2_with_drug = post.PostSynapticNeuron('D2', drug)
    
    "RE-Allocate output-arrays"
    DAout_w_drug   = np.zeros(Nit)
    D1_cAMP_w_drug = np.zeros(Nit)
    D2_cAMP_w_drug = np.zeros(Nit)
    
    # Cligands = np.array([0, Cdrug])
    # "Run simulation"
    # for k in range(Nit):
    #     da_with_drug.update(dt, NU[k],Conc= Cligands)
    #     d1_with_drug.updateNeuron(dt, da_with_drug.Conc_DA_term)
    #     Cligands[0] = da_with_drug.Conc_DA_term
    #     d2_with_drug.updateNeuron(dt, Cligands)
    #     DAout_w_drug[k] = da_with_drug.Conc_DA_term
    #     D1_cAMP_w_drug[k] = d1_with_drug.cAMP
    #     D2_cAMP_w_drug[k] = d2_with_drug.cAMP
        
    
    # "plot results"
    # f, ax = plt.subplots(dpi = 150, facecolor = 'w', nrows = 3, sharex = True)
    # line = ax[0].plot(timeax, DAout, timeax, DAout_w_drug, [0, Tmax], [0,0], 'k--')
    # line[0].set_linewidth(1)
    # line[1].set_linewidth(0.5)
    # line[0].set_label('No drug')
    # line[1].set_label('with D2 agonist')
    # ax[0].set_title('Simulation output: Tonic and Phasic DA firing')
    # ax[0].set_ylabel('DA (nM)')
    # ax[0].legend()
    
    # line = ax[1].plot(timeax, D1_cAMP, timeax, D1_cAMP_w_drug, linewidth=1)
    # ax[1].set_ylabel('D1-cAMP')
    # ax[1].set_ylim([0, 20])
    
    # line = ax[2].plot(timeax, D2_cAMP, timeax, D2_cAMP_w_drug, linewidth=1)
    # ax[2].set_ylabel('D2-cAMP')
    # ax[2].set_ylim([0, 20])
    # ax[-1].set_xlabel('Time (s)')
  
    # print('Running via AnalyzeSpikesFromFile:')
    # "We use the same firing rate as before to generate spikes from one cell"
    # "The first part will be a constant firing rate and a perfect tonic pattern."
    # "The mean firing rate for the simulation:"
    # nu_mean = 4;

    # spikes_tonic = np.arange(0, Tmax*0.5, step = 1/nu_mean)
    # phasic_index = np.nonzero(timeax > Tmax*0.5)[0]
    # spikes_phasic = Tmax*0.5 + dt*np.nonzero(np.random.poisson(lam = dt*NU[phasic_index]))[0]
    # spikes = np.concatenate( (spikes_tonic, spikes_phasic) )
    
    # "The AnalyzeSpikesFromFile method can make a tonic prerun by itselv. Here we added the tonic spikes ourselves so we set prerun to be zero"
    # prerun = 0
    # #Run simulation and add  constant firing rate:
    # result = tools.AnalyzeSpikesFromFile(spikes, da, pre_run = prerun)
    # print(result)
    # #plot main outputs:
    # f, ax = plt.subplots(facecolor = 'w', nrows = 4, sharex = True, figsize = (6, 10))
    # ax[0].plot(result.timeax, result.InputFiringrate, label='All simulation')
    # preindx = np.nonzero(result.timeax < prerun)[0]
    
    # ax[0].plot(result.timeax[preindx], result.InputFiringrate[preindx], '--', label = 'Prerun')
    # ax[0].legend(loc=2)
    # ax[0].set_title('AnalyzeSpikesFromFile: Firing rate from simulation')
    # ax[1].plot(result.timeax, result.DAfromFile)
    # ax[1].set_title('DA levels')
    # ax[2].plot(result.timeax, result.d1.cAMPfromFile)
    # ax[2].set_title('D1 cAMP response')
    # ax[3].plot(result.timeax, result.d2.cAMPfromFile)
    # ax[3].set_title('D2 cAMP response')
    
    print('testing cholinergic interneuron')
    cin = dancin.DA_CIN()
    
    DACINout = np.zeros( (Nit, 2) )
    
    
    
    "Run simulation with DA+Ach iteractions"
    "Hypothesis that ach interacts via AR's"
    for k in range(Nit):
        cin.update(dt, NU_da= NU[k], NU_cin=NU2[k])       
        d1.updateNeuron(dt, cin.DA.Conc_DA_term, cin.CIN.Conc_ACh)
        d2.updateNeuron(dt, cin.DA.Conc_DA_term)
        DACINout[k,:] = [cin.DA.Conc_DA_term, cin.CIN.Conc_ACh]
        D1_cAMP[k] = d1.cAMP
        D2_cAMP[k] = d2.cAMP
    
    f, ax = plt.subplots(dpi = 150, facecolor = 'w', nrows = 2, sharex = True)
    
    line = ax[0].plot(timeax, DAout, timeax, DACINout[:,0],timeax, DACINout[:,1], [0, Tmax], [0,0], 'k--')
    line[0].set_linewidth(1)
    line[1].set_linewidth(0.5)
    line[0].set_label('DA')
    line[1].set_label('DA w Nacr feedback')
    line[2].set_label('AChR')
    ax[0].legend()
    
    ax[0].set_title('Simulation output: Tonic and Phasic DA firing')
    ax[0].set_ylabel('DA (nM)')
    line = ax[1].plot(timeax, D1_cAMP, timeax, D2_cAMP, linewidth=1)
    line[0].set_label('D1-MSN')
    line[1].set_label('D2-MSN')
    
    ax[1].set_ylabel('cAMP')
    ax[1].legend()
    ax[1].set_ylim([0, 20])
    
    
    
    ax[-1].set_xlabel('Time (s)')
    

