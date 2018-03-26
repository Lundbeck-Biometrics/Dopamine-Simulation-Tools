# Dopamine-Simulation-Tool
This is a toolbox for making simulations of dopamine (DA) signaling in python. It is developed under Python 3.5.

## Importing
For example:

from DopamineNeuronClass import DA, D1MSN, D2MSN

## class D1MSN(EC50, Gain = 18, Threshold = 0.04, kPDE = 1) (new defaults to appear)
Classes that represent post synaptic neurons. The different classes differ in the way receptors couple to AC5 and default parameters. 

#### EC50
the 50% activation level of dopamine, in nM. 

#### Gain
the gearing between receptor occupancy and activation of AC5
#### Threshold
Threshold for activating AC5
#### occupancy
Value between 0 and 1 that depends on DA concentration and EC50. 
#### AC5
returns current AC5 activity, depends on receptor occupancy, gain and threshold. 
#### cAMP
VAriable that holds current level of cAMP. A function of AC5 and kPDE
#### updateCAMP(dt, C)
Finds the next cAMP level given the current DA concentration. 
#### kPDE 
k-value of the enzyme that degrades cAMP. Based on Yapo et al, J Physiol., 2017.
#### updateG_and_T(dt, cAMP_levels)
Here an np.array of cAMP levels can be used to create an update of gain and threshold parameters. 
This updating runs on a different timescale than the core DA simulations and the timestep dt can be different than elsewhere. 
If cAMP < cAMPlow threshold will be reduced, if cAMP > cAMPhigh then gain will be reduced. 
#### cAMPlow = 0.1; cAMPhigh = 1;
Variables that define upper and lower limits of cAMP. 

## Class D2MSN(EC50, Gain = 18, Threshold = 0.04, kPDE = 1) (new defaults to appear)
Same as above but with differences in methods AC5 and updateG_and_T

 
## class PostSynapticNeuron:
This the parent of D1- or D2 MSN's. They respond to DA conc and generate cAMP. They 
have gain and threshold for activating cascades.  
    

## class feedback([alpha1, alpha2], k_on, k_off, occupancy = 0.5)
Creates a feedback system. Provides a gain attribute which is used to regulate cell firing and release. This class is used by the DA class. 

#### gain (will be updated)
This is the feedback delivered and which is used to regulate cell firing and release. It is a function of the occupancy on the receptor (0 <= occ <= 1). 
This part has a nonlinear and a linear term:

Gain = bool(alpha1)/(1 + alpha1*feedback.occupancy) + alpha2*feedback.occupancy

Note that I always use these separately. If alpha1 = alpha2 == 0, then Gain = 0. 
#### alpha1
Strength of non-linear feedback
#### alpha2
Strength of linear part.

#### occupancy
This is the current fraction of receptors bound by the ligand. Depends on ligand concentration. 

#### update(dt, C)
Updates occupancy and feedback based on latest 

## class DA(area = 'VTA')
Class definition of a DA system. 

This is main class used to make simulations of two compartment DA signalling. Concentraions are always in units of nM. Time is always in units of seconds. Firing rates are in Hz

### How to create an instance 
Create VTA system using DAsys = DA("vta").  This will establish equations that corresponds to mesolimbic signaling. Ie. VTA  > nucleus accumbens DA signaling.

Create SNc DA system using DA("SNC"). System will be set up for simulating nigrostriatal signalling.  Ie SNc -> dorsal striatum

The difference between these two systems is the amount of DA released and reuptake. 

Arguments are not case sensitive. All DA systems contain two instances of feedback-classes which represent somatodendritic and terminal autoreceptors respectively. 

### Attributes and methods in DA class: 

#### update(dt,  nu_in = 5, e_stim = False):
dt = float Euler timestep. 0.01 seconds is often OK. 
nu_in = float input cell firing rate in Hz. Default is 5 Hz. The actual firing rate will include negative somatodendritic feedback. Steady state with default input is ~4 Hz. 
e_stim = bool. If True somatodendritic feedback will be bypassed. Used to emulate experiments where electric or optogenetic simulus drives neurons. 

This method updates the status of feedback systems and the DA concetrations in each compartment. 

#### D2soma
Instance of feedback-class. Representing somatodendritic feedback. 

Created with 'feedback([0, 10], 1e-2, 10, 0)'

The D2soma.gain attribute is in units of Hz and is subtracted from the firing rate. 


#### D2term
Representing terminal D2 autoreceptors.

Created with D2term = feedback([3. , 0.], 0.3e-2, 0.3)

The D2term.gain attribute is multiplied to the release quanta. 

#### NNeurons (=100)
The number of DA neurons in the system. When NNeurons = 100,  the release and uptake will correspond to litterature parameters for intact. For denervation studies reduce manually for the evoked instance, release and uptake will be reduced accordingly. 

#### area ( = "SNc" / "VTA")
Describes the anatomical location of the cell bodies. If other area that the two standard areas is given, the DA class issues a warning.
#### Vmax_pr_neuron
This is the Michalis menten reuptake Vmax pr neuron in the simulation.  

= 40 nM/s if area == "SNc"

= 15 nM/s if area == "VTA"

If other area than VTA or SNC is chosen these attributes are not set automatically and must be set manually for the class instance. 

#### Gamma_pr_neuron

= 4 nM if area == "SNc"  

= 2 nM/s if area == "VTA"

If other area than VTA or SNC is chosen these attributes are not set and must be set manually for the class instance.  

The values present the incremental release pr action potentional of a single DA axon WHEN AUTORECEPTORS ARE FULLY BLOCKED. Expect that under steady state cell firing the actual release to action potential is half... 

####  Vmax_pr_neuron_soma = 2
Michaelis Menten Vmax for soamtodendritic compartment. 

#### self.Gamma_pr_neuron_soma = 0.2;
Incremental release pr actino potential for single neuron. 

#### self.Conc_DA_soma
This is the somatodendritic DA concentration and is updated by method 'update'

#### self.Conc_DA_term
This is the terminal DA concentration and is updated by method 'update'

#### Km = 160.0; 
MicMen reuptake parameter. Affected by cocaine or methylphenidate

#### k_nonDAT = 0.0; 
First order reuptake constant. Change if needed. 

#### Precurser = 1.0; 
DA release multiplier. Change this to 2 or 3 to simulate L-dopa. 



# OLD STUFF

## DAsystem class (obsolete)
Sets up a DA system. Import this class using 'from MouseClassV5 import DASimulation'.

For example see the 'DA_run1.py' script 

## MouseClass (obsolete)
This is an old project of economic decion making in rodents. The mouse class simulates a mouse that likes reward and will press levers to get it. It is motivated by dopamine concentrations. It contains a simple version of a DA-system which is used here as a starting point for a more generic DA simulation. 

See 'mouserun1.py' for an example... 

#### MarkovLeverTool
This is a list of functions used by the mouse class

