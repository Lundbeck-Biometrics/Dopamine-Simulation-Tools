#Copied from notebook 'Adaptations'
# coding: utf-8

# 
# Here I want to create a simluation that illustrates gain and threshold adaptations, and how they are influenced by drugs etc 

# In[1]:


from DopamineToolbox import DA, PostSynapticNeuron, Drug
import matplotlib.pyplot as plt
import numpy as np
import copy

#get_ipython().run_line_magic('matplotlib', 'notebook')


# ## First create  firing patterns. One tonic-dominated and one phasiuc dominated. One high intensity phasic
# 
# All are 1000 s. 
# 
#     dasys0 = 50% tonic, 50% phasic
#     dasys1 = 80% tonic, 20% phasic
#     dasys2 = 20% tonic, 80% phasic
#     dasys3 = 50% tonic, 50% high intensity phasic (nuburst = 40 Hz, nu = 5 Hz,Tperios = 2.5)

# In[2]:


Nsys = 7;

dasys = [DA('SNc') for k in range(Nsys)]

Tmax = 400;

dt = 0.01

labels = ['Normal', 
          'High Phasic',
          'High Firing', 
          'low firing',
          'denervated', 
          'Uptake block']

dasys[0].nuin = DA().CreatePhasicFiringRate(dt, Tmax, Tpre = 0.500*Tmax, Tperiod = 0.8 )

dasys[1].nuin = DA().CreatePhasicFiringRate(dt, Tmax, Tpre = 0.5*Tmax, Tperiod = 2., Nuburst = 30)
dasys[2].nuin = DA().CreatePhasicFiringRate(dt, Tmax, Tpre = 0.5*Tmax, Tperiod = 0.8, Nuaverage = 7)
dasys[3].nuin = DA().CreatePhasicFiringRate(dt, Tmax, Tpre = 0.5*Tmax, Tperiod = 0.8, Nuaverage = 3)

dasys[4].nuin = dasys[0].nuin
dasys[4].NNeurons = 25

dasys[5].nuin = dasys[0].nuin
dasys[5].Km = 300;

dasys[6].nuin = dasys[0].nuin
dasys[6].Precu;



Nit = dasys[0].nuin.size

#Allocate output arrays:
for i in range(Nsys):
    dasys[i].daout = np.zeros(Nit)

    print('generating DA system:...', i+1)
    for k in range(Nit):
        dasys[i].update(dt, dasys[i].nuin[k])
        dasys[i].daout[k] = dasys[i].Conc_DA_term
        
    
print('...done')

timeax = dt* np.arange(Nit)

#


    

# # connect with post synaptic neurons
# 
# Establish neurons and allocate output. Calculate the activity vector that will be used to generate cAMP

#%%


d1 = [PostSynapticNeuron('d1') for k in range(Nsys)]
d2 = [PostSynapticNeuron('d2') for k in range(Nsys)]

for k in range(Nsys):

    d1[k].DA_receptor.actout = np.zeros(Nit)
    d1[k].Other_receptor.actout = np.zeros(Nit)
    d2[k].DA_receptor.actout = np.zeros(Nit)
    d2[k].Other_receptor.actout = np.zeros(Nit)
    
    d1[k].campout = np.zeros(Nit)
    d2[k].campout = np.zeros(Nit)

    print('updating post-neurons...', k+1)

    for i in range(Nit):
        d1[k].updateNeuron(dt, dasys[k].daout[i] )
        d1[k].DA_receptor.actout[i]    = d1[k].DA_receptor.activity()
        d1[k].Other_receptor.actout[i] = d1[k].Other_receptor.activity()
    
        
        d1[k].campout[i] = d1[k].cAMP
        
        
        d2[k].updateNeuron(dt, dasys[k].daout[i] )
        d2[k].DA_receptor.actout[i]    = d2[k].DA_receptor.activity()
        d2[k].Other_receptor.actout[i] = d2[k].Other_receptor.activity()
        d2[k].campout[i] = d2[k].cAMP

print('all ...done')

#%%


plt.close(1)
plt.figure(1)
for k in range(Nsys):
    line = plt.plot(timeax, dasys[k].daout)
    line[0].set_label(labels[k])
plt.title('DA input')

plt.legend()

#%%
plt.close(100)
plt.figure(100)
for k in range(Nsys):
    line = plt.plot(timeax,d1[k].campout)
    line[0].set_label(labels[k])
plt.title('D1-camp')
plt.legend();

plt.close(101)
plt.figure(101)
for k in range(Nsys):
    line =     plt.plot(timeax, d2[k].campout)
    line[0].set_label(labels[k])
plt.title('D2-camp')
plt.legend()
# ## Update from receptor activity to caMP in one line
# 
# note to self: must inlude this function in package. Speeds up by 100 times

# In[5]:


kPDE = d1[0].kPDE

Nupds = 500;

dtB = 1e-4

for i in [0] + list(range(Nsys)):
    d1[i].DA_receptor.bmaxit    = np.zeros(Nupds)
    d1[i].Other_receptor.bmaxit = np.zeros(Nupds)


    d2[i].DA_receptor.bmaxit    = np.zeros(Nupds)
    d2[i].Other_receptor.bmaxit = np.zeros(Nupds)
    #Reset the gain for the other conditions to be the endpooint of the i = 0 condition
    
    
    d1[i].DA_receptor.bmax = d1[0].DA_receptor.bmax
    d2[i].DA_receptor.bmax = d2[0].DA_receptor.bmax
    d1[i].Other_receptor.bmax = d1[0].Other_receptor.bmax
    d2[i].Other_receptor.bmax = d2[0].Other_receptor.bmax
   

    for k in range(Nupds):
        d1inp = (d1[i].DA_receptor.bmax*d1[i].DA_receptor.actout - d1[i].Other_receptor.bmax*d1[i].Other_receptor.actout);
        d1inp *= np.heaviside(d1inp,0.5)
        
        d2inp = - (d2[i].DA_receptor.bmax*d2[i].DA_receptor.actout - d2[i].Other_receptor.bmax*d2[i].Other_receptor.actout);
        d2inp *= np.heaviside(d2inp,0.5)
        
        
        d1[i].campout = np.exp(-kPDE*timeax)*np.cumsum(np.exp(kPDE*timeax)*d1inp)*dt; # d1[i].Gain*(d1[i].actout - d1[i].Threshold)*( d1[i].Threshold < d1[i].actout)*dt)
        d2[i].campout = np.exp(-kPDE*timeax)*np.cumsum(np.exp(kPDE*timeax)*d2inp)*dt; # d1[i].Gain*(d1[i].actout - d1[i].Threshold)*( d1[i].Threshold < d1[i].actout)*dt)
          
        d1[i].updateBmax(dtB, d1[i].campout)
        d2[i].updateBmax(dtB, d2[i].campout)

        d1[i].DA_receptor.bmaxit[k]    = d1[i].DA_receptor.bmax
        d1[i].Other_receptor.bmaxit[k] = d1[i].Other_receptor.bmax
        
        d2[i].DA_receptor.bmaxit[k]    = d2[i].DA_receptor.bmax
        d2[i].Other_receptor.bmaxit[k] = d2[i].Other_receptor.bmax
        
    #print('D1: \t\t\t\t\t D2:')
    #print('Gain = ', np.round(d1.Gain, rnd), 'Thold = ', round(d1.Threshold, rnd), '\t\t', 'Gain = ',round( d2.Gain, rnd), 'Thold = ', round(d2.Threshold, rnd))  
    print('... done ... it no', k)

#%%
    
plt.close(10)
plt.figure(10)
plt.subplot(2,1,1)
for k in range(Nsys):
    line = plt.plot(d1[k].DA_receptor.bmaxit)
    line[0].set_label(labels[k])
    
plt.ylabel('D1-Bmax')
plt.legend()



plt.subplot(2,1,2)
for k in range(Nsys):
    line = plt.plot(d1[k].Other_receptor.bmaxit)
    line[0].set_label(labels[k])
plt.ylabel('M4-Bmax')

#%%

plt.close(11)
plt.figure(11)
plt.subplot(2,1,1)
for k in range(Nsys):
    line = plt.plot(d2[k].DA_receptor.bmaxit)
    line[0].set_label(labels[k])
plt.ylabel('D2-Bmax')
plt.legend()
plt.subplot(2,1,2)
for k in range(Nsys):
    plt.plot(d2[k].Other_receptor.bmaxit)

plt.ylabel('A2A-Bmax')

#%%
plt.close(12)
plt.figure(12)
for k in range(Nsys):
    line = plt.plot(timeax, d1[k].campout)
    line[0].set_label(labels[k])
    
plt.legend()

plt.close(13)
plt.figure(13)
for k in range(Nsys):
    plt.plot(timeax, d2[k].campout)

"""
dasys[0].nuin = DA().CreatePhasicFiringRate(dt, Tmax, Tpre = 0.500*Tmax, Tperiod = 0.8 )
dasys[1].nuin = DA().CreatePhasicFiringRate(dt, Tmax, Tpre = 0.5*Tmax, Tperiod = 2., Nuburst = 30)
dasys[2].nuin = DA().CreatePhasicFiringRate(dt, Tmax, Tpre = 0.5*Tmax, Tperiod = 0.8, Nuaverage = 7)
dasys[3].nuin = DA().CreatePhasicFiringRate(dt, Tmax, Tpre = 0.5*Tmax, Tperiod = 0.8, Nuaverage = 3)
dasys[4].nuin = dasys[0].nuin
dasys[4].NNeurons = 25

dasys[5].nuin = dasys[0].nuin
dasys[5].Km = 300;
"""

plt.close(14)
plt.figure(14)
for k in [0, 1]:
    plt.plot(d1[k].DA_receptor.bmaxit, d1[k].Other_receptor.bmaxit)
plt.title('D1-system')
plt.xlabel('D1 bmax')
plt.ylabel('M4R')



plt.close(15)
plt.figure(15)
for k in [0, 5]:
    plt.plot(d2[k].DA_receptor.bmaxit, d2[k].Other_receptor.bmaxit, d2[k].DA_receptor.bmaxit[-1], d2[k].Other_receptor.bmaxit[-1], 'r*')
plt.title('D2-system')
plt.xlabel('D2 bmax')
plt.ylabel('A2A bmax')

#%%

testsys = [1, 2, 3, 4, 5]
plt.close(142)
plt.figure(142)
for k in testsys:
    line = plt.plot([1, d1[k].DA_receptor.bmax/d1[0].DA_receptor.bmax], [1, d1[k].Other_receptor.bmax/d1[0].Other_receptor.bmax], '-o')
    line[0].set_label(labels[k])
plt.legend()
plt.title('D1')
plt.xlabel('D1R')
plt.ylabel('M4R')

plt.close(143)
plt.figure(143)
for k in testsys:
    line = plt.plot([1, d2[k].DA_receptor.bmax/d2[0].DA_receptor.bmax], [1, d2[k].Other_receptor.bmax/d2[0].Other_receptor.bmax], '-o')
    line[0].set_label(labels[k])
plt.legend()
plt.title('D2')
plt.xlabel('D2R')
plt.ylabel('A2AR')
