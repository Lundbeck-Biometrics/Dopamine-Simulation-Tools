import numpy as np
import MarkovLeverToolsV4 as tools
import matplotlib.pyplot as plt

#reload(tools)


class Experiment:
    "This is an experiment-class. Set up with Tmax, criterion, Reward, inital guess, and test-type., If test-type is a list of numbers represeting timepoints, the criteria will permutate at this time. Anneal is used for PR tests"
    
    def __init__(self, Tmax,  crit, Reward, init = [], testtype = 'FR', A = [1]):
        if len(init) == 0:
            init = crit;
            
        self.crit = np.array(crit);
        self.Reward = 1.0*np.array(Reward);
        self.initGuess = 1.0*np.array(init);
        self.type = testtype;
        self.Tmax = 1.0*Tmax;
        self.MaxPress = 500;
        if len(A) == 1:
            A = A*len(crit); #this copies A
        self.Anneal = np.array(A);
        self.TotalCount = 0;
        self.TrialCount = 0;
        self.PressTime = [];
        self.FR_Event = [];
        self.FR_Event.append([0, list(crit)]);
       
        self.RewardEarned = [];
        if np.max(self.crit) > self.MaxPress:
            print("Criterion exceedes Maxinumber of presses")
    def GetNextCrit(self, ThisFR, t = 0):
        "This function updates the criterion for reward."
        if isinstance( self.type, list):
            Nupdates = len(self.FR_Event) - 1;
            if Nupdates < len(self.type):
                Nextupdate = self.type[Nupdates]
                if t >= Nextupdate:
                    temp = ThisFR.tolist();
                    temp0 = temp[1:];
                    temp0.append(temp[0])
                    self.FR_Event.append([t , temp0])
                    return np.array(temp0)
                else:    
                    return ThisFR 
            else:    
                return ThisFR  
                     
        elif self.type == 'PR':
            FR = ThisFR*self.Anneal
           
            self.FR_Event.append([t , list(FR)])
                   
            return FR
        elif self.type == 'PR2':
            FR = [0,0];
            FR[0] = ThisFR[0];
            FR[1] = ThisFR[1]*self.Anneal[1]
           
            self.FR_Event.append([t , list(FR)])           
            return FR
        elif self.type == 'RR':
            FR = tools.GetFR(self.crit, -2)
            return FR
        elif self.type == 'FR':
            return ThisFR;
        else:
            print("unknown schedule type")
       


class DASimulation:
    dt_small = 0.005;
    dt_large = 0.25;
    t = 0;
    Km = 160;
    NNeurons = 100;
    gamma1 = 1;
    
    N0 = 3000 ## molecules released pr release
    Pmax = 0.15 # maximal release prob
    alfa = 0.2 #DA can be release into 20% of availble volume, thus only in free extracellular space. 
    
    Na = 6.022E23#advogrados number
   
    rho1 = 0.0006#density of terminals for single DA axon. 0.0006 in NA, 0.001 in Dorsal Striatum. 
    gamma0=((N0)/(alfa*Na))*rho1*1E24 # 
    

    class Results:
        terminalDA = [];
        sdDA = [];
        NUeff = [];
        
    class DAsystem:
        def __init__(self, Vmax, Gamma):
            self.Vmax = Vmax;
            self.Gamma = Gamma;
        
        DA = 50;  
        
        
    Soma = DAsystem(200, 20);
    Soma.alpha = 0.008; #units: Hz pr nM. 
    
    Terminal        = DAsystem(20*NNeurons, gamma0);
    Terminal.beta   = 2;
    Terminal.kon    = 1e-2 #Presyn AR: On rate in nanomolar! nM^-1 * s^-1
    Terminal.koff   = 0.4; #Presyn AR off rate! s^-1
    Terminal.D2auto = 0.5;
   

    Pmax = 0.15 
         
    
    class Firing:
        nu_tonic_baseline = 4; #this is the baseline firing in absence of inputs
        nu_tonic_max = 20;
        nu_burst = 15;
        spburst = 0.01;
        dtpause = 1;
        nu_tonic = nu_tonic_baseline; #this is the tonic firing rate which may include corrections
        nu_in = nu_tonic_baseline;
        nu_now = 0;
        tonic = True
        
        
    def UpdateDA(self):
        
        #Get correct firing rate
        self.Firing.nu_now = np.max([self.Firing.nu_in - self.Soma.alpha*self.Soma.DA, 0])
        
        rel = np.random.poisson(self.NNeurons*self.Firing.nu_now*self.dt_small);
        #print "NU", self.Firing.nu_now, "nu_in", nu, "rel/dt", rel
        #update s.d. DA:
        dC = rel*self.Soma.Gamma/self.NNeurons - self.dt_small*self.Soma.Vmax*self.Soma.DA/(self.Km + self.Soma.DA);
        self.Soma.DA += dC
        #print "sdDA", self.Soma.DA
        
        
        #update term DA:
        #first D2:
        self.Terminal.D2auto += self.dt_small*( (1 - self.Terminal.D2auto)*self.Terminal.kon*self.Terminal.DA - self.Terminal.D2auto*self.Terminal.koff)
        #print "AR", self.Terminal.D2auto
        P = self.Pmax/(1 + self.Terminal.beta*self.Terminal.D2auto);
        #print "p", P
        dC = P*rel*self.Terminal.Gamma - self.Terminal.Vmax*self.Terminal.DA/(self.Km + self.Terminal.DA)*self.dt_small;
        
        self.Terminal.DA += dC
        
        #print "termDA", self.Terminal.DA
        
    def Saveresults(self, filename):
        L   = len(dir(self.Results))

        print(L)
        
    
class Mouse(DASimulation):
    "This is a mouse class. It is derived from the DA simulation class\\ Most important method is the RunSchedule"
    State = 0;
    LastRPE = [];
    alpha_value = 0.0;
    alpha_entropy = 0.1;
    alpha_RPE = 0.005;
    
    D2affinity = 5;
    DA2rate = 10.0/50;
    
    pause = 2;

    class belief:
        Eloss   = 0;
        Reward  = 0;
        Epress  = 0;        
        Spress  = -1; # standard value for 25% of mean.         
        Hazard  = 0;
        Utility = 0;
     
        
    def __init__(self):
         self.Results.rewards = [];
         self.Results.Epress = [];
         self.Results.Eloss = [];
         self.Results.Psurv = [];
         self.Results.State = [];
         self.Results.Utility = [];
         self.Results.RPE = [];
         #print "check learn rates"
         self.learnrate_press = 0.005
         self.learnrate_loss = 0.05
         self.learnrate_psurv = 0.1
         
         
 
    def RunSchedule(self, Experiment):
  
        
        
        DT = int(self.dt_large/self.dt_small);
        
        
        
        self.SetupBelief(Experiment)
        
        steps = int(np.ceil(Experiment.Tmax/self.dt_small));
        self.SetupResults(steps)
        
        workstate = 0
        ActualReward = False;
        dropout = False;
        klastRPE = 0
        dt_phasic = 0;
        ShannonH = 0;
        timeout = -1;
        RPEvec = np.zeros(Experiment.MaxPress)
        #NewState = self.State;  
        
        FR = Experiment.crit
        

                
        k = 0;
        #simplifying variables: utilitycount prevents overflow of Hazard and utility functions
        utilitycount = np.min([Experiment.TrialCount, Experiment.MaxPress - 1]);
        while k < steps:
            
            
            Response_rate = self.DA2rate*self.Terminal.DA ; #times pr second. 
       
            steps_to_next_press = 1/(Response_rate*self.dt_small)
            nu_tonic = self.Firing.nu_tonic_baseline + self.alpha_value*self.belief.Utility[self.State, utilitycount] + self.alpha_entropy*self.belief.Reward[self.State]*ShannonH
            nu_tonic = np.min([nu_tonic, self.Firing.nu_tonic_max])    
            
            if (k%steps_to_next_press < 1) &  (self.State != 0) :
                   
                   
                Experiment.PressTime.append(k); # the time point is kept by the experiment box 
                klastRPE = k;
            
            
                #What is the probability of reward? We use the Hazard rate
                Prew =  self.belief.Hazard[self.State,  utilitycount]
                ShannonH = tools.ShannonH(Prew);
                
                ActualReward = ((Experiment.TrialCount + 1)%FR[workstate] == 0);
                
                RPE = Experiment.Reward[workstate]*(ActualReward - Prew);
                
                
                
                
                if ActualReward:
                    nu_phasic = nu_tonic + self.Firing.nu_burst;
                    dt_phasic = RPE*self.Firing.spburst/self.Firing.nu_burst
                    #print "burst duration" , dt_phasic
                    self.Results.rewards.append([k, Experiment.Reward[workstate]])
                    Experiment.RewardEarned.append([k, workstate])
                    
                else:#note that RPE is negative here. 
                    nu_phasic = 0;#np.max( [nu_tonic_eff + a_RPE*RPE[pressnum], 0]) ;
                    dt_phasic = - self.alpha_RPE*RPE*self.Firing.dtpause
                
                RPEvec[Experiment.TrialCount] = RPE;
                Experiment.TrialCount += 1;  #increment local counts
                Experiment.TotalCount+= 1;   #increment global counts
                utilitycount = np.min([Experiment.TrialCount, Experiment.MaxPress - 2]);
                
                self.Results.RPE.append([k, RPE]);
                
            #After press we can decide if firing rate is phasic or tonic. 
 
            if (k - klastRPE + 1.0)*self.dt_small < (dt_phasic):
                nu_eff = nu_phasic
            else:
                nu_eff = nu_tonic
            
            #print nu_eff 
            self.Firing.nu_in = nu_eff;
            self.UpdateDA()    
            
           
            if (k%DT < 1) & (k > timeout):
                D2 = self.Terminal.DA/(self.D2affinity + self.Terminal.DA)
                Pstay = D2**(self.dt_large);
                #print "Uti" , self.belief.Utility[:, utilitycount]
                NewState = tools.NewState2(self.State, Pstay,  self.belief.Utility[:, utilitycount])
                #dropout parameter: Did we stop working? 
                dropout = (NewState != self.State) & bool(self.State)
                #print NewState, self.State, dropout
                if (NewState != self.State):
                    self.Results.State.append([k,[NewState]])
            
                
                
            if dropout | ActualReward:
            
               
                RPE_PRESS = np.sum(RPEvec)
                RPE_LOSS =  (ActualReward==False)*Experiment.TrialCount - self.belief.Eloss[self.State];
                RPE_SURV = ActualReward - self.belief.Psurv[self.State];
                
                #print 'dropout', dropout, 'RPE_press', RPE_PRESS
                
                self.belief.Epress[self.State] += - self.learnrate_press*RPE_PRESS
                self.belief.Epress[self.State] = np.max([self.belief.Epress[self.State], 1])
                
                self.belief.Eloss[self.State]  += self.learnrate_loss*RPE_LOSS;# Use tools.Eloss(ExpectedMeanReward[State[t-1]], Phere)
                self.belief.Eloss[self.State] = np.max([self.belief.Eloss[self.State], 0])
                
                self.belief.Psurv[self.State]  += self.learnrate_psurv*RPE_SURV; 
                self.belief.Psurv[self.State] = np.max([self.belief.Psurv[self.State], 0.001])
                self.belief.Psurv[self.State] = np.min([self.belief.Psurv[self.State], 1])
                
                #ETotalLoss = (1.0/self.belief.Psurv[self.State]  - 1)*self.belief.Eloss[self.State]
                
                print("Progress: ", round((1.0*k)/steps, 1))
                #print "Eloss est. = " , np.round(ExpectedMeanLoss, 2), "Total_loss. = ",  np.round(ExpectedTotalLoss, 2) 
                #print  ExpectedMeanReward, RPE_State
                self.belief.Hazard[self.State] = tools.HazardRate(self.belief.Epress[self.State] , self.belief.Spress, Experiment.MaxPress)
                #print np.round(Tplot[t]), Reward,    Value
                
                self.belief.Utility[self.State] = tools.Utility_full2( Experiment.Reward[workstate], self.belief.Epress[self.State] , self.belief.Spress, Experiment.MaxPress,  self.belief.Psurv[self.State], self.belief.Eloss[self.State])
            
        
                #reset counters
                if dropout:
                    timeout = k + self.pause    
                dropout = False;
                ActualReward = False;
                ShannonH = 0;
                Experiment.TrialCount = 0;
                utilitycount = 0;
                RPEvec = np.zeros(Experiment.MaxPress)
                
                
                self.Results.Epress.append([k, list(self.belief.Epress[1:])])
                self.Results.Eloss.append([k, list(self.belief.Eloss[1:])])
                self.Results.Psurv.append([k, list(self.belief.Psurv[1:])])
                self.Results.Leverutility.append([k, list(self.belief.Utility[1:, 0])])

                
            workstate = NewState - 1;    
            self.State = NewState  
            self.Results.terminalDA[k] = self.Terminal.DA;
            self.Results.sdDA[k] = self.Soma.DA;
            self.Results.NUeff[k] = self.Firing.nu_now;
            FR = Experiment.GetNextCrit(FR, k*self.dt_small)
            
           
            k += 1;
    
            
   
        
    def SetupBelief(self, Experiment):
        Workstates = len(Experiment.crit)
        #steps = int(np.ceil(Experiment.Tmax/self.dt_small));
        self.belief.Epress = np.insert(Experiment.initGuess, 0, np.inf);
        self.belief.Reward = np.insert(Experiment.Reward, 0, 0);
        self.belief.Eloss  = np.zeros(Workstates + 1);
        self.belief.Psurv  = np.ones(Workstates + 1);
        self.belief.Hazard  = np.zeros([Workstates+1, Experiment.MaxPress]);
        self.belief.Utility = np.ones([Workstates+1, Experiment.MaxPress]);
        for k in range(1, Workstates + 1):
            self.belief.Hazard[k,:] = tools.HazardRate(self.belief.Epress[k], self.belief.Spress, Experiment.MaxPress)
            self.belief.Utility[k, :] = tools.Utility_full2(self.belief.Reward[k], self.belief.Epress[k], self.belief.Spress, Experiment.MaxPress ,  self.belief.Psurv[k], self.belief.Eloss[k])
        
    def SetupResults(self, steps):
        self.Results.terminalDA = np.zeros(steps);
        self.Results.sdDA = np.zeros(steps);
        self.Results.NUeff = np.zeros(steps);
        self.Results.time = np.linspace(0,steps*self.dt_small, steps)
        #self.Results.State = np.zeros(steps);
        self.Results.Epress = [];
        self.Results.Eloss= [];
        self.Results.Psurv= [];
        self.Results.Leverutility = [];
        
        self.Results.Epress.append([0, list(self.belief.Epress[1:])])
        self.Results.Eloss.append([0, list(self.belief.Eloss[1:])])
        self.Results.Psurv.append([0, list(self.belief.Psurv[1:])])
        self.Results.Leverutility.append([0, list(self.belief.Reward[1:]/self.belief.Epress[1:])])
        self.Results.State.append([0, [self.State]])
                
    def CleanUpResults(self):
        def plotify(R,N ):
            L = len(R)
            n = len(R[0][1]);
            R0 = list(R)
            R0.append([N,0])
            r0 = np.zeros([n, N])
            
            for k in range(0, L):
                indx1 = R0[k][0];
                indx2 = R0[k+1][0];
                for l in range(0, n):
                    r0[l][indx1:indx2] = R0[k][1][l];
            return r0
            
        #def GetWork(st, press_time):
        #    L = len(st)
        #   
        #    for k in range(0, L):
        #        indx1 = st[k][0];
        #        indx2 = st[k+1][0];
        #        while press_time[
        #      
        #    return r0
            
        n = len(self.Results.Epress[0][1]);
        m = len(self.Results.time);
      
        self.Results.P_State  = plotify(self.Results.State, m)[0]
        self.Results.P_Epress = plotify(self.Results.Epress, m)
        self.Results.P_Eloss  = plotify(self.Results.Eloss,  m)
        self.Results.P_Psurv  = plotify(self.Results.Psurv,  m)
        self.Results.P_Totalloss = np.zeros([n,m])
        self.Results.P_cumReward = np.zeros(m)
        self.Results.P_cumPress  = np.zeros(m)
        Rcount = 0;
        Rcum = 0;
        for k1 in range(0, n):      
            self.Results.P_Totalloss[k1] = (1.0/self.Results.P_Psurv[k1]  - 1)*self.Results.P_Eloss[k1]
        for k1 in range(0,m):
            if (k1 >= self.Results.rewards[Rcount][0]) & (Rcount < len(self.Results.rewards)-1):
                Rcum   += self.Results.rewards[Rcount][1]
                Rcount += 1;
            self.Results.P_cumReward[k1] = Rcum;
        Rcount += 1;
        Stepcum = 0;
        for k1 in range(m):
            if (k1 >= self.Results.RPE[Stepcum][0]) & (Stepcum < len(self.Results.RPE)-1):
                Stepcum   += 1       
            self.Results.P_cumPress[k1] = Stepcum; 
            
        
            
       
        self.Results.P_Leverutility =plotify(    self.Results.Leverutility, m)
        
        
    def DisplayResults(self, Experiment, tmin = 0):
        
        temp = np.array(Experiment.RewardEarned)
        temp2  = list(temp[:,1])

        NStates = len(Experiment.crit)
        Rst = list(self.Results.P_State[Experiment.PressTime])
        
        TransCount = np.zeros(NStates)

        for k in range(0,NStates):
            print('Work-state ' ,k + 1,  ' Rewards: ' , temp2.count(k), ', Total gain: ' , temp2.count(k)*Experiment.Reward[k], ', Total effort: ' , Rst.count(k+1), '. Rat: ' , temp2.count(k)*Experiment.Reward[k]/Rst.count(k+1))
        for k in range(1, NStates+1):
            TransCount = 0;
            for k2 in range(len(self.Results.State)):
                if self.Results.State[k2][1] == [k]:
                    TransCount += 1;
            print("Transisitions into State ", k, " = " , TransCount)
                
                  
        

    def PrintCurrent(self):
        print("Epress: ", self.belief.Epress)
        print( "Eloss: ",  self.belief.Eloss)
        print("Psurv: ",  self.belief.Psurv)
        
        #print "Hazard Rate: " , self.belief.Hazard
        #print "Lever Util: " , self.belief.Utility[:,1]
         
         
    def loaddata(self , filename):
        t3 = self.DAsystem.Vmax
        
        print( "NOT Implemented yet!, loading data from file = " , filename, t3)
        
    def savedata(self , filename): 
        "this function saves the data. " 
        def save_the_piece( wd):   
            "This is a subroutine that saves a single file of data." 
            if (hasattr(self.Results, wd)):
                
                indx = filename.find('.txt')
                FN2 =  filename[:indx] + '_' + wd + filename[indx:]
        
                np.savetxt(FN2, getattr(self.Results, wd))
                print( "saving " + wd + " to file = " , FN2)
            else:
                print('did not find ', wd)
                
        save_the_piece('P_State')
        save_the_piece('P_Epress')
        save_the_piece('P_Eloss')
        save_the_piece('P_Psurv')
        save_the_piece('P_Totalloss')
        save_the_piece('P_cumReward')
        save_the_piece('P_cumPress')
        save_the_piece('P_Leverutility')
        save_the_piece('time')
        save_the_piece('terminalDA')
        save_the_piece('RPE')
        save_the_piece('rewards')
    
 
        
 ###################################################################################
 
#
M1 = Mouse();
#fir
M1.Firing.nu_tonic_baseline = 4;
#E = Experiment(1000, [10, 40], [100, 0], [10, 10], [500])
##E = Experiment(5000, [15, 30, 60, 120], [200, 0, 0, 0], [15, 40, 2, 2], [500, 1000, 2000])
E = Experiment(1500, [10], [100] )
#
M1.RunSchedule(E)


M1.CleanUpResults()
#M1.savedata('LowDA_FR20.txt')

NStates = len(E.crit)
plt.close('all')

plt.figure(1); plt.plot(M1.Results.time, M1.Results.NUeff);plt.show()
plt.figure(2); plt.plot(M1.Results.time, M1.Results.terminalDA);plt.show()
plt.figure(3); plt.plot(M1.Results.time, M1.Results.P_State + 1); plt.ylim(0.9, NStates +1.1);plt.show()



plt.figure(30); 
for k in range(0, len(E.crit)): 
    plt.plot(M1.Results.time, M1.Results.P_Epress[k]);
plt.title('E press')
plt.show()

plt.figure(31); 
for k in range(0, len(E.crit)): 
    plt.plot(M1.Results.time, M1.Results.P_Psurv[k]);
plt.title('Survival probability')
plt.show()

plt.figure(32); 
for k in range(0, len(E.crit)): 
    plt.plot(M1.Results.time, M1.Results.P_Eloss[k]);
plt.title('Loss pr trial')

plt.show()

plt.figure(33); 
for k in range(0, len(E.crit)): 
    plt.plot(M1.Results.time, M1.Results.P_Totalloss[k]);
plt.title('Total loss pr reward')

plt.show()


plt.figure(34); 
for k in range(0, len(E.crit)): 
    plt.plot(M1.Results.time, M1.Results.P_Leverutility[k]);
plt.title('Utility of lever')

plt.show()


plt.figure(35); 
plt.plot(M1.Results.time, M1.Results.P_cumReward)
plt.title('Cumulative Reward')
plt.show()

plt.figure(136); 
plt.plot(M1.Results.time, M1.Results.P_cumPress)

plt.title('Cumulative Presses')


M1.DisplayResults(E)
