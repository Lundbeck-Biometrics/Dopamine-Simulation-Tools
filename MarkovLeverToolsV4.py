# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import gamma
import scipy as scp

def NewState(ThisState, Pstay, Value_vec):
    def Ratios(thisstate, Valuevec):
        Nstates = len(Valuevec);
        Valuevec = np.double(Valuevec);
        ent = np.array(range(0,Nstates));
        Val_list = Valuevec[np.where(ent != thisstate)];
        Z = np.sum(Val_list);   
        return np.divide(Val_list,Z)
    
    Nstates = len(Value_vec);
    Push = np.random.rand(1)    
    Value_vec = np.array(Value_vec);    
    
    M = np.zeros(Nstates);
    M[ThisState] = Pstay;
    if ThisState == 0:
        M[1:Nstates] = (1-Pstay)*Ratios(ThisState, Value_vec);
    else:
        M[0] = 1-Pstay;
    Transvec = np.cumsum(M)
    
    #print "this state", ThisState, "transVec", Transvec, Value_vec, Push, Pstay
    #print "this state", ThisState, "transVec", Transvec
    NewState= np.int(np.min(np.where(Transvec > Push)))
    
    return NewState
    
def NewState2(ThisState, Pstay, Value):
    
    Rem = 1-Pstay;
    N = Rem/(np.sum(Value) - Value[ThisState]);
    
    Targ = np.zeros(len(Value))
    Targ[ThisState] = Pstay 
    
    
    for k in range(ThisState):
        Targ[k] = Value[k]*N;
    for k in range(ThisState+1, len(Value)):
        Targ[k] = Value[k]*N;
        
    Targ = np.cumsum(Targ);
    Push = np.random.rand(1)    
    #print Push, Targ, Value
    NewState= np.int(np.min(np.where(Targ > Push)))
    
    return NewState
    
    

def HazardRate_old(Expected_mean, Expected_std, MaxN, future = 1):  
    #future parameter tell how many steps in the future is the reward. 
    #Future = 1 is normal hazard function. Higher valus are used for generatlied utility. 
    MaxF = 0.9999; 
    Egam_k, Egam_theta = GetShape_n_Scale(Expected_mean, Expected_std)
    

   
    Epdf    = gamma.pdf( range(1,MaxN+2), Egam_k, 0, Egam_theta);
    
    Epdf = Epdf/np.sum(Epdf);
    Ecumpdf = np.cumsum(Epdf);
    #print Epdf[1:MaxN].size, Ecumpdf[0:(MaxN-1)].size
    SurvRate = Epdf[(future):(MaxN+1)]/(1 - Ecumpdf[0:(MaxN-future+1)]);
     #Cutting off near 0:
    remindx = np.where(Ecumpdf[0:(MaxN-future)] > MaxF)
    if remindx[0].size > 0:
        SurvRate[remindx] = SurvRate[remindx[0][0]]
       
    return SurvRate
    
def HazardRate(Expected_mean, Expected_std, MaxN, future = 1):  
    #future parameter tell how many steps in the future is the reward. 
    #Future = 1 is normal hazard function. Higher valus are used for generatlied utility. 
    
    Egam_k, Egam_theta, RealS = GetShape_n_Scale2(Expected_mean, Expected_std) 
    MaxSafe = min([int(Expected_mean + 8*RealS), MaxN - future + 1])
    SurvRate = np.zeros(MaxN - future + 1)
    SurvRate[0:MaxSafe] = HazardFormula(Egam_k, Egam_theta, np.array(range(1, MaxSafe+1)), future)
    SurvRate[MaxSafe:] = SurvRate[MaxSafe-1]
    return SurvRate
    
def HazardFormula( k, theta, N, future = 1):
    "k is shape, theta is scale" 
    N = np.array(N)
    x = (N + future - 1)/theta
    xden = N/theta;#np.divide(N,theta)
    #print scp.special.gamma(k),  scp.special.gammainc(k, n)*scp.special.gamma(k)
    H = theta**-1*x**(k-1)*np.exp(-x)/(scp.special.gamma(k) - scp.special.gammainc(k, xden)*scp.special.gamma(k))
    return H
    
def Hazard(Expected_mean, Expected_std, K, future = 1):  
    #future parameter tell how many steps in the future is the reward. 
    #Future = 1 is normal hazard function. Higher valus are used for generatlied utility. 
    MaxF = 0.999; 
 
    Egam_k, Egam_theta = GetShape_n_Scale(Expected_mean, Expected_std)    
    SurvRate = np.min([gamma.pdf(K+future-1, Egam_k, 0, Egam_theta)/(1-gamma.cdf(K,Egam_k, 0, Egam_theta)), MaxF])       
    return SurvRate

def GetFR(mPress, sPress):
    Egam_k, Egam_theta = GetShape_n_Scale(mPress, sPress)
    M = len(mPress);
    NewFR = np.zeros(M) ;
    print(Egam_k, Egam_theta)
    for indx in range(M):

        #print indx, NewFR
        NewFR[indx] =  np.round(gamma.rvs(Egam_k[indx],0, Egam_theta[indx]))
    
    return NewFR
        
def GetShape_n_Scale(M, S):
    M = np.array(M);    
    if S == -1:
        S = M*0.25;
    elif S == -2: 
        S = M;  
    theta = np.array(np.double(S)**2/np.double(M)); #Scale parameter
    k = np.array(M/theta);#Shape parameter
    
    return k, theta
    
def GetShape_n_Scale2(M, S):
    M = np.array(M);    
    if S == -1:
        S = M*0.25;
    elif S == -2: 
        S = M;  
    theta = np.array(np.double(S)**2/np.double(M)); #Scale parameter
    k = np.array(M/theta);#Shape parameter
    
    return k, theta, S


    
def ShannonH(Prew):
               
    if Prew == 0.0:
        H = 0;
    elif Prew == 1.0:
        H= 0;
    else:
        H = -(Prew*np.log(Prew) + (1-Prew)*np.log(1-Prew));

    return H
    
    

def Utility_full_old(reward, Ework, Swork, MaxN, alpha = 0.0):
    
     Ework = np.array(Ework)
     Egam_k, Egam_theta = GetShape_n_Scale(Ework, Swork)
     
     #Epdf    = gamma.pdf( range(0,MaxN), Egam_k, 0, Egam_theta);
     #
     #Ecumpdf = np.cumsum(Epdf);
     #if any(Ecumpdf > MaxF):
     #    
     #    cutindx = np.min(np.where(Ecumpdf > MaxF))
     #else:
     #    cutindx = MaxN - 1;
     #    
    
     #print Egam_k, Egam_theta
     Haz2D = np.array(np.zeros([MaxN, MaxN]))
     #U = np.zeros(MaxN)
     den = np.transpose(np.array(1.0/(np.array(range(1, MaxN +1 )) + alpha)))
     
     for k in range(0,MaxN):
         #indx1 = range(k+1, MaxN )
         indx = range(0, MaxN - k )
         Haz2D[k,indx] = HazardRate(Ework, Swork, MaxN, k + 1)
      
     
     Haz2D = np.transpose(Haz2D)
     
     U = reward*np.dot(Haz2D,den)#This is a matrix multiplication!!
     #taking care of round-off errors: 
    #
    # umax = U[cutindx];
    # 
    # 
    # #print cutindx, umax
    # #print reward, Ework, Swork, MaxN
    # #umax = np.min([umax, reward])
    # U[cutindx:MaxN] = umax
     return U

def Utility_full2(reward, Ework, Swork, MaxN, Psurv, EL):
     Ework = np.array(Ework)
     Egam_k, Egam_theta = GetShape_n_Scale(Ework, Swork)
     
     #Epdf    = gamma.pdf( range(0,MaxN), Egam_k, 0, Egam_theta);
     #
     #Ecumpdf = np.cumsum(Epdf);
     #if any(Ecumpdf > MaxF):
     #    
     #    cutindx = np.min(np.where(Ecumpdf > MaxF))
     #else:
     #    cutindx = MaxN - 1;
     #    
    
     #print Egam_k, Egam_theta
     Haz2D = np.array(np.zeros([MaxN, MaxN]))
     alpha = np.zeros(MaxN)
     
     #Totloss = (1 - Psurv )*EL;
     Totloss = (1/Psurv - 1)*EL;
     #print 'EL=', EL, 'Psurv:' , Psurv, 'TL:', Totloss
     G = Totloss**(1.0/Ework)
     alpha = G**(range(MaxN)) - 1
     for k in range(0,MaxN):
         indx = range(0, MaxN - k)
         Haz2D[k,indx] = HazardRate(Ework, Swork, MaxN, k+1)
      
     den = np.transpose(np.array(1.0/(np.array(range(1, MaxN +1 )) + alpha)))
     Haz2D = np.transpose(Haz2D)
     
     U = reward*np.dot(Haz2D,den)
     
     #This is a matrix multiplication!!
     
     #taking care of round-off errors: 
    
     #umax = U[cutindx];
     #
     #
     ##print cutindx, umax
     ##print reward, Ework, Swork, MaxN
     ##umax = np.min([umax, reward])
     #U[cutindx:MaxN] = umax
     return U

     
    
#def Eloss(Epress, Psurv):
#    #Her udregnes estimated af tab
#    #formel: r + 2r^2 + 3r^3 +...+(n-1)r^(n-1) = ... osv
#    N = int(np.round(Epress));
#    #print N, Psurv
#   
#    loss = 0;
#    for k in range(1, N):
#        p0 = (1-Psurv)*k*Psurv**(k-1)
#        loss += p0
#    return loss

def Eloss(N, Psurv):
    #Her udregnes estimated af tab
    #formel: r + 2r^2 + 3r^3 +...+(n-1)r^(n-1) = ... osv
    #N = int(np.round(Epress));
    loss = (1 - N*Psurv**(N-1) + (N-1)*Psurv**N)/(1 - Psurv)
    
    return loss
    

  
        
        
    
    
    
    
    