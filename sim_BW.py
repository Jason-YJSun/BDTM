# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 20:45:33 2023

@author: Jay Sun
"""

#simulation supplement

from scipy import stats
import numpy as np
from scipy.stats import wasserstein_distance, entropy
import pandas as pd
#import pickle
wordnum = stats.poisson.rvs(mu=50, size=1200)#review set

#IC_rew
IC_theta = [0.385, 0.132, 0.483]

s1 = [0.390, 0.132, 0.478]
s2 = [0.395, 0.132, 0.473]
s3 = [0.400, 0.132, 0.468]
shift_theta = [s1, s2, s3]  

#%%
#file = open('psi_com.data', 'rb')
#IC_psi_com = pickle.load(file)
#%%simulation corpus
#step 2 
def sim_init():
    topic_assign = []
    for w in wordnum:#for each review
        topic_assign.append(np.random.multinomial(w, IC_theta))
        

#%theta of simulation corpus
    theta_s = []
    for d in topic_assign:
        theta_si = []
        for i in d:
            theta_si.append(i / sum(d))
        theta_s.append(theta_si)


    Qtall = []
    for t in theta_s:
        wd = wasserstein_distance(t, IC_theta)
        Qt = 2 * 3 * wordnum[theta_s.index(t)] * wd
        Qtall.append(Qt)
#step 1, L0 = (Lmin + Lmax)/2
    CL = (np.max(Qtall) + np.min(Qtall)) / 2
    return Qtall, CL

def sim_upd(simnum, CL):
    RL = []
    for s in range(simnum):
        topic_assign = []
        for w in wordnum:#for each review
            topic_assign.append(np.random.multinomial(w, IC_theta))
#%theta of simulation corpus
        theta_s = []
        for d in topic_assign:
            theta_si = []
            for i in d:
                theta_si.append(i / sum(d))
            theta_s.append(theta_si)

        Qtall = []
        for t in theta_s:
            wd = wasserstein_distance(t, IC_theta)
            Qt = wordnum[theta_s.index(t)] * wd
            Qtall.append(Qt)
        
        for t in Qtall:
            if t > CL:
                RL.append(Qtall.index(t))
    ARL = np.mean(RL)
    
    return ARL
#update CL

def CL_upd(Q, ARL, CL, simnum):
    CLupd = CL
    while ARL > 370.4:
        CLupd = (CLupd + np.min(Q)) / 2
        ARLupd = sim_upd(simnum, CLupd)
        print('updated CL:', ARLupd)
        if ARLupd < 370.4 and ARLupd > 369.6:
            break
        while ARL < 369.6:
            CLupd = (CLupd + np.max(Q)) / 2
            ARLupd = sim_upd(simnum, CLupd)
            print('updated CL:', ARLupd)
            if ARLupd < 370.4 and ARLupd > 369.6:
                break
    while ARL < 369.6:
        CLupd = (CLupd + np.max(Q)) / 2
        ARLupd = sim_upd(simnum, CLupd)
        print('updated CL:', ARLupd)
        if ARLupd < 370.4 and ARLupd > 369.6:
            break        
        while ARL > 370.4:
            CLupd = (CLupd + np.min(Q)) / 2
            ARLupd = sim_upd(simnum, CLupd)
            print('updated CL:', ARLupd)
            if ARLupd < 370.4 and ARLupd > 369.6:
                break
    return CLupd

#step 1 
Qt, L0 = sim_init()
#step 2 & 3 & 4
ARL0 = sim_upd(10, L0)
#step 5
final_CL = CL_upd(Qt, ARL0, L0, 10)

    
#%
shift_kl = []
for i in shift_theta:
    shift_kl.append(entropy(i, IC_theta))

shift_kl.sort()
#%%
def OC_sim(simCL, simnum, simtheta, simARL):
    RL = []
    for s in range(simnum):
        topic_assign = []
        wordnum = stats.poisson.rvs(mu = 50, size = simARL)
        for w in wordnum:#for each review
            topic_assign.append(np.random.multinomial(w, simtheta))
#%theta of simulation corpus
        theta_s = []
        for d in topic_assign:
            theta_si = []
            for i in d:
                theta_si.append(i / sum(d))
            theta_s.append(theta_si)

        Qtall = []
        for t in theta_s:
            wd = wasserstein_distance(t, IC_theta)
            Qt = 2 * 3 * wordnum[theta_s.index(t)] * wd
            Qtall.append(Qt)
        
        for t in Qtall:
            if t > simCL:
                RL.append(Qtall.index(t))
    ARL = np.nanmean(RL)
     
    return ARL

#%
try:
    ARL_all = []
    ARL_n = 370
    for s in shift_theta:
        ARL_ts = []
        for iteration in range(10):
            ARL_s = OC_sim(final_CL, 100, s, ARL_n)
            ARL_ts.append(ARL_s)
#    A = [a for a in ARL_ts if a != 'nan']
        ARL_all.append(np.nanmean(ARL_ts))
        ARL_n -= 30 
#        ARL_n = int(np.nanmean(ARL_ts)) 
except:
    saved_arl = pd.DataFrame(ARL_all)
    saved_arl.to_csv('saved_arl_sup.csv')
    
#%%
KL = []
for t in shift_theta:
    KL.append(stats.entropy(t, IC_theta))
saved_kl = pd.DataFrame(KL)
saved_kl.to_csv('saved_kl_sup.csv')