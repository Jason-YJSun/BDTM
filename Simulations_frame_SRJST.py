# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 20:20:43 2022

@author: Jay Sun
"""

import utils
import numpy as np
import pandas as pd
#import time

#begin = time.time()
#%%control limit searching
reviews = utils.data_reader1()

theta0, lda0 = utils.theta_under_lda(reviews)

#step(1)based on simulation step = 0.2
Lmin = 0
Lmax = utils.SRJST_chart((0.55, 0.33, 0.12), (0.33, 0.33, 0.34))
#LmaxB = utils.BDTM_chart((0.35, 0.33, 0.32), (0.33, 0.33, 0.34))
CL_initial = (Lmax + Lmin)/2
#LB = (LmaxB + Lmin)/2

wordnum = utils.doc_word_number(1000, 370)

def simulation(CL):    
    snum = 10
    sim_Qt1 = np.zeros(shape=(370, snum))
    for sim in range(snum):
#step(2)    
        simulated_theta = utils.simulation_corpus_generation(wordnum, theta0, 370)

        sQt_SK = np.zeros(370)

        #sQt_BW = np.zeros(370)
#step(3)
        for t in range(len(sQt_SK)):
            sQt_SK[t] = utils.SRJST_chart(simulated_theta[t], theta0, wordnum[t])
            #sQt_BW[t] = utils.BDTM_chart(simulated_theta[t], theta0)
            
            sim_Qt1[:, sim] = sQt_SK
#step(4)    
    ARL = utils.ARL_count(snum, sim_Qt1, CL)
#print(ARL1,ARL2)
    return ARL

#end = time.time()
#time = end - begin
#print(time)

#step(5) 
def CL_update1(CL):#ARL<370
    global Lmin
    Lmin = CL
    L = (Lmax + Lmin)/2
    ARL = simulation(L)
    return ARL, L

def CL_update2(CL):#ARL>370
    global Lmax
    Lmax = CL
    L = (Lmax + Lmin)/2
    ARL = simulation(L)
    return ARL, L

ARL_initial = simulation(CL_initial)
ARL_upd = ARL_initial
CL_upd = CL_initial

while ARL_upd < 369:
      ARL_upd, CL_upd = CL_update1(CL_upd)
      print(ARL_upd, CL_upd)
while ARL_upd > 371:
      ARL_upd, CL_upd = CL_update2(CL_upd)
      print(ARL_upd, CL_upd)
else: 
      print('final ARL = ', ARL_upd)
      print('final CL = ', CL_upd)
        

#%%vibration simulating
final_CL =  38.06497779157908#0.0035828880803026007
shifts = utils.shift_intro(theta0, 0.02)

row = pd.DataFrame(shifts).shape[0]
col = pd.DataFrame(shifts).shape[1]

def simulation_OC(CL, shift_theta):    
    snum = 100
    sim_Qt1 = np.zeros(shape=(370, snum))
    for sim in range(snum):
#step(2)    
        simulated_theta = utils.simulation_corpus_generation(wordnum, shift_theta, lda0)

        sQt_SK = np.zeros(370)

        #sQt_BW = np.zeros(370)
#step(3)
        for t in range(len(sQt_SK)):
            sQt_SK[t] = utils.SRJST_chart(simulated_theta[t], shift_theta, wordnum[t])
            #sQt_BW[t] = utils.BDTM_chart(simulated_theta[t], theta0)
            
            sim_Qt1[:, sim] = sQt_SK
#step(4)    
    ARL = utils.ARL_count(snum, sim_Qt1, CL)
#print(ARL1,ARL2)
    return ARL
    
for i in range(row):
    sARL = simulation_OC(final_CL, shifts[i,:])
    print(sARL)

#print each line in for loop of all function