# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 14:30:52 2024

@author: Jay Sun
"""

#debug figure10
import csv
with open('thetair_bdtm.csv',encoding='gbk',errors='ignore') as total:
    reader1 = csv.reader(total)
    thetair1 = [row[1] for row in reader1]
with open('thetair_bdtm.csv',encoding='gbk',errors='ignore') as total:
    reader1 = csv.reader(total)
    thetair2 = [row[2] for row in reader1]
with open('thetair_bdtm.csv',encoding='gbk',errors='ignore') as total:
    reader1 = csv.reader(total)
    thetair3 = [row[3] for row in reader1]
    

with open('thetaar_bdtm.csv',encoding='gbk',errors='ignore') as total:
    reader2 = csv.reader(total)
    thetaar1 = [row[1] for row in reader2]
with open('thetaar_bdtm.csv',encoding='gbk',errors='ignore') as total:
    reader2 = csv.reader(total)
    thetaar2 = [row[2] for row in reader2]
with open('thetaar_bdtm.csv',encoding='gbk',errors='ignore') as total:
    reader2 = csv.reader(total)
    thetaar3 = [row[3] for row in reader2]

with open('sent_pi.csv',encoding='gbk',errors='ignore') as total:
    reader3 = csv.reader(total)
    pi1 = [row[1] for row in reader3]
with open('sent_pi.csv',encoding='gbk',errors='ignore') as total:
    reader3 = csv.reader(total)
    pi2 = [row[2] for row in reader3]
with open('sent_pi.csv',encoding='gbk',errors='ignore') as total:
    reader3 = csv.reader(total)
    pi3 = [row[3] for row in reader3]  
    
thetair = []
thetaar = []
pi = []

for i in range(1, len(pi1)):
    thetair.append((thetair1[i], thetair2[i], thetair3[i]))
    thetaar.append((thetaar1[i], thetaar2[i], thetaar3[i]))
    pi.append((pi1[i], pi2[i], pi3[i]))
#%%
from class_BDTM import BDTM
bdtm = BDTM(3)
#%%
#sentiment theta normalize
sent_theta = []
for i in range(len(thetair)):
    stheta = []
    for j in range(3):
        stheta.append(float(thetair[i][j]) * float(pi[i][j]))
    nstheta = []
    for st in stheta:
        nstheta.append(st / sum(stheta))
    sent_theta.append(nstheta)

#%%
from scipy.stats import wasserstein_distance   
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
IC = [0.252, 0.423, 0.325]

BWall = np.zeros(len(sent_theta))
for t in range(len(sent_theta)):
    BWall[t] = sum(bdtm.nmk[t, :]) *\
        wasserstein_distance(IC, sent_theta[t])
BWi = pd.DataFrame(BWall) 
BWie = BWi[0].ewm(alpha = 0.3).mean()
BWie_example = []
for e in range(400, 500):
    BWie_example.append(BWie[e])

CL_w = np.percentile(BWie_example, 90)
cl_listw = np.zeros(len(BWie_example))
for i in range(len(cl_listw)):
    cl_listw[i] = CL_w
    
plt.xlabel('time')
plt.ylabel('Charting Statistics')
plt.plot(range(len(BWie_example)), BWie_example,'-o',color = 'k', \
         markersize=2, linewidth = 0.8)

plt.plot(range(len(cl_listw)), cl_listw, color = 'k',\
         dashes=(5, 3), linewidth = 0.8, linestyle = '--',\
             label = 'Control Limit = 12.187')
plt.legend()
#%%
with open('sent_pi_ar.csv',encoding='gbk',errors='ignore') as total:
    reader3 = csv.reader(total)
    piar1 = [row[1] for row in reader3]
with open('sent_pi_ar.csv',encoding='gbk',errors='ignore') as total:
    reader3 = csv.reader(total)
    piar2 = [row[2] for row in reader3]
with open('sent_pi_ar.csv',encoding='gbk',errors='ignore') as total:
    reader3 = csv.reader(total)
    piar3 = [row[3] for row in reader3]  

piar = []
for i in range(1, len(pi1)):
    piar.append((piar1[i], piar2[i], piar3[i]))

#sentiment theta ar normalize
sent_thetaar = []
for i in range(len(thetair)):
    stheta = []
    for j in range(3):
        stheta.append(float(thetaar[i][j]) * float(piar[i][j]))
    nstheta = []
    for st in stheta:
        nstheta.append(st / sum(stheta))
    sent_thetaar.append(nstheta)


#%%
import pandas as pd
BWalla = np.zeros(len(sent_thetaar))
for t in range(len(thetaar)):
    BWalla[t] = sum(bdtm.nmk_ar[t, :]) *\
        wasserstein_distance(IC, sent_thetaar[t])
BWa = pd.DataFrame(BWalla) 
BWae = BWa[0].ewm(alpha = 0.3).mean()
BWae_example = []
for e in range(400, 500):
    BWae_example.append(BWae[e])

CL_wa = np.percentile(BWae_example, 90)
cl_listwa = np.zeros(len(BWae_example))
for i in range(len(cl_listwa)):
    cl_listwa[i] = CL_wa
 
plt.xlabel('time')
plt.ylabel('Charting Statistics')
plt.plot(range(len(BWae_example)), BWae_example,'-o',color = 'k', \
         markersize=2, linewidth = 0.8)

plt.plot(range(len(cl_listwa)), cl_listwa, color = 'k',\
         dashes=(5, 3), linewidth = 0.8, linestyle = '--',\
             label = 'Control Limit=9.5432')
plt.legend()