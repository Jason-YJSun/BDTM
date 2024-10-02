# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 20:28:55 2023

@author: Jay Sun
"""

#debug Figure 12
import csv
with open('thetair_bdtm_case2.csv',encoding='gbk',errors='ignore') as total:
    reader = csv.reader(total)
    t1 = [row[1] for row in reader]
with open('thetair_bdtm_case2.csv',encoding='gbk',errors='ignore') as total:
    reader = csv.reader(total)    
    t2 = [row[2] for row in reader]
with open('thetair_bdtm_case2.csv',encoding='gbk',errors='ignore') as total:
    reader = csv.reader(total)
    t3 = [row[3] for row in reader]

tir = []
for i in range(1, len(t1)):
    t = [float(t1[i]), float(t2[i]), float(t3[i])]
    tir.append(t)
#%%    
import csv
with open('thetaar_bdtm_case2.csv',encoding='gbk',errors='ignore') as total:
    reader = csv.reader(total)
    t1a = [row[1] for row in reader]
with open('thetaar_bdtm_case2.csv',encoding='gbk',errors='ignore') as total:
    reader = csv.reader(total)    
    t2a = [row[2] for row in reader]
with open('thetaar_bdtm_case2.csv',encoding='gbk',errors='ignore') as total:
    reader = csv.reader(total)
    t3a = [row[3] for row in reader]

tar = []
for i in range(1, len(t1a)):
    t = [float(t1a[i]), float(t2a[i]), float(t3a[i])]
    tar.append(t)
#%%
from class_BDTM_case2 import BDTM
bdtm = BDTM(3)
#%%
from scipy.stats import wasserstein_distance   
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
IC = [0.252, 0.423, 0.325]

BWall = np.zeros(len(tir))
for t in range(len(tir)):
    BWall[t] = sum(bdtm.nmk[t, :]) *\
        wasserstein_distance(IC, tir[t])
BWi = pd.DataFrame(BWall) 
BWie = BWi[0].ewm(alpha = 0.3).mean()
BWie_example = []
for e in range(200, 250):
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
             label = 'Control Limit = 3.9312')
plt.legend()
#%%
BWalla = np.zeros(len(tar))
for t in range(len(tar)):
    BWalla[t] = sum(bdtm.nmk_ar[t, :]) *\
        wasserstein_distance(IC, tar[t])
BWia = pd.DataFrame(BWalla) 
BWiea = BWia[0].ewm(alpha = 0.3).mean()
BWiea_example = []
for e in range(200, 250):
    BWiea_example.append(BWiea[e])
#%%    
CL_wa = np.percentile(BWiea_example, 90)
#%%
cl_listwa = np.zeros(len(BWiea_example))
for i in range(len(cl_listwa)):
    cl_listwa[i] = CL_wa
    
plt.xlabel('time')
plt.ylabel('Charting Statistics')
plt.plot(range(len(BWiea_example)), BWiea_example,'-o',color = 'k', \
         markersize=2, linewidth = 0.8)

plt.plot(range(len(cl_listwa)), cl_listwa, color = 'k',\
         dashes=(5, 3), linewidth = 0.8, linestyle = '--',\
             label = 'Control Limit = 1.0834')
plt.legend()