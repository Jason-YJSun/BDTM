# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 16:34:26 2023

@author: Jay Sun
"""

#debug JSTRMR monitoring 

from class_MDTM import MDTM

mdtm = MDTM(3, 4, 4)
#%
mdtm.fit()
#%
thetamdtm = mdtm.get_thetalist()
#%
theta1row = []
for e in thetamdtm:
    for d in e:
        theta1row.append(d)
#%%
from MDTM_ar import MDTMar     
mdtmar = MDTMar(3, 4, 4)
mdtmar.fit()
thetaar = mdtmar.get_thetalist()
theta1rowar = []
for e in thetaar:
    for d in e:
        theta1rowar.append(d)


#%%
import numpy as np
import pandas as pd
IC = [0.252, 0.423, 0.325]

nmk = mdtm.Ntd
Nmk = []
for e in nmk:
    for n in e:
        Nmk.append(n)
#%%
from scipy.stats import entropy  
import matplotlib.pyplot as plt

MKall = np.zeros(len(theta1row))
for t in range(len(theta1row)):
    MKall[t] = Nmk[t] * entropy(IC, theta1row[t])
MKi = pd.DataFrame(MKall) 
MKie = MKi[0].ewm(alpha = 0.3).mean()
MKie_example = []
for e in range(400, 500):
    MKie_example.append(MKie[e])

CL_k = np.percentile(MKie_example, 90)
cl_listk = np.zeros(len(MKie_example))
for i in range(len(cl_listk)):
    cl_listk[i] = CL_k
plt.xlabel('time')
plt.ylabel('Charting Statistics')
plt.plot(range(len(MKie_example)), MKie_example,'-o',color = 'k', \
         markersize=2, linewidth = 0.8)

plt.plot(range(len(cl_listk)), cl_listk, color = 'k',\
         dashes=(5, 3), linewidth = 0.8, linestyle = '--',\
             label = 'Control Limit = 4.4803')
plt.legend()    
#%%

nmka = mdtmar.Ntd
Nmka = []
for e in nmka:
    for n in e:
        Nmka.append(n)
        
MKalla = np.zeros(len(theta1rowar))
for t in range(len(theta1rowar)):
    MKalla[t] = Nmka[t] * entropy(IC, theta1rowar[t])
MKia = pd.DataFrame(MKalla) 
MKiea = MKia[0].ewm(alpha = 0.3).mean()
MKiea_example = []
for e in range(400, 500):
    MKiea_example.append(MKiea[e])

CL_ka = np.percentile(MKiea_example, 90)
cl_listka = np.zeros(len(MKiea_example))
for i in range(len(cl_listka)):
    cl_listka[i] = CL_ka
plt.xlabel('time')
plt.ylabel('Charting Statistics')
plt.plot(range(len(MKiea_example)), MKiea_example,'-o',color = 'k', \
         markersize=2, linewidth = 0.8)

plt.plot(range(len(cl_listka)), cl_listka, color = 'k',\
         dashes=(5, 3), linewidth = 0.8, linestyle = '--',\
             label = 'Control Limit = 3.5290')
plt.legend() 

#%%
import pandas as pd
savedi = pd.DataFrame(thetamdtm)
savedi.to_csv('thetair_mdtm.csv')
saveda = pd.DataFrame(thetaar)
saveda.to_csv('thetaar_mdtm.csv')  
