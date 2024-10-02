# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 14:50:27 2024

@author: Jay Sun
"""

#case2 JSTRMR ar

from class_JSTRMR_case2_ar import JST_RMR
import pandas as pd
jr = JST_RMR(3, 2, 5)
#%
psilzw, mulr, thetadlz, pidl, pidlw, pidlr = jr.get_psi()

IC_theta = [0.252, 0.423, 0.325]
IC_pi = [0.3306, 0.6694]


savedt = pd.DataFrame(thetadlz[:, 1, :])
savedt.to_csv('thetadlz_jstrmr_ar.csv')
saveds = pd.DataFrame(pidl)
saveds.to_csv('pidlz_jstrmr_ar.csv')

#%%
from scipy.stats import entropy
#thetadz = [thetadlz[:, 1, :]]
nd = jr.nd
jrtopic = []
for i in range(len(thetadlz)):
    sumfunc = 0
    for l in range(2):
        sumfunc += pidl[i, l] * entropy(IC_theta, thetadlz[i, l, :])
    jrtopic.append(2 * nd[i] * sumfunc)
#%%
md = jr.md
jrsent = []
for i in range(len(pidl)):
    jrsent.append(2 * nd[i] * md[i] *entropy(IC_pi, pidl[i]))
#%%    
jrrate = []
for i in range(len(pidlr)):
    jrrate.append(2 * md[i] *entropy(IC_pi, pidlr[i]))
#%%    
import numpy as np
import matplotlib.pyplot as plt

jti = pd.DataFrame(jrtopic) 
jtie = jti[0].ewm(alpha = 0.3).mean()
jtie_example = []
for e in range(200, 250):
    jtie_example.append(jtie[e])

CL_t = np.percentile(jtie_example, 90)
cl_listt = np.zeros(len(jtie_example))
for i in range(len(cl_listt)):
    cl_listt[i] = CL_t
    
plt.xlabel('time')
plt.ylabel('Charting Statistics')
plt.plot(range(len(jtie_example)), jtie_example,'-o',color = 'k', \
         markersize=2, linewidth = 0.8)

plt.plot(range(len(cl_listt)), cl_listt, color = 'k',\
         dashes=(5, 3), linewidth = 0.8, linestyle = '--',\
             label = 'Control Limit = 6.9114')
plt.legend()

#%%    
import numpy as np
import matplotlib.pyplot as plt

jsi = pd.DataFrame(jrsent) 
jsie = jsi[0].ewm(alpha = 0.3).mean()
jsie_example = []
for e in range(200, 250):
    jsie_example.append(jsie[e])

CL_ts = np.percentile(jsie_example, 90)
cl_lists = np.zeros(len(jsie_example))
for i in range(len(cl_lists)):
    cl_lists[i] = CL_ts
    
plt.xlabel('time')
plt.ylabel('Charting Statistics')
plt.plot(range(len(jsie_example)), jsie_example,'-o',color = 'k', \
         markersize=2, linewidth = 0.8)

plt.plot(range(len(cl_lists)), cl_lists, color = 'k',\
         dashes=(5, 3), linewidth = 0.8, linestyle = '--',\
             label = 'Control Limit = 8.2049')
plt.legend()

