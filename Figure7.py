# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 20:25:33 2023

@author: Jay Sun
"""

#Figure 5

import csv
import matplotlib.pyplot as plt

with open('perplexity_lda.csv',encoding='gbk',errors='ignore') as total:
    reader = csv.reader(total)
    perplda = [row[1] for row in reader]

with open('perplexity_mdtm.csv',encoding='gbk',errors='ignore') as total:
    reader = csv.reader(total)
    perpmdtm = [row[1] for row in reader]
    
with open('perplexity_jr.csv',encoding='gbk',errors='ignore') as total:
    reader = csv.reader(total)
    perpjr = [row[1] for row in reader]
    
with open('perplexity_BDTM.csv',encoding='gbk',errors='ignore') as total:
    reader = csv.reader(total)
    perpbdtm = [row[1] for row in reader]
#%%
topic = []
plda = []
pmdtm = []
pjr = []
pbdtm = []
for i in range(1, 11):
    topic.append(i)
    plda.append(float(perplda[i]))
    pmdtm.append(float(perpmdtm[i]))
    pjr.append(float(perpjr[i]))
    pbdtm.append(float(perpbdtm[i]))

plt.plot(topic, plda, linewidth = 1, label = 'LDA', \
         linestyle = ':',color = 'y', marker = 's', markersize = 4)
plt.plot(topic, pmdtm, linewidth = 1, label = 'MDTM', \
         linestyle = '--',color = 'c', marker = 's', markersize = 4)
plt.plot(topic, pjr, linewidth = 1, label = 'JST-RMR', \
         linestyle = '-.',color = 'g', marker = 's', markersize = 4)
plt.plot(topic, pbdtm, linewidth = 1, label = 'BDTM', \
         linestyle = '-',color = 'b', marker = 's', markersize = 4)
plt.legend()
plt.xlabel("Number of Topic")
plt.ylabel("Perplexity")
plt.ylim(220, 410)
plt.show()
    