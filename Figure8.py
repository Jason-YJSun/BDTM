# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 10:38:38 2023

@author: Jay Sun
"""

#Figure 6

import csv
import matplotlib.pyplot as plt

with open('coherence_lda.csv',encoding='gbk',errors='ignore') as total:
    reader = csv.reader(total)
    clda = [row[1] for row in reader]

with open('coherence_mdtm.csv',encoding='gbk',errors='ignore') as total:
    reader = csv.reader(total)
    cmdtm = [row[1] for row in reader]
    
with open('coherence_jr.csv',encoding='gbk',errors='ignore') as total:
    reader = csv.reader(total)
    cjr = [row[1] for row in reader]
    
with open('coherence_bdtm.csv',encoding='gbk',errors='ignore') as total:
    reader = csv.reader(total)
    cbdtm = [row[1] for row in reader]
#%%
topic = []
lda = []
mdtm = []
jr = []
bdtm = []
for i in range(1, 11):
    topic.append(i)
    lda.append(abs(float(clda[i])))
    mdtm.append(abs(float(cmdtm[i])))
    jr.append(abs(float(cjr[i])))
    bdtm.append(abs(float(cbdtm[i])))

def normalize(original):
    norm = []
    for i in original:
        norm.append((i - min(original)) /\
                    (max(original) - min(original)))
    return norm

nlda = normalize(lda)
nmdtm = normalize(mdtm)
njr = normalize(jr)
nbdtm = normalize(bdtm)
#%%
plt.plot(topic, nlda, linewidth = 1, label = 'LDA', \
         linestyle = ':',color = 'y', marker = 's', markersize = 4)
plt.plot(topic, nmdtm, linewidth = 1, label = 'MDTM', \
         linestyle = '--',color = 'c', marker = 's', markersize = 4)
plt.plot(topic, njr, linewidth = 1, label = 'JST-RMR', \
         linestyle = '-.',color = 'g', marker = 's', markersize = 4)
plt.plot(topic, nbdtm, linewidth = 1, label = 'BDTM', \
         linestyle = '-',color = 'b', marker = 's', markersize = 4)
plt.legend(bbox_to_anchor = (1.25, 1.01), \
           loc = 'upper right', borderaxespad = 0.1)
plt.xlabel("Number of Topic")
plt.ylabel("Coherence")
#plt.ylim(220, 410)
plt.show()