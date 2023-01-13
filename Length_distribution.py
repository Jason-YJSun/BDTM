# -*- coding: utf-8 -*-
"""
Created on Wed May 11 09:52:04 2022

@author: Jay Sun
"""

#top words of topics under LDA, DTM, MDTM
import numpy as np
import jieba
import jieba.posseg
import csv
import matplotlib.pyplot as plt
from gensim import corpora, models
from gensim.models import CoherenceModel, ldaseqmodel
from zhon.hanzi import punctuation as p



#%%reading
with open('review_JD_TM.csv',encoding='gbk',errors='ignore') as total:
    reader = csv.reader(total)
    initial_review = [row[3] for row in reader]
    
with open('review_JD_TM.csv',encoding='gbk',errors='ignore') as total:
    reader = csv.reader(total)
    additional_review = [row[5] for row in reader]

#%%length plot
length_IR = np.zeros(len(initial_review))
length_AR = np.zeros(len(additional_review))

for i in range(len(initial_review)):
    length_IR[i] = len(initial_review[i])
    length_AR[i] = len(additional_review[i])

#%%sorting
length_IR.sort()
length_AR.sort()

data = np.zeros(shape=(len(length_IR),2))
data[:,0] = length_IR
data[:,1] = length_AR

plt.hist(data, bins=200)
plt.xlabel('length of reviews')
plt.ylabel('number of reviews in specific length')
plt.legend(['initial review','additional review'])
plt.show()

#%%avr
total_avr = np.average(data)
print(total_avr)

with open('review_JD_TM.csv',encoding='gbk',errors='ignore') as total:
    reader = csv.reader(total)
    review_interval = [row[4] for row in reader]

avr_inter=[]
for i in range(len(review_interval)):
    avr_inter.append(int(review_interval[i]))

avr_inter.sort()

print(np.mean(avr_inter))

