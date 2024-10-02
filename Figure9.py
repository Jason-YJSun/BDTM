# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 22:13:59 2023

@author: Jay Sun
"""

#debug overlap
from class_BDTM import BDTM

bdtm = BDTM(5)
bdtm.fitIR()
tw_b = bdtm.get_toic_words(bdtm.dictionary, bdtm.nkt, 10, 5)
#%
from class_LDA import LDA
lda = LDA(5)
tw_l = lda.get_top_words_id(10, 5)

#%
from class_JSTRMR import JST_RMR
jr = JST_RMR(5, 2, 5)
jr.fit()
tw_j = jr.get_toic_words(jr.nlzw[:, 0, :], 10, 5)

#%
from class_MDTM import MDTM
mdtm = MDTM(5, 4, 4)
mdtm.fit()
#%
tw_m = mdtm.get_toic_words(mdtm.Ntzw[1, :, :], 10, 5)
#%
from itertools import combinations
#find combination
def overlaps(top_words_id):
    freq_num = list(combinations\
                (range(len(top_words_id)), 2))
    #%overlap scores
    freqlist = []
    overlap_score = []
    for freqency in freq_num:
        set1 = set(top_words_id[freqency[0]])
        set2 = set(top_words_id[freqency[1]])
        freqlist.append(len(list(set1 & set2)))
        overlap_score.append(len(list(set1 & set2)) /\
                         len(freq_num))
    return overlap_score


ol_b = overlaps(tw_b)
ol_j = overlaps(tw_j)
ol_l = overlaps(tw_l)
ol_m = overlaps(tw_m)


#%%plot
import seaborn as sns
import matplotlib.pyplot as plt
sns.kdeplot(ol_b, label='BDTM', linewidth = 1, linestyle = '-', 
            color = 'b')
sns.kdeplot(ol_j, label='JST-RMR', linewidth = 1, linestyle = '-.', 
            color = 'g')
sns.kdeplot(ol_l, label='LDA', linewidth = 1, linestyle = ':', 
            color = 'y')
sns.kdeplot(ol_m, label='MDTM',  linewidth = 1, linestyle = '--', 
            color = 'c')
#print(tp_over)
plt.legend()
plt.xlabel('Topic overlap')
plt.ylabel('Frequency')
plt.show()



        