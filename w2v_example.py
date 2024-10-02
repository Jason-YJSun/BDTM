# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 14:23:40 2023

@author: Jay Sun
"""

#word2vec example

from gensim.models import Word2Vec


s1 = ['Apple','reveals','a', 'video',\
      'about', 'its', 'products']
s2 = ['A', 'trailer', 'of', 'the',\
      'new', 'iPhone', 'just', 'released']
s3 = ['the', 'apple', 'tastes', 'good']
document = [s1, s2, s3]

model = Word2Vec(document,vector_size = 3, min_count = 1)
 
#sims = model.wv.similarity('the', 'apple')
#%%vector of words
import numpy as np
vectori = model.wv['products']
vectorj = model.wv['iPhone']
dist = np.sqrt(np.sum(np.square(vectori - vectorj)))

#%%
vec = model.wv['good']