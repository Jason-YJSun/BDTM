# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 15:28:36 2022

@author: Jay Sun
"""

import numpy as np
import jieba
import jieba.posseg
import csv
import matplotlib.pyplot as plt
from gensim import corpora, models
from gensim.models import CoherenceModel, ldaseqmodel
from zhon.hanzi import punctuation as p


with open('review_JD_TM.csv',encoding='gbk',errors='ignore') as total:
    reader = csv.reader(total)
    initial_review = [row[3] for row in reader]
    
with open('review_JD_TM.csv',encoding='gbk',errors='ignore') as total:
    reader = csv.reader(total)
    additional_review = [row[5] for row in reader]
    
X={}
stop_words = open('stopword_1.txt','r',encoding='UTF-8') 
X=stop_words.read()
stop_words.close
X_list = X.split()


def remove_stopwords(ls):  # 去除停用词
    return [word for word in ls if word not in X]

words_list_ir = []
#for word in words_ir:
 #   if word  != ' ':
  #     words_list_ir.append(word)

synonyms=[]

def replace_synonyms(ls):  # 替换同义词
    return [synonyms[i] if i in synonyms else i for i in ls]

#%%
words_list_ir = []
for text in initial_review:
    words = replace_synonyms(remove_stopwords([w.word for w in jieba.posseg.cut(text)]))
    words_list_ir.append(words)
    
words_list_ar = []
for text in additional_review:
    words = replace_synonyms(remove_stopwords([w.word for w in jieba.posseg.cut(text)]))
    words_list_ar.append(words)
#%%
word_list_all = []
for i in range(len(words_list_ir)):
    word_list_all.append(words_list_ir[i])
    word_list_all.append(words_list_ar[i])
    

#%%
dictionary_all = corpora.Dictionary(word_list_all)

corpus_all = [dictionary_all.doc2bow(words) for words in word_list_all]
corpora.MmCorpus.serialize('corpus_all.mm', corpus_all)

#%%
lda3 = models.ldamodel.LdaModel(corpus=corpus_all, id2word=dictionary_all, num_topics=3)
for t in lda3.print_topics(num_words=20):
    print(t)