# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 11:21:52 2022

@author: Jay Sun
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 23 16:47:40 2022

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
dictionary_ir = corpora.Dictionary(words_list_ir) #['doc1', 'doc2',...]

dictionary_ar = corpora.Dictionary(words_list_ar)

dictionary_all = corpora.Dictionary(word_list_all)

corpus_ir = [dictionary_ir.doc2bow(words) for words in words_list_ir]
corpora.MmCorpus.serialize('corpus_i.mm', corpus_ir)

corpus_ar = [dictionary_ar.doc2bow(words) for words in words_list_ar]
corpora.MmCorpus.serialize('corpus_a.mm', corpus_ar)

corpus_all = [dictionary_all.doc2bow(words) for words in word_list_all]
corpora.MmCorpus.serialize('corpus_all.mm', corpus_all)

#%%
lda1 = models.ldamodel.LdaModel(corpus=corpus_ir, id2word=dictionary_ir, num_topics=5)

lda2 = models.ldamodel.LdaModel(corpus=corpus_ar, id2word=dictionary_ar, num_topics=5)

lda3 = models.ldamodel.LdaModel(corpus=corpus_all, id2word=dictionary_all, num_topics=5)

#%%
t_words_ir = []
t_words_ar = []
t_words_all = []
for topic_id in range(5):
    t_words_ir.append( lda1.get_topic_terms(topicid=topic_id))
    t_words_ar.append( lda2.get_topic_terms(topicid=topic_id))
    t_words_all.append( lda3.get_topic_terms(topicid=topic_id))

#%%
#%%get all top 10 words id of 5 topics
def get_id(top_word_str):
    top_words_id = []
    for i in top_word_str:
        for t in i:
            top_words_id.append(t[0])
    return top_words_id

top_words_ir = get_id(t_words_ir)

top_words_ar = get_id(t_words_ar)

top_words_all = get_id(t_words_all)

wordid_ary1_ir = []
wordid_ary2_ir = []
wordid_ary3_ir = []
wordid_ary4_ir = []
wordid_ary5_ir = []

wordid_ary1_ar = []
wordid_ary2_ar = []
wordid_ary3_ar = []
wordid_ary4_ar = []
wordid_ary5_ar = []

wordid_ary1_all = []
wordid_ary2_all = []
wordid_ary3_all = []
wordid_ary4_all = []
wordid_ary5_all = []

for x in range(10):
    wordid_ary1_ir.append(top_words_ir[x])
    wordid_ary2_ir.append(top_words_ir[x+10])
    wordid_ary3_ir.append(top_words_ir[x+20])
    wordid_ary4_ir.append(top_words_ir[x+30])
    wordid_ary5_ir.append(top_words_ir[x+40])
    
    wordid_ary1_ar.append(top_words_ar[x])
    wordid_ary2_ar.append(top_words_ar[x+10])
    wordid_ary3_ar.append(top_words_ar[x+20])
    wordid_ary4_ar.append(top_words_ar[x+30])
    wordid_ary5_ar.append(top_words_ar[x+40])
    
    wordid_ary1_all.append(top_words_all[x])
    wordid_ary2_all.append(top_words_all[x+10])
    wordid_ary3_all.append(top_words_all[x+20])
    wordid_ary4_all.append(top_words_all[x+30])
    wordid_ary5_all.append(top_words_all[x+40])         

#%%
#%%overlap IR

frequency_ir = []
topic_overlap_ir = []
words_num1_ir = 0
score1_ir = 0
for i in range(len(wordid_ary1_ir)):
    for j in range(len(wordid_ary2_ir)):
        if wordid_ary1_ir[i] == wordid_ary2_ir[j]:
            words_num1_ir += 1
score1_ir = words_num1_ir/10
frequency_ir.append(words_num1_ir)
topic_overlap_ir.append(score1_ir)

words_num2_ir = 0
score2_ir = 0
for i in range(len(wordid_ary1_ir)):
    for j in range(len(wordid_ary3_ir)):
        if wordid_ary1_ir[i] == wordid_ary3_ir[j]:
            words_num2_ir += 1
score2_ir = words_num2_ir/10            
frequency_ir.append(words_num2_ir)
topic_overlap_ir.append(score2_ir)

words_num3_ir = 0
score3_ir = 0
for i in range(len(wordid_ary1_ir)):
    for j in range(len(wordid_ary4_ir)):
        if wordid_ary1_ir[i] == wordid_ary4_ir[j]:
            words_num3_ir += 1 
score3_ir = words_num3_ir/10
frequency_ir.append(words_num3_ir)
topic_overlap_ir.append(score3_ir)
            
words_num4_ir = 0
score4_ir = 0            
for i in range(len(wordid_ary1_ir)):
    for j in range(len(wordid_ary5_ir)):
        if wordid_ary1_ir[i] == wordid_ary5_ir[j]:
            words_num4_ir += 1
score4_ir = words_num4_ir/10
frequency_ir.append(words_num4_ir)
topic_overlap_ir.append(score4_ir)

words_num5_ir = 0
score5_ir = 0
for i in range(len(wordid_ary2_ir)):
    for j in range(len(wordid_ary3_ir)):
        if wordid_ary2_ir[i] == wordid_ary3_ir[j]:
            words_num5_ir += 1
score5_ir = words_num5_ir/10
frequency_ir.append(words_num5_ir)
topic_overlap_ir.append(score5_ir)

words_num6_ir = 0
score6_ir = 0
for i in range(len(wordid_ary2_ir)):
    for j in range(len(wordid_ary4_ir)):
        if wordid_ary2_ir[i] == wordid_ary4_ir[j]:
            words_num6_ir += 1
score6_ir = words_num6_ir/10
frequency_ir.append(words_num6_ir)
topic_overlap_ir.append(score6_ir)

words_num7_ir = 0
score7_ir = 0
for i in range(len(wordid_ary2_ir)):
    for j in range(len(wordid_ary5_ir)):
        if wordid_ary2_ir[i] == wordid_ary5_ir[j]:
            words_num7_ir += 1
score7_ir = words_num7_ir/10
frequency_ir.append(words_num7_ir)
topic_overlap_ir.append(score7_ir)

words_num8_ir = 0
score8_ir = 0
for i in range(len(wordid_ary3_ir)):
    for j in range(len(wordid_ary4_ir)):
        if wordid_ary3_ir[i] == wordid_ary4_ir[j]:
            words_num8_ir += 1
score8_ir = words_num8_ir/10
frequency_ir.append(words_num8_ir)
topic_overlap_ir.append(score8_ir)

words_num9_ir = 0
score9_ir = 0
for i in range(len(wordid_ary3_ir)):
    for j in range(len(wordid_ary5_ir)):
        if wordid_ary3_ir[i] == wordid_ary5_ir[j]:
            words_num9_ir += 1
score9_ir = words_num9_ir/10
frequency_ir.append(words_num9_ir)
topic_overlap_ir.append(score9_ir)

words_num10_ir = 0
score10_ir = 0
for i in range(len(wordid_ary4_ir)):
    for j in range(len(wordid_ary5_ir)):
        if wordid_ary4_ir[i] == wordid_ary5_ir[j]:
            words_num10_ir += 1
score10_ir = words_num10_ir/10
frequency_ir.append(words_num10_ir)
topic_overlap_ir.append(score10_ir)
    
#frequency = overlap_num
#overlap_score = topic_overlap
#overlap_index = overlap_num/(comb(5,2)*5)

#print(overlap_index)

#%%overlap AR

frequency_ar = []
topic_overlap_ar = []
words_num1_ar = 0
score1_ar = 0
for i in range(len(wordid_ary1_ar)):
    for j in range(len(wordid_ary2_ar)):
        if wordid_ary1_ar[i] == wordid_ary2_ar[j]:
            words_num1_ar += 1
score1_ar = words_num1_ar/10
frequency_ar.append(words_num1_ar)
topic_overlap_ar.append(score1_ar)

words_num2_ar = 0
score2_ar = 0
for i in range(len(wordid_ary1_ar)):
    for j in range(len(wordid_ary3_ar)):
        if wordid_ary1_ar[i] == wordid_ary3_ar[j]:
            words_num2_ar += 1
score2_ar = words_num2_ar/10            
frequency_ar.append(words_num2_ar)
topic_overlap_ar.append(score2_ar)

words_num3_ar = 0
score3_ar = 0
for i in range(len(wordid_ary1_ar)):
    for j in range(len(wordid_ary4_ar)):
        if wordid_ary1_ar[i] == wordid_ary4_ar[j]:
            words_num3_ar += 1 
score3_ar = words_num3_ar/10
frequency_ar.append(words_num3_ar)
topic_overlap_ar.append(score3_ar)
            
words_num4_ar = 0
score4_ar = 0            
for i in range(len(wordid_ary1_ar)):
    for j in range(len(wordid_ary5_ar)):
        if wordid_ary1_ar[i] == wordid_ary5_ar[j]:
            words_num4_ar += 1
score4_ar = words_num4_ar/10
frequency_ar.append(words_num4_ar)
topic_overlap_ar.append(score4_ar)

words_num5_ar = 0
score5_ar = 0
for i in range(len(wordid_ary2_ar)):
    for j in range(len(wordid_ary3_ar)):
        if wordid_ary2_ar[i] == wordid_ary3_ar[j]:
            words_num5_ar += 1
score5_ar = words_num5_ar/10
frequency_ar.append(words_num5_ar)
topic_overlap_ar.append(score5_ar)

words_num6_ar = 0
score6_ar = 0
for i in range(len(wordid_ary2_ar)):
    for j in range(len(wordid_ary4_ar)):
        if wordid_ary2_ar[i] == wordid_ary4_ar[j]:
            words_num6_ar += 1
score6_ar = words_num6_ar/10
frequency_ar.append(words_num6_ar)
topic_overlap_ar.append(score6_ar)

words_num7_ar = 0
score7_ar = 0
for i in range(len(wordid_ary2_ar)):
    for j in range(len(wordid_ary5_ar)):
        if wordid_ary2_ir[i] == wordid_ary5_ar[j]:
            words_num7_ar += 1
score7_ar = words_num7_ar/10
frequency_ar.append(words_num7_ar)
topic_overlap_ar.append(score7_ar)

words_num8_ar = 0
score8_ar = 0
for i in range(len(wordid_ary3_ar)):
    for j in range(len(wordid_ary4_ar)):
        if wordid_ary3_ar[i] == wordid_ary4_ar[j]:
            words_num8_ar += 1
score8_ar = words_num8_ar/10
frequency_ar.append(words_num8_ar)
topic_overlap_ar.append(score8_ar)

words_num9_ar = 0
score9_ar = 0
for i in range(len(wordid_ary3_ar)):
    for j in range(len(wordid_ary5_ar)):
        if wordid_ary3_ar[i] == wordid_ary5_ar[j]:
            words_num9_ar += 1
score9_ar = words_num9_ar/10
frequency_ar.append(words_num9_ar)
topic_overlap_ar.append(score9_ar)

words_num10_ar = 0
score10_ar = 0
for i in range(len(wordid_ary4_ar)):
    for j in range(len(wordid_ary5_ar)):
        if wordid_ary4_ar[i] == wordid_ary5_ar[j]:
            words_num10_ar += 1
score10_ar = words_num10_ar/10
frequency_ar.append(words_num10_ar)
topic_overlap_ar.append(score10_ar)    

#%%overlap all

frequency_all = []
topic_overlap_all = []
words_num1_all = 0
score1_all = 0
for i in range(len(wordid_ary1_all)):
    for j in range(len(wordid_ary2_all)):
        if wordid_ary1_all[i] == wordid_ary2_all[j]:
            words_num1_all += 1
score1_all = words_num1_all/10
frequency_all.append(words_num1_all)
topic_overlap_all.append(score1_all)

words_num2_all = 0
score2_all = 0
for i in range(len(wordid_ary1_all)):
    for j in range(len(wordid_ary3_all)):
        if wordid_ary1_all[i] == wordid_ary3_all[j]:
            words_num2_all += 1
score2_all = words_num2_all/10            
frequency_all.append(words_num2_all)
topic_overlap_all.append(score2_all)

words_num3_all = 0
score3_all = 0
for i in range(len(wordid_ary1_all)):
    for j in range(len(wordid_ary4_all)):
        if wordid_ary1_all[i] == wordid_ary4_all[j]:
            words_num3_all += 1 
score3_all = words_num3_all/10
frequency_all.append(words_num3_all)
topic_overlap_all.append(score3_all)
            
words_num4_all = 0
score4_all = 0            
for i in range(len(wordid_ary1_all)):
    for j in range(len(wordid_ary5_all)):
        if wordid_ary1_all[i] == wordid_ary5_all[j]:
            words_num4_all += 1
score4_all = words_num4_all/10
frequency_all.append(words_num4_all)
topic_overlap_all.append(score4_all)

words_num5_all = 0
score5_all = 0
for i in range(len(wordid_ary2_all)):
    for j in range(len(wordid_ary3_all)):
        if wordid_ary2_all[i] == wordid_ary3_all[j]:
            words_num5_all += 1
score5_all = words_num5_all/10
frequency_all.append(words_num5_all)
topic_overlap_all.append(score5_all)

words_num6_all = 0
score6_all = 0
for i in range(len(wordid_ary2_all)):
    for j in range(len(wordid_ary4_all)):
        if wordid_ary2_all[i] == wordid_ary4_all[j]:
            words_num6_all += 1
score6_all = words_num6_all/10
frequency_all.append(words_num6_all)
topic_overlap_all.append(score6_all)

words_num7_all = 0
score7_all = 0
for i in range(len(wordid_ary2_all)):
    for j in range(len(wordid_ary5_all)):
        if wordid_ary2_all[i] == wordid_ary5_all[j]:
            words_num7_all += 1
score7_all = words_num7_all/10
frequency_all.append(words_num7_all)
topic_overlap_all.append(score7_all)

words_num8_all = 0
score8_all = 0
for i in range(len(wordid_ary3_all)):
    for j in range(len(wordid_ary4_all)):
        if wordid_ary3_all[i] == wordid_ary4_all[j]:
            words_num8_all += 1
score8_all = words_num8_all/10
frequency_all.append(words_num8_all)
topic_overlap_all.append(score8_all)

words_num9_all = 0
score9_all = 0
for i in range(len(wordid_ary3_all)):
    for j in range(len(wordid_ary5_all)):
        if wordid_ary3_all[i] == wordid_ary5_all[j]:
            words_num9_all += 1
score9_all = words_num9_all/10
frequency_all.append(words_num9_all)
topic_overlap_all.append(score9_all)

words_num10_all = 0
score10_all = 0
for i in range(len(wordid_ary4_all)):
    for j in range(len(wordid_ary5_all)):
        if wordid_ary4_all[i] == wordid_ary5_all[j]:
            words_num10_all += 1
score10_all = words_num10_all/10
frequency_all.append(words_num10_all)
topic_overlap_all.append(score10_all)  

#%%normalization then plot
#overlap score, (0,1) 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns


#freq = frequency.sort()


#%%
#tp_over = pd.value_counts(frequency)
#t_over_num =
sns.kdeplot(topic_overlap_ir,label='BDTM',linestyle = '-.',color = 'black')#Initial reviews
sns.kdeplot(topic_overlap_ar,label='SRJST',linestyle = '--',color = 'black')#Additional reviews
sns.kdeplot(topic_overlap_all,label='LDA',linestyle = '-',color = 'black')#All reviews
#print(tp_over)
plt.legend()
plt.xlabel('Topic overlap')
plt.ylabel('Frequency')
plt.show()

#%%perplexity of LDA
import math
def perplexity(ldamodel, testset, dictionary, size_dictionary, num_topics):
    """calculate the perplexity of a lda-model"""
    
    #print ('the info of this ldamodel: \n')
    #print ('num of testset: %s; size_dictionary: %s; num of topics: %s'%(len(testset), size_dictionary, num_topics))
    prep = 0.0
    prob_doc_sum = 0.0
    topic_word_list = [] # store the probablity of topic-word:[(u'business', 0.010020942661849608),(u'family', 0.0088027946271537413)...]
    for topic_id in range(num_topics):
        topic_word = ldamodel.show_topic(topic_id, size_dictionary)
        dic = {}
        for word, probability in topic_word:
            dic[word] = probability
        topic_word_list.append(dic)
    doc_topics_ist = [] #store the doc-topic tuples:[(0, 0.0006211180124223594),(1, 0.0006211180124223594),...]
    for doc in testset:
        doc_topics_ist.append(ldamodel.get_document_topics(doc, minimum_probability=0))
    testset_word_num = 0
    for i in range(len(testset)):
        prob_doc = 0.0 # the probablity of the doc
        doc = testset[i]
        doc_word_num = 0 # the num of words in the doc
        for word_id, num in dict(doc).items():
            prob_word = 0.0 # the probablity of the word 
            doc_word_num += num
            word = dictionary[word_id]
            for topic_id in range(num_topics):
                # cal p(w) : p(w) = sumz(p(z)*p(w|z))
                prob_topic = doc_topics_ist[i][topic_id][1]
                prob_topic_word = topic_word_list[topic_id][word]
                prob_word += prob_topic*prob_topic_word
            prob_doc += math.log(prob_word) # p(d) = sum(log(p(w)))
        prob_doc_sum += prob_doc
        testset_word_num += doc_word_num
    prep = math.exp(-prob_doc_sum/testset_word_num) # perplexity = exp(-sum(p(d)/sum(Nd))
    print ("the perplexity of this ldamodel is : %s"%prep)
    return prep

topic=[] #初始主题数
log_perplexity_list1=[] #初始困惑度
log_perplexity_list2=[]
log_perplexity_list3=[]

perp_list1=[]
perp_list2=[]
perp_list3=[]

coherence_values1 = []
coherence_values2 = []
coherence_values3 = []

log_lhood_list = []

for i in range(1,20):
    topic.append(i)
    
    lda_ir = models.ldamodel.LdaModel(corpus=corpus_ir, id2word=dictionary_ir, num_topics=i)
    #log_perplexity1=lda_ir.log_perplexity(corpus_ir)
    #log_perplexity_list1.append(log_perplexity1)
    #perp=np.exp(-log_perplexity_list1[i-1])
    #perp = log_perplexity_list[i-1]
    corpus_i = corpora.MmCorpus('corpus_i.mm')
    test1 = []
    for c in range(corpus_i.num_docs):
        test1.append(corpus_i[c])
    perp = perplexity(lda_ir, test1, dictionary_ir, len(dictionary_ir.keys()), i)
    perp_list1.append(perp)
    
    coherencemodel1 = CoherenceModel(model=lda_ir, texts=words_list_ir,\
                                    dictionary=dictionary_ir, coherence='u_mass')
    coherence_values1.append(coherencemodel1.get_coherence())
    
    
    lda_ar = models.ldamodel.LdaModel(corpus=corpus_ar, id2word=dictionary_ar, num_topics=i)
    #log_perplexity2=lda_ar.log_perplexity(corpus_ar)
    #log_perplexity_list2.append(log_perplexity2)
    #perp=np.exp(-log_perplexity_list2[i-1])
    #perp = log_perplexity_list[i-1]
    corpus_a = corpora.MmCorpus('corpus_a.mm')
    test2 = []
    for c in range(corpus_a.num_docs):
        test2.append(corpus_a[c])
    perp = perplexity(lda_ar, test2, dictionary_ar, len(dictionary_ar.keys()), i)    
    perp_list2.append(perp)

    coherencemodel2 = CoherenceModel(model=lda_ar, texts=words_list_ar,\
                                    dictionary=dictionary_ar, coherence='u_mass')
    coherence_values2.append(coherencemodel2.get_coherence())    


    lda_all = models.ldamodel.LdaModel(corpus=corpus_all, id2word=dictionary_all, num_topics=i)
    #log_perplexity3=lda_all.log_perplexity(corpus_all)
    #log_perplexity_list3.append(log_perplexity3)
    #perp=np.exp(-log_perplexity_list3[i-1])
    #perp = log_perplexity_list[i-1]
    corpus_all = corpora.MmCorpus('corpus_all.mm')
    test3 = []
    for c in range(corpus_all.num_docs):
        test3.append(corpus_all[c])
    perp = perplexity(lda_all, test3, dictionary_all, len(dictionary_all.keys()), i)      
    perp_list3.append(perp)

    coherencemodel3 = CoherenceModel(model=lda_all, texts=word_list_all,\
                                    dictionary=dictionary_all, coherence='u_mass')
    coherence_values3.append(coherencemodel3.get_coherence())   

#%%

x=topic
y1=perp_list1
y2=perp_list2
y3=perp_list3
plt.plot(x,y1,linewidth=2,label='BDTM',linestyle = '-.',color = 'black')#Initial reviews
plt.plot(x,y2,linewidth=2,label='SRJST',linestyle = '--',color = 'black')#Additional reviews
plt.plot(x,y3,linewidth=2,label='LDA',linestyle = '-',color = 'black')#All reviews
plt.legend()
plt.xlabel("Number of Topic")
plt.ylabel("Perplexity")
plt.show()

#%%coherence_values
abs_coherence1 = []
abs_coherence2 = []
abs_coherence3 = []
for i in range(len(coherence_values1)):
    abs_coherence1.append(abs(coherence_values1[i])/10)
    abs_coherence2.append(abs(coherence_values2[i])/10)
    abs_coherence3.append(abs(coherence_values3[i])/10)

x=topic
z1=abs_coherence1
z2=abs_coherence2
z3=abs_coherence3
plt.plot(x,z1,linewidth=2,label='BDTM',linestyle = '-.',color = 'black')#Initial reviews
plt.plot(x,z2,linewidth=2,label='SRJST',linestyle = '--',color = 'black')#Additional reviews
plt.plot(x,z3,linewidth=2,label='LDA',linestyle = '-',color = 'black')#All reviews
plt.legend()
plt.xlabel("Number of Topic")
plt.ylabel("Coherence")
plt.show()
