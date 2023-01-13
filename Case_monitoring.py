# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 10:11:51 2022

@author: Jay Sun
"""

#charts by topic descriptions-IR

import jieba
import jieba.posseg as jp
from gensim import corpora, models
import csv
import numpy as np
#from scipy.stats import dirichlet
#import matplotlab.pyplot as plt
#from sklearn import joblib
#删除停用词
#%pre-process
print ('Loading initial reviews')
with open ('review_JD_TM.csv',encoding='gbk',errors='ignore') as ir:
    reader = csv.reader(ir)
    ir_text_list = [row[3] for row in reader]
    ir_text_array = np.array(ir_text_list)
    ir_text_corpus = ' '.join(ir_text_array)
    #print(ir_text_list)
    
print('Deleting punctuates')
from zhon.hanzi import punctuation as p#删除标点
#punctuation= ['，','：','。','*','！','？','~']
#print(p)
for i in p:
   ir_text_corpus = ir_text_corpus.replace(i,'')
#print(ir_text_corpus)
    
print('Cutting sentences into phrases')#假设评论词都在词典中，不启用HMM模型
seg_list = jieba.cut(ir_text_corpus, HMM=True)
seg_list1 = ("".join(seg_list))
#print(seg_list1)

print('Loading stopwrods')
X=[]
f_stop = open('stopwords_scu.txt','r',encoding='UTF-8') 
X=f_stop.read()
f_stop.close

print('Deleting stopwords')
final=''
for myword in seg_list1:
       if myword not in X:
              final+=myword
#print(final)        

#格式转化：string to list 
words_list= final.split(' ')
words_list = [i for i in words_list if i!= '']
#for i in range(len(words_list)):
 #  if words_list[i] == '':
  #  #清楚list中空白string
   #   del words_list[i]
words_ls = []
for text in words_list:
    words = [w.word for w in jp.cut(text)]
    words_ls.append(words)  
#提取词概率
import re
#single_lda=[]
    # 生成语料词典
# single_lda=words_ls[d:d+10]
dictionary = corpora.Dictionary(words_ls)
# 生成稀疏向量集
corpus = [dictionary.doc2bow(words) for words in words_ls]
# LDA模型，num_topics设置聚类数，即最终主题的数量
lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=3)
# 展示每个主题的前5的词语
wordlist=dictionary
#print(wordlist)
 #word_probability=[]
wp_array=[]
wp_float=[]
for topic_description in lda.print_topics(num_words=5):
    word_prob=re.findall(r'\d+\.\d*',topic_description[1])
    #转存数组
    for prob in word_prob:
        wp_array.append(prob)
wp_ary=[]
for n in wp_array:
     wp_ary.append(float(n))
     
#主题所有词概率
tp_w_mat=[]
tp_w_mat=np.array(lda.get_topics())
print('probability of words in topic0 is:',tp_w_mat[0])#第0主题下的词分布   
#模型的主题分布,相对熵
tp_dis=[]
new_topics = lda[corpus]
for topic in new_topics:
    tp_tuple=np.array(topic)
    tp_dis.append(tp_tuple[:,1])
#print(tp_dis)
#mean of topic distribution used for simulation
tp_dis3=[]
for dis in tp_dis:
    if len(dis)==3:
      tp_dis3.append(dis) 
#print(tp_dis3)
tp1=[]
tp2=[]
tp3=[]
for tpdis in tp_dis3:
    tp1.append(tpdis[0])
    tp2.append(tpdis[1])
    tp3.append(tpdis[2])
theta1=np.mean(tp1)
theta2=np.mean(tp2)
theta3=np.mean(tp3)
print(theta1,theta2,theta3)


#%%B-W chart
#w2v_model_file='w2v_model_file'
#filemodel=gensim.models.Word2Vec.load(w2v_model_file)
#filemodel.init_sims(replace=True)
#distance_array=np.zeros(len(words_ls))
#for t in range(len(words_ls)):
#    distance=filemodel.wmdistance(words_list[t],words_list[t+1])
#    distance_array[t]=distance
from scipy.stats import wasserstein_distance    
import scipy.stats
import matplotlib.pyplot as plt
from SRJST import thetaz0
KL=np.zeros(len(tp_dis3))
time=np.zeros(len(tp_dis3))
for t in range(len(tp_dis3)):
    time[t]=t
    KL[t]=wasserstein_distance(thetaz0,tp_dis3[t])
time_case=np.zeros(25)
KL_case=np.zeros(25)
for case in range(25):
    time_case[case]=time[case]
    KL_case[case]=KL[case+57]
    
CL=np.percentile(KL_case,60)
print(CL)
CL_case=np.zeros(25)
for case in range(25):
    CL_case[case]=CL
#KL_case.sort()  #从小到大排列
plt.rcParams['savefig.dpi'] = 1000 #图片像素
plt.xlabel("Time")
plt.ylabel("Charting Statistics")
plt.plot(time_case,CL_case,color = 'k',\
         dashes=(5, 3), linewidth = 0.8, linestyle = '--',\
             label='Control Limit=0.259')
plt.legend()
plt.plot(time_case,KL_case,'-o',color = 'k', \
         markersize=2, linewidth = 1.2)

print(tp_dis3[59], tp_dis3[60], tp_dis3[61])#OC signal 3
print(tp_dis3[62], tp_dis3[63], tp_dis3[64])#OC signal 6
print(tp_dis3[71], tp_dis3[72], tp_dis3[73])#OC signal 15



#%%S-K chart
import matplotlib.pyplot as plt
#from SRJST import thetaz0
KL=np.zeros(len(tp_dis3))
time=np.zeros(len(tp_dis3))
for t in range(len(tp_dis3)):
    time[t]=t
    KL[t]=scipy.stats.entropy(thetaz0,tp_dis3[t])
time_case=np.zeros(25)
KL_case=np.zeros(25)
for case in range(25):
    time_case[case]=time[case]
    KL_case[case]=KL[case+57]
CL=np.percentile(KL_case,90)*58.4
print(CL)
CL_case=np.zeros(25)
for case in range(25):
    CL_case[case]=CL
#KL_case.sort()  #从小到大排列
#plt.rcParams['savefig.dpi'] = 1000 #图片像素
plt.xlabel("Time")
plt.ylabel("Charting Statistics")
plt.plot(time_case,CL_case,color='k', \
         dashes=(5, 3), linewidth = 0.8, linestyle = '--',\
             label='Control Limit=51.331')
plt.legend()
plt.plot(time_case,KL_case*58.4,'-o',color = 'k', \
         markersize=2, linewidth = 1.2)

#%%R-K chart
import scipy.stats
import matplotlib.pyplot as plt
import utils
#from SRJST import thetaz0
KL=np.zeros(len(tp_dis3))
time=np.zeros(len(tp_dis3))
for t in range(len(tp_dis3)):
    time[t]=t
    KL[t]=utils.RJST_chart1(thetaz0,tp_dis3[t])
time_case=np.zeros(25)
KL_case=np.zeros(25)
for case in range(25):
    time_case[case]=time[case]
    KL_case[case]=KL[case+57]
    
CL=np.percentile(KL_case,90)*58.4*2
print(CL)
CL_case=np.zeros(25)
for case in range(25):
    CL_case[case]=CL
#KL_case.sort()  #从小到大排列
#plt.rcParams['savefig.dpi'] = 1000 #图片像素
plt.xlabel("Time")
plt.ylabel("Charting Statistics")
plt.plot(time_case,CL_case,color='k', \
         dashes=(5, 3), linewidth = 0.8, linestyle = '--',\
             label='Control Limit=123.322')
plt.legend()
plt.plot(time_case,KL_case*58.4*2,'-o',color = 'k', \
         markersize=2, linewidth = 1.2)