# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 12:16:27 2022

@author: Jay Sun
"""


import jieba
import jieba.posseg
import csv

#%reading
with open('reviews_English.csv',encoding='gbk',errors='ignore') as total:
    reader = csv.reader(total)
    initial_review_e = [row[1] for row in reader]
    
with open('reviews_English.csv',encoding='gbk',errors='ignore') as total:
    reader = csv.reader(total)
    additional_review_e = [row[3] for row in reader]
    

review_str_ir_e = ' '.join(initial_review_e)
review_str_ar_e = ' '.join(additional_review_e)

words_ir_e = ' '.join(jieba.cut(review_str_ir_e))
words_ar_e = ' '.join(jieba.cut(review_str_ar_e))


from wordcloud import WordCloud

X={}
stop_words_e = open('Estopwords.txt','r',encoding='UTF-8') 
X=stop_words_e.read()
stop_words_e.close
X_list1 = X.split()#%%ploting


import matplotlib.colors as colors
#define wordcloud
colormaps = colors.ListedColormap(['#B0E0E6', '#ADD8E6', '#00BFFF',
                                   '#87CEEB', '#87CEFA', '#4682B4',
                                   '#B0C4DE', '#6495ED', '#4169E1'])
wc_e = WordCloud(
    prefer_horizontal=1,
    font_path='msyh.ttc',
    scale= 32,
    colormap = colormaps,
    background_color='white',
    width= 300,
    height=400,
    stopwords=X_list1,
    max_font_size=50,
    random_state=50
    )

#%%
wc_e.generate(words_ir_e)
wc_e.to_file('Ecloud1.jpg')

wc_e.generate(words_ar_e)
wc_e.to_file('Ecloud2.jpg')

#%%reading
with open('review_JD_TM.csv',encoding='gbk',errors='ignore') as total:
    reader = csv.reader(total)
    initial_review = [row[3] for row in reader]
    
with open('review_JD_TM.csv',encoding='gbk',errors='ignore') as total:
    reader = csv.reader(total)
    additional_review = [row[5] for row in reader]

review_str_ir = ' '.join(initial_review)
review_str_ar = ' '.join(additional_review)

words_ir = ' '.join(jieba.cut(review_str_ir))
words_ar = ' '.join(jieba.cut(review_str_ar))


from wordcloud import WordCloud

X={}
stop_words = open('stopword_1.txt','r',encoding='UTF-8') 
X=stop_words.read()
stop_words.close
X_list = X.split()#%%ploting


import matplotlib.colors as colors
#define wordcloud
colormaps = colors.ListedColormap(['#B0E0E6', '#ADD8E6', '#00BFFF',
                                   '#87CEEB', '#87CEFA', '#4682B4',
                                   '#B0C4DE', '#6495ED', '#4169E1'])
wc = WordCloud(
    prefer_horizontal=1,
    font_path='msyh.ttc',
    scale= 32,
    colormap = colormaps,
    background_color='white',
    width= 300,
    height=400,
    stopwords=X_list,
    max_font_size=50,
    random_state=50
    )

wc.generate(words_ir)
wc.to_file('Ccloud1.jpg')

wc.generate(words_ar)
wc.to_file('Ccloud2.jpg')
