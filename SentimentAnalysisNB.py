#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv(r"C:\Users\ASUS\Downloads\datasets\IMDB Dataset.csv")


# In[3]:


df


# In[4]:


df['sentiment'].value_counts()


# In[5]:


df['sentiment'].value_counts()


# In[6]:


df.sample(1).values


# In[7]:


from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
df['sentiment']=l.fit_transform(df['sentiment'])


# In[8]:


df


# In[9]:


from nltk.tokenize import word_tokenize


# In[10]:


from nltk.corpus import stopwords


# In[11]:


sw=stopwords.words('english')


# In[12]:


import string
p=string.punctuation


# In[13]:


len(sw)
sw


# In[14]:


len(p)


# In[15]:


import emoji
def rem_special_char(x):
    text=''
    for i in x:
        if i.isalnum():
            text=text+i
        else:
            text=text+' '
    return text


# In[16]:


import re
def remove_tags(x):
    pattern=re.compile('<.*>')
    return re.sub(pattern,'',x)

def rem_sw(x):
    l=[]
    for i in x:
        if i not in sw:
            l.append(i)
    return l

from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

def stem_words(x):
    l=[]
    for i in x:
        l.append(ps.stem(i))
    return l


# In[17]:


def preprocessing(text_low):
    
    text_no_tags=remove_tags(text_low)
    
    text_no_sc=rem_special_char(text_no_tags)
    
    tokens=word_tokenize(text_no_sc)
    
    tokens_no_sw=rem_sw(tokens)
    
    tokens_stem=stem_words(tokens_no_sw)
    
    return tokens_stem
    
    


# In[18]:


df['text_lower']=df['review'].apply(lambda x: x.lower())
df['text']=df['text_lower'].apply(preprocessing)


# In[19]:


df


# In[20]:


new_df=df[['text','sentiment']]


# In[21]:


new_df


# In[22]:


from sklearn.feature_extraction.text import CountVectorizer


# In[23]:


cv=CountVectorizer(max_features=2000)


# In[24]:


new_df['text']=new_df['text'].apply(lambda x: ''.join(x))
X=cv.fit_transform(new_df['text']).toarray()


# In[25]:


X.shape


# In[26]:


X


# In[27]:


y=new_df['sentiment']


# In[28]:


y.shape


# In[29]:


from sklearn.model_selection import train_test_split


# In[30]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[31]:


from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB


# In[32]:


m1=GaussianNB()
m2=MultinomialNB()
m3=BernoulliNB()


# In[33]:


from sklearn.metrics import confusion_matrix


# In[34]:


m1.fit(X_train,y_train)
m2.fit(X_train,y_train)
m3.fit(X_train,y_train)


# In[35]:


y_pred1=m1.predict(X_test)
y_pred2=m2.predict(X_test)
y_pred3=m3.predict(X_test)


# In[36]:


confusion_matrix_1=confusion_matrix(y_test,y_pred1)


# In[37]:


confusion_matrix_1


# In[38]:


confusion_matrix_2=confusion_matrix(y_test,y_pred2)


# In[39]:


confusion_matrix_2


# In[40]:


confusion_matrix_3=confusion_matrix(y_test,y_pred3)


# In[41]:


confusion_matrix_3


# In[42]:


from sklearn.metrics import accuracy_score
a1=accuracy_score(y_test,y_pred1)
a2=accuracy_score(y_test,y_pred2)
a3=accuracy_score(y_test,y_pred3)


# In[43]:


pd.DataFrame({'Model':['m1','m2','m3'],'Accuracy':[a1,a2,a3]})


# In[ ]:




