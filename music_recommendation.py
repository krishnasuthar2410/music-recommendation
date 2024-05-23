#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 
from sklearn import tree
df=pd.read_csv('D:\\music.csv')
df


# In[24]:


X=df.drop(columns = ['genre'])
X
Y=df['genre']
Y
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
model=DecisionTreeClassifier()
model.fit(X,Y)
tree.export_graphviz(model,out_file='music-recommendation.dot',feature_names=['age','gender'],class_names=Y.unique(),label='all',rounded=True,filled=True)


# In[ ]:




