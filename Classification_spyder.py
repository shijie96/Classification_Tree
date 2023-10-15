# -*- coding: Classification *-
"""
Created on Wed Sep 27 16:19:46 2023

@author: Shijie Geng
"""
#%%
import pandas as pd 
import numpy as np
heart_disease = pd.read_csv("C:/Users/shiji/OneDrive/python/DATASET/heart.csv",
                            header = None)
heart_disease
heart_disease.head()
heart_disease.columns = ['age',
                         'sex',
                         'cp',
                         'restbp',
                         'chol',
                         'fbs',
                         'restecg',
                         'thalach',
                         'exang',
                         'oldpeak',
                         'slope',
                         'ca',
                         'thal',
                         'hd']
#%%
heart_disease.dtypes
heart_disease['ca'].unique()
heart_disease.shape
#%%
heart_disease = heart_disease.loc[1:1026]
#%%
heart_disease['ca'].unique()
len(heart_disease.loc[(heart_disease['ca']=='?')
                      |
                      (heart_disease['thal']=='?')])
#%%
len(heart_disease.loc[(heart_disease['ca'] != '?')
                      |
                      (heart_disease['thal']!= '?')]
    )
heart_disease.shape
#%%
x = heart_disease.drop('hd',axis=1).copy()
x.head()
y = heart_disease['hd'].copy()
y.head()
#%%
x_encode = pd.get_dummies(x, columns = ['cp',
                                        'restecg',
                                        'slope',
                                        'thal'])
x_encode.head()
#%%
y.unique()
#%%
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import plot_tree
#from sklearn.metrics import plot_confusion_matrix
#%%
## split data into train dataset and test dataset for decison tree building.
x_train, x_test, y_train,y_test = train_test_split(x_encode, y,random_state=42)
#%%
### Create a decision tree and fit it to the training data
clf_dt = DecisionTreeClassifier(random_state=42)
clf_dt = clf_dt.fit(x_train,y_train)
#%%
###plot the tree
plt.figure(figsize=(15,7.5))
plot_tree(clf_dt,
          filled=True,
          rounded = True,
          class_names = ['No HD','Yes HD'],
          feature_names = x_encode.columns)
#%%
## plot confusion matrix
plot_confusion_metrix(clf_dt,
                      x_test,
                      y_test,
                      display_labels = ['Does not have HD','Have HD'])
#%%
### Graph the accuracy of trees using the training dataset and testing dataset as a function of alpha.
train_scores = [clf_dt.score(x_train,y_train) for clf_dt in clf_dt]
test_scores = [clf_dt.score (x_test, y_test) for clf_dt in clf_dt]
#%%
