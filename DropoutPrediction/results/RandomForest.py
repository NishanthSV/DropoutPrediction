#!/usr/bin/env python
# coding: utf-8

# In[12]:


# Constants and imports.

BASE_NUM = 1
RANDOM_STATE = None
CV = 5
TEST_SIZE = 0.2

import os
import itertools
import pandas as pd
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[13]:


# Load data.

data = pd.read_csv('base_1.csv', sep=';')

data.head()


# In[14]:


import random
COURSE = []
for i in range(100):
    randlist = ['CSE', 'AUTOMOBILE_ENGINEERING', 'APPLIED_SCIENCE', 'BIO_TECHNOLOGY', 'BIOMEDICAL_ENGINEERING','CHEMISTRY','CIVIL_ENGINEERING','ECE','EEE','ENGLISH','FASHION_TECHNOLOGY','HUMANITICS','IT','MATHEMATICS','COMPUTER_APPLICATIONS','MECHANICAL','METALLURIGCAL','PHYSICS','PRODUCTION','ROBOTICS','TEXTILE']
    COURSE.append(random.choice(randlist))
# for val in COURSE:
#     print(val)
data["COURSE"] = COURSE
# print(data['COURSE'])
ATTENDANCE = []
for i in data['LARGE_PERIOD_ABSENT']:
    ATTENDANCE.append(1-i)
data['ATTENDANCE'] = ATTENDANCE    
data.drop(["COURSE_OF_STUDY","NATIONALITY", "AGE_WHEN_STARTED", "ELEMENTARY_SCHOOL", "SCHOOL", "ELEMENTARY_GRADE_9", "ELEMENTARY_GRADE_1", "ELEMENTARY_GRADE_2", "ELEMENTARY_GRADE_3", "ELEMENTARY_GRADE_4", "ELEMENTARY_GRADE_AVG", "SUPERVISOR_GROUP_SIZE","CLASS_BASED_SCHOOL","SMALL_PERIOD_ON_TIME","SMALL_PERIOD_ABSENT","SMALL_PERIOD_LATE",
           "SMALL_PERIOD_AVG_ASSIGNMENT_GRADE","LARGE_PERIOD_AVG_ASSIGNMENT_GRADE","CREDITS_LAST_SEMESTER","CLASSES_LAST_SEMESTER","LARGE_PERIOD_ON_TIME","LARGE_PERIOD_LATE","LARGE_PERIOD_ABSENT"], axis = 1, inplace = True)
# for col in data.columns:
#     print(col)
data.head()
# for val in data.columns:
#     print(val)
data.rename(columns={"LARGE_PERIOD_AVG_GRADE":"CGPA"})


# In[15]:


data.fillna(data.mean(), inplace=True)
data = data.drop(['COURSE'],axis = 1)


# In[16]:


# Split train / test

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
for train_index, test_index in split.split(data, data['DROPPED_OUT']):
    train_set = data.loc[train_index]
    test_set = data.loc[test_index]


# In[17]:


X_train = train_set.drop(['DROPPED_OUT'],axis = 1)
Y_train = train_set['DROPPED_OUT']
X_test = test_set.drop(['DROPPED_OUT'],axis = 1)
Y_test = test_set['DROPPED_OUT']
print(X_train.columns)


# In[18]:


from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=200)
clf.fit(X_train,Y_train)
y_pred = clf.predict(X_test)
score = metrics.accuracy_score(Y_test,y_pred)
print((score))


# In[20]:


from sklearn.metrics import roc_curve , roc_auc_score

yscore = clf.predict_proba(X_test)[:,1]
false_positive_rate , true_positive_rate , threshold = roc_curve(Y_test,yscore)
print("roc_auc_score : ", roc_auc_score(Y_test,yscore))


# In[21]:


plt.title('Receiver operating characteristic')
plt.plot(false_positive_rate , true_positive_rate)
plt.plot([0,1] , ls = "--")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:





# In[ ]:




