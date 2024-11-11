# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 14:09:55 2024

@author: ketan


In the recruitment domain, HR faces the challenge of predicting 
if the candidate is faking their salary or not. For example, a 
candidate claims to have 5 years of experience and earns 70,000 per 
month working as a regional manager. The candidate expects more money 
than his previous CTC. We need a way to verify their claims
(is 70,000 a month working as a regional manager with an experience 
of 5 years a genuine claim or does he/she make less than that?) 
Build a Decision Tree and Random Forest model with monthly 
income as the target variable. 

Business Objective
Minimize: To reduce or keep something as small as possible, 
often referring to costs, risks, or inefficiencies in a business process.

Maximize: To increase or optimize something to the greatest extent, 
often used in the context of profits, efficiency, or positive outcomes.

Data Dictionary

 Features                                      Type             Relevance
0   Position of the employee                 Qualititative data  Relevant
1   no of Years of Experience of employee    Continious data     Relevant
2   monthly income of employee               Continious data     Relevant
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("D:/Github datascience/Decision Tree and Random Forest/Dataset & problemstement/HR_DT.csv")

df.head(10)
df.tail(10)

df.columns
'''
(['Position of the employee', 'no of Years of Experience of employee',
       ' monthly income of employee']
'''

#Check the null value
df.isnull()
#False
df.isnull().sum()
#0 

#Box plot
sns.boxplot(df['Position of the employee'])
#No outliers 
sns.boxplot(df['no of Years of Experience of employee'])
#No outliers
sns.boxplot(df)
#NO outliers in the entiers datasets

#Data-precrossing
df.dtypes
#Some are object,float,int

#check the duplicated 
duplicates=df.duplicated()
duplicates
#False
sum(duplicates)
#38

#remove the dupkicates
df=df.drop_duplicates()
df=df.drop_duplicates(subset=('Position of the employee', 'no of Years of Experience of employee',' monthly income of employee'))
df.drop_duplicates(inplace=True)

duplicates=df.duplicated()
print(duplicates.sum())
#0 duplicates is mremove

#Convert into binary
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
df['Position of the employee']=lb.fit_transform(df['Position of the employee'])

X=df.iloc[:,:3].values
#Removing extra spaces can be present in column names
#clean up the column names by running
df.columns = df.columns.str.strip()
#Rename the Column
df.rename(columns={'actual_column_name': 'monthly income of employee'}, inplace=True)


#Target columns is separated
y=df['monthly income of employee']


from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=20)
#n_estimator:number of trees in the forest
model.fit(X_train,y_train)

model.score(X_test, y_test)
#0.508
y_predicted=model.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_predicted)
cm
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True)
plt.xlabel("Predicted")
plt.ylabel("Truth")




