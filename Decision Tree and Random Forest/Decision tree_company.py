# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 08:58:05 2024

@author: ketan

A cloth manufacturing company is interested to know about
the different attributes contributing to high sales. 
Build a decision tree & random forest model with Sales 
as target variable (first convert it into categorical variable).


Business Objective 
Minimize : Minimize costs of cloths.
Maximaze : Maximize overall Sales.
"""

import pandas as pd
import numpy as np

df=pd.read_csv("D:/Github datascience/Decision Tree and Random Forest/Dataset & problemstement/Company_Data.csv")
df.head(10)
df.tail(10)

#5 Number summary
df.describe()

df.shape
#400 rows 11 columns

df.columns
'''
(['Sales', 'CompPrice', 'Income', 'Advertising', 'Population', 'Price',
       'ShelveLoc', 'Age', 'Education', 'Urban', 'US')
'''

#Check the null value
df.isnull()
#False
df.isnull().sum()
#0 no null vall

#Boxplot
import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(df.Sales)
#Ther is one outliers
sns.boxplot(df.Income)
#There is no outliers

sns.boxplot(df.Advertising)
#There is also no outliers

sns.boxplot(df)
#There is some outliers in some columns in price,comprice and sales

#Data pre-processing
df.dtypes
# Some columns in int, float data types and some Object

duplicate=df.duplicated()
duplicate
#False means no duplicate

sum(duplicate)
#0 no duplicate

df.isnull().sum()
df.dropna()
df.columns

#converting into binary
data=df[::]
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
data["ShelveLoc"]=lb.fit_transform(data["ShelveLoc"])#For Bad is 0 for good it is 1 and for medium is 2
data['US']=lb.fit_transform(data['US']) #For Yes it is 1 for No it is 0
data['Urban']=lb.fit_transform(data['Urban']) #For Yes it is 1 for No it is 0

data['US'].unique()
data['US'].value_counts() #The total number of 1 is 258 and 0 is 142
colnames=list(data.columns)

predictors=colnames[:10]
target=colnames[9]

#Spliting data into training and testing data set
from sklearn.model_selection import train_test_split
train,test=train_test_split(data,test_size=0.3)

from sklearn.tree import DecisionTreeClassifier as DT
model=DT(criterion='entropy')
model.fit(train[predictors], train[target])
preds_test=model.predict(test[predictors])
preds_test
pd.crosstab(test[target], preds_test,rownames=['Actual'],colnames=['predictions'])
np.mean(preds_test==test[target])

# Now let us check accuracy on training dataset
preds_train=model.predict(train[predictors])
pd.crosstab(train[target], preds_train,rownames=['Actual'],colnames=['predictions'])
np.mean(preds_train==train[target])
#Output is 1.0 means model is 100% accurate
# Accuracy of train data > Accuracy test data i.e Overfit model










