# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 17:13:16 2024

@author: ketan

Build a Decision Tree & Random Forest model on the fraud data. 
Treat those who have taxable_income <= 30000 as Risky and others 
as Good (discretize the taxable income column).

Business Objective: Attaining a defined goal for the organization.

Minimize: Decreasing or eliminating certain factors or costs.
Maximize: Enhancing or optimizing specific aspects for the greatest benefit.

Data Dictionary

 Features               Type          Relevance
0        Undergrad       Nominal data  Relevant
1   Marital.Status   Categorical data  Relevant
2   Taxable.Income  Quantititave data  Relevant
3  City.Population  Quantitative data  Relevant
4  Work.Experience  Quantitative data  Relevant
5            Urban       Nominal data  Relevant

"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("D:/Github datascience/Decision Tree and Random Forest/Dataset & problemstement/Fraud_check.csv")
df.head()
df.tail()

#5 number summary
df.describe()

df.shape
#600 rows and 6 columns

df.columns
'''
(['Undergrad', 'Marital.Status', 'Taxable.Income', 'City.Population',
       'Work.Experience', 'Urban']
'''

# check for null values
df.isnull()
# False
df.isnull().sum()
# 0 no null values

#Box plot
sns.boxplot(df["City.Population"])

sns.boxplot(df)
#There is no outliers in the datasets

#data pre-processing
df.dtypes
# Some columns in int, float data types and some object

duplicates=df.duplicated()
duplicates
#False Not having duplicates values

sum(duplicates)
#0

#Converting into binary
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
df['Undergrad']=lb.fit_transform(df['Undergrad'])
#For NO is 0 and for YES is 1 
df['Marital.Status']=lb.fit_transform(df['Marital.Status']) 
#for Single is 2,for Divorced is 0 and for Married is 1
df['Urban']=lb.fit_transform(df['Urban'])
#for YES is 1 and NO is 0

df['Urban'].unique()
df['Urban'].value_counts()
#for 1 is 302 for 0 is 298
colnames=list(df.columns)

predictors=colnames[:5]
target=colnames[5]

# Spliting data into training and testing data set
from sklearn.model_selection import train_test_split
train,test=train_test_split(df,test_size=0.3)

from sklearn.tree import DecisionTreeClassifier as DT

model=DT(criterion='entropy')
model.fit(train[predictors], train[target])


#confusion matrix for test
preds_test=model.predict(test[predictors])
preds_test
pd.crosstab(test[target], preds_test,rownames=['Actual'],colnames=['predictions'])
np.mean(preds_test==test[target])


#confusion matrix for train
# Now let us check accuracy on training dataset
preds_train=model.predict(train[predictors])
pd.crosstab(train[target], preds_train,rownames=['Actual'],colnames=['predictions'])
np.mean(preds_train==train[target])

# Accuracy of train data > Accuracy test data i.e Overfit model