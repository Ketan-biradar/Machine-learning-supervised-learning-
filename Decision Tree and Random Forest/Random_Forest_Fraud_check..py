# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 13:20:14 2024

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

X=df.iloc[:,:5].values  
y=df.Urban


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




















