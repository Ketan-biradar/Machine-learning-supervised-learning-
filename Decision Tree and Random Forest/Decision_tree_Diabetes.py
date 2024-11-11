# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 15:49:34 2024

@author: ketan

Divide the diabetes data into train and test datasets and 
build a Random Forest and Decision Tree model with Outcome 
as the output variable.

Business Objective: To improving efficiency, increasing revenue,
reducing costs, or enhancing customer satisfaction.

Minimize: Minimizing the number of negatives prediction.
Maximize: Maximizing the true positive rate.

Data Dictionary

        Features                 Type Relevance
0       Number of times pregnant   Quantitative data  Relevant
1   Plasma glucose concentration   Quantitative data  Relevant
2       Diastolic blood pressure   Quantitative data  Relevant
3    Triceps skin fold thickness   Quantitative data  Relevant
4           2-Hour serum insulin   Quantitative data  Relevant
5                Body mass index   Quantitative data  Relevant
6     Diabetes pedigree function   Quantitative data  Relevant
7                    Age (years)   Quantitative data  Relevant
8                 Class variable         Nominal data  Relevant


"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("D:/Github datascience/Decision Tree and Random Forest/Dataset & problemstement/Diabetes.csv")

df.head(10)
df.tail(10)

#5 number sumary
df.describe()

df.shape
#768 rows 9 columns
df.columns
'''
([' Number of times pregnant', ' Plasma glucose concentration',
       ' Diastolic blood pressure', ' Triceps skin fold thickness',
       ' 2-Hour serum insulin', ' Body mass index',
       ' Diabetes pedigree function', ' Age (years)', ' Class variable']
 '''
 
#Check the null value 
df.isnull()
#False no null value
df.isnull().sum()
#0 means no null value

#Box Plot
sns.boxplot(df)
#ALL columns having outliers expect  Plasma glucose concentration

#Data preprocessing

df.dtypes
# Some columns in int, float data types and some Object

duplicate=df.duplicated()
duplicate
#False no duplicate value

sum(duplicate)
#0 

#Normalize the data
def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
# Apply the norm_fun to data 
df1=norm_fun(df.iloc[:,:8])

df['Class variable']
df1['Class variable']=df['Class variable']

df.isnull().sum()
df.dropna()
df.columns

# Converting into binary
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
df1['Class variable']=lb.fit_transform(df1['Class variable'])
#The Yes become 1 and No become 0
df1['Class variable'].unique()
df1['Class variable'].value_counts()
#There are 0 is 500 and 1 is 268
colnames=list(df1.columns)

predictors=colnames[:8]
target=colnames[8]

#Spliting the data for training and testing
from sklearn.model_selection import train_test_split
train,test=train_test_split(df1,test_size=0.3)

from sklearn.tree import DecisionTreeClassifier as DT

model=DT(criterion='entropy')
model.fit(train[predictors],train[target])

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

#Output is 1.0 means model is 100% accurate
# Accuracy of train data > Accuracy test data i.e Overfit model



