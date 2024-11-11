# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 13:06:08 2024

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

# Clean the column names by removing any unwanted spaces or special characters
df.columns = df.columns.str.replace(r'\s+', '', regex=True).str.strip()

# Print cleaned column names to check
print(df.columns.tolist())

# Separate the target variable and features
y = df['Classvariable']  # Target variable
X = df.drop(columns=['Classvariable'])  # Drop the target column for normalization

# Normalize the data (only the features)
def norm_fun(i):
    x = (i - i.min()) / (i.max() - i.min())
    return x

# Apply normalization to the feature columns
X_normalized = norm_fun(X)

# Check for any missing values after normalization (although it shouldn't create any)
print(X_normalized.isnull().sum())


from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=20)
#n_estimator:number of trees in the forest
model.fit(X_train,y_train)

model.score(X_test, y_test)
#0.7922 
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