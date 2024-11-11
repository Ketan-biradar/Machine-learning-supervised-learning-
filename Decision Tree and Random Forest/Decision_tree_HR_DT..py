# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 17:49:23 2024

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

#Boxplot
sns.boxplot(df["no of Years of Experience of employee"])
#There is no outliers

sns.boxplot(df)
#There is no outliers in the given dataset

#Data pre-processing
df.dtypes
# Some columns in int data types and some Object

#Check the duplicated
duplicates=df.duplicated()
duplicates
#False
sum(duplicates)
#There are 38 duplicates 

#Remove the duplicates
df = df.drop_duplicates()
df = df.drop_duplicates(subset=['no of Years of Experience of employee',' monthly income of employee','Position of the employee'])
df.drop_duplicates(inplace=True)

duplicates = df.duplicated()
print(duplicates.sum())  
#Now there is 0 duplicates in the dataset

# Converting into binary
# Convert categorical column to numerical using LabelEncoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["Position_of_the_employee"] = le.fit_transform(df["Position of the employee"])

# Split the data into predictors and target
colnames = list(df.columns)
predictors = colnames[:2]
target = colnames[2]

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.3)

# Create and train the Decision Tree model
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='entropy')
model.fit(train[predictors], train[target])

# Make predictions on the test set
preds_test = model.predict(test[predictors])

# Evaluate the model
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_test = accuracy_score(test[target], preds_test)
conf_matrix_test = confusion_matrix(test[target], preds_test)

print("Accuracy on Test Set:", accuracy_test)
print("Confusion Matrix on Test Set:\n", conf_matrix_test)
