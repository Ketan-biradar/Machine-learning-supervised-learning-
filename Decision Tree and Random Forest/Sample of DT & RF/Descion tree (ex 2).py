# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 09:25:46 2024

@author: ketan
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
data=pd.read_csv("C:/12-supervised algoritm/Decsion-tree/salaries.csv")
#data preparation
#check for null values
data.isnull().sum()
data.dropna()
data.columns
#now there are 16 columns
lb=LabelEncoder()
data['company']=lb.fit_transform(data["company"])
data['job']=lb.fit_transform(data["job"])
data['degree']=lb.fit_transform(data["degree"])
data['salary_more_than_100k']=lb.fit_transform(data["salary_more_than_100k"])

non_numeric_cols= data.select_dtypes(include=['object']).columns
print(non_numeric_cols)
data['company']=lb.fit_transform(data["company"])

data["company"].unique()
data["company"].value_counts()

## Now we want to split tree, we need all feature columns
colnames=list(data.columns)
# Now let us assign these columns to variable predictor
predictor=colnames[:15]
target=colnames[3]

from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size = 0.3)

from sklearn.tree import DecisionTreeClassifier as DT

help(DT)
model = DT(criterion= 'entropy')
model.fit(train[predictor], train[target])

#Prediction on test data
preds = model.predict(test[predictor])
pd.crosstab(test[target], preds, rownames=['Actual'],colnames=['Predictions'])

np.mean(preds == test[target])
#model.predit([[2,1,1]])
