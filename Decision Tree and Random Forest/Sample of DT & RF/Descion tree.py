# -- coding: utf-8 --
"""
Created on Wed Oct 23 08:57:42 2024

@author: Ketan biradar
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
data=pd.read_csv("C:/12-supervised algoritm/Decsion-tree/credit.csv")
#data preparation
#check for null values
data.isnull().sum()
data.dropna()
data.columns
#There are 9 columns having non numeric values.let us try
#there is one column called phone which is not useful,
data=data.drop(["phone"],axis=1)
#now there are 16 columns
lb=LabelEncoder()
data['Checking_balance']=lb.fit_transform(data["checking_balance"])
data['credit_history']=lb.fit_transform(data["credit_history"])
data['purpose']=lb.fit_transform(data["purpose"])
data['savings_balance']=lb.fit_transform(data["savings_balance"])
data['employment_duration']=lb.fit_transform(data["employment_duration"])
data['other_credit']=lb.fit_transform(data["other_credit"])
data['housing']=lb.fit_transform(data["housing"])
data['job']=lb.fit_transform(data["job"])

#check for non numeric columns(I.M.P)
non_numeric_cols= data.select_dtypes(include=['object']).columns
print(non_numeric_cols)
data['checking_balance']=lb.fit_transform(data["checking_balance"])
data['default']=lb.fit_transform(data["default"])

## now let us check how many unique values are there in target column
data["default"].unique()
data["default"].value_counts()
## Now we want to split tree, we need all feature columns
colnames=list(data.columns)
# Now let us assign these columns to variable predictor
predictor=colnames[:15]
target=colnames[15]

from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size = 0.3)

from sklearn.tree import DecisionTreeClassifier as DT

help(DT)
model = DT(criterion= 'entropy')
model.fit(train[predictor], train[target])

#Prediction on test data
preds = model.predict(test[predictor])
pd.crosstab(test[target], preds, rownames=['Actual'],colnames=['Predictions'])

np.mean(preds == test[target]) #Taget the Accuracy
