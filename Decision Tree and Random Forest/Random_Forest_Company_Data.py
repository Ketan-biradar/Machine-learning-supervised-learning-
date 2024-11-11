# -*- coding: utf-8 -*-
'''
Created on Sat Nov  9 10:22:01 2024

@author: ketan

A cloth manufacturing company is interested to know about
the different attributes contributing to high sales. 
Build a decision tree & random forest model with Sales 
as target variable (first convert it into categorical variable).


Business Objective 
Minimize : Minimize costs of cloths.
Maximaze : Maximize overall Sales.

'''
import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv("D:/Github datascience/Decision Tree and Random Forest/Dataset & problemstement/Company_Data.csv")
df.head(10)
df.tail(10)

# 5 Number summary
df.describe()

df.shape
# 400 rows 11 columns

df.columns
'''
(['Sales', 'CompPrice', 'Income', 'Advertising', 'Population', 'Price',
       'ShelveLoc', 'Age', 'Education', 'Urban', 'US')
'''

# Check the null value
df.isnull()
# False
df.isnull().sum()
# 0 no null vall

# Boxplot
import matplotlib.pyplot as plt

sns.boxplot(df.Sales)
# There are two outliers
sns.boxplot(df.Income)
# There are no outliers

sns.boxplot(df.Advertising)
# There are also no outliers

sns.boxplot(df)
# There are some outliers in some columns in Price, CompPrice, and Sales

# Data pre-processing
df.dtypes
# Some columns are int, float data types and some Object

duplicate = df.duplicated()
duplicate
# False means no duplicate

sum(duplicate)
# 0 no duplicate

df.isnull().sum()
df.dropna()
df.columns

# Converting into binary
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()

# Apply LabelEncoder to the categorical columns: ShelveLoc, US, and Urban (skip strip here)
df["ShelveLoc"] = lb.fit_transform(df["ShelveLoc"])
df["US"] = lb.fit_transform(df["US"])
df["Urban"] = lb.fit_transform(df["Urban"])

# Convert Sales to categorical variable (High/Low based on median)
# We add the columns of SalesCategory based on the Sales 
# If Sales value if value is greater than 9 then High, less than 9 Low
df["SalesCategory"] = np.where(df["Sales"] > df["Sales"].median(), "High", "Low")

# After converting high and low, converting into numerical value i.e for high 0 and low 1
y = lb.fit_transform(df["SalesCategory"])

# Define X after encoding categorical columns and excluding 'Sales' and 'SalesCategory'
X = df.drop(["Sales", "SalesCategory"], axis=1).values
print("Data types of X:", pd.DataFrame(X).dtypes)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=20)
# n_estimators: number of trees in the forest
model.fit(X_train, y_train)

# Model accuracy
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)
#The model Accuracy is 0.7375

# Predictions
y_predicted = model.predict(X_test)

# confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()
