# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 09:21:16 2024

@author: ketan
"""

import pandas as pd
df=pd.read_csv("C:/12-supervised algoritm/Random forest/movies_classification.csv")
df.info()
#Movies_classification dataset contain two columns which are object data type
#Hence convert into dummies
df=pd.get_dummies(df,columns=['3D_available','Genre'],drop_first=True)
#Let us assign input and out variable
predictors=df.loc[:,df.columns!='Start_Tech_Oscar']
#Except start_Tech_Oscar rest all columns are assigned as predictorss
#It 20 columns
target=df['Start_Tech_Oscar']
####################################
#Let us partition the datat
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(predictors,target,test_size=0.3 )
#####################################
#Model selection
from sklearn.ensemble import RandomForestClassifier
rand_for=RandomForestClassifier(n_estimators=500,n_jobs=1,random_state=42)
#n_estimators:It is number of job running parallel=1,if it is -1 then multiple
#random_state=control the randomness in bootstrapping
rand_for.fit(X_train,y_train)
pred_X_train=rand_for.predict(X_train)
pred_X_test=rand_for.predict(X_test)
###################################
#Now let us check the performance of the model
from sklearn.metrics import accuracy_score,confusion_matrix
accuracy_score(pred_X_test,y_test)
confusion_matrix(pred_X_test,y_test)
#####################
#for training dataset
accuracy_score(pred_X_train,y_train)
confusion_matrix(pred_X_train,y_train)
##The model is overfitted not overcome we need to optimize the code#########