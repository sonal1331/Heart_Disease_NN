# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 17:43:04 2019

@author: Sonal
"""
import numpy as np
import pandas as pd
dataset=pd.read_csv('heart.csv')
from sklearn import preprocessing
X=dataset.iloc[:,0:13].values
y=dataset.iloc[:,13].values
from keras.models import Sequential
from keras.layers import Dense
X=preprocessing.scale(X)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.3,random_state=0)
classifier=Sequential()
classifier.add(Dense(output_dim=13,activation='relu',input_dim=13))
classifier.add(Dense(output_dim=13,activation='relu'))
classifier.add(Dense(output_dim=13,activation='relu'))
classifier.add(Dense(output_dim=1,activation='sigmoid'))
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
classifier.fit(X_train,Y_train,batch_size=10,nb_epoch=70)
Y_pred=classifier.predict(X_test)
for i in range(91):
    if (Y_pred[i,0]>=0.5):
        Y_pred[i,0]=1
    else:
        Y_pred[i,0]=0
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,Y_pred)