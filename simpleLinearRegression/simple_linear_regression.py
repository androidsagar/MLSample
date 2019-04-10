#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 14:23:57 2019

@author: agl-android
"""
import numpy as np
import pandas as pd

df = pd.read_csv('/home/agl-android/.ML/MLSample/simpleLinearRegression/Salary_Data.csv')
x = df.iloc[:,:-1].values
y = df.iloc[:,1].values

#Split Data for Test and Train
from sklearn.model_selection import train_test_split
xTrain,xTest,yTrain,yTest = train_test_split(x,y,test_size = 0.3)

#Regression
from sklearn.linear_model import LinearRegression
regr = LinearRegression()
regr = regr.fit(xTrain,yTrain)
val = regr.predict(xTest)
