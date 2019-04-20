# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 15:53:28 2019

@author: Sagar
"""
import pandas as pd
import numpy as np

#import data sects
df = pd.read_csv('D:\MachineLearning\Project\MLSample\MultipleLinearRegression\startups.csv')

x = df.iloc[:,:-1].values
y = df.iloc[:,4].values 

# enode catagorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelEncode_x = LabelEncoder()
x[:,3] = labelEncode_x.fit_transform(x[:,3])

# create dummy vairables
oneHotCodeEncoder = OneHotEncoder(categorical_features=[3])
x = oneHotCodeEncoder.fit_transform(x).toarray() 

# remove dummy variable
# replace 1 dummy variable with ones for constants
x[:,0] = np.ones(50).astype(int) 
#x = np.append(arr = np.ones((50,1)).astype(int) ,values = x ,axis = 1)

# split data test and train
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2)

# Do Linear Regression
#from sklearn.linear_model import LinearRegression
#regr = LinearRegression()
#regr = regr.fit(xTrain,yTrain)
#yPridict = regr.predict(xTest) 

# Do Backword Elimination
# SL = 5%
import statsmodels.formula.api as sm
#x_opt = x[:,[0,1,2,3,4,5]]
#x_opt = x[:,[0,1,3,4,5]]
#x_opt = x[:,[0,3,4,5]]
#x_opt = x[:,[0,3,5]]
x_opt = x[:,[0,3]]
regr_OLS = sm.OLS(endog = y,exog = x_opt).fit() 
regr_OLS.summary()


