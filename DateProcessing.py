#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 22:37:58 2019

@author: sagar
"""
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#from sklearn.compose import ColumnTransformer

df = pd.read_csv("/home/sagar/Desktop/ML/Part 1 - Data Preprocessing/Data.csv")
x = df.iloc[:, :-1].values
y = df.iloc[:, 3].values

# Fix missing values
imputer = SimpleImputer(strategy="mean")
x[:, 1:3] = imputer.fit_transform(x[:, 1:3])

# Encode Categorical Data
labelEncode_x = LabelEncoder()
x[:, 0] = labelEncode_x.fit_transform(x[:, 0])

# Deppricated using ColumnTransformer instead
oneHotCodeEncoder = OneHotEncoder(categorical_features=[0])
x = oneHotCodeEncoder.fit_transform(x).toarray()

#ct = ColumnTransformer(
#    [('hot_encoder', OneHotEncoder(), [0])],
#    remainder='passthrough')
#x = np.array(ct.fit_transform(x), dtype=np.int)

labelEncode_y = LabelEncoder()
y = labelEncode_y.fit_transform(y)

# Splitting Data into training and Test
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2)

# Feature Scalling
from sklearn.preprocessing import StandardScaler
scX = StandardScaler()
xTrain = scX.fit_transform(xTrain)
xTest = scX.transform(xTest)








