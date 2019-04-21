# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 15:57:51 2019

@author: Sagar
"""

import pandas as pd
import numpy as np

#import data sects
df = pd.read_csv('D:\MachineLearning\Project\MLSample\PolynomialRegression\Position_Salaries.csv')

x = df.iloc[:,1:2].values
y = df.iloc[:,2].values 

#change X for polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly_f = PolynomialFeatures(degree = 4)
x_pol = poly_f.fit_transform(x, y)

# Regression
from sklearn.linear_model import LinearRegression
regr = LinearRegression()
regr.fit(x_pol,y)


# plot data
import matplotlib.pyplot as plt
plt.scatter(x, y, color = 'green')
plt.plot(x, regr.predict(poly_f.fit_transform(x)), color = 'blue')
plt.show()


