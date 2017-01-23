#!/usr/bin/python

import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

#read data using pandas
dataframe = pd.read_fwf('brain_body.txt')
xValues = dataframe[['Brain']]
yValues = dataframe[['Body']]

#train model on data
bodyReg = linear_model.LinearRegression()
bodyReg.fit(xValues, yValues)
pred = bodyReg.predict(xValues)

#test new for new value
brainSz = 300
print('Brain Size = ', brainSz, ' Body Weight = ', bodyReg.predict(brainSz))

#Plot
plt.scatter(xValues, yValues)
plt.plot(xValues, pred)
plt.show()

