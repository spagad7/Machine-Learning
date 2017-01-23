#!/usr/bin/python

import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

df = pd.read_csv('challenge_dataset.txt')
xValues = df[[0]]
yValues = df[[1]]

linReg = linear_model.LinearRegression()
linReg.fit(xValues, yValues)
predict = linReg.predict(xValues)

plt.scatter(xValues, yValues)
plt.plot(xValues, predict)
plt.show()
