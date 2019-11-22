# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 21:47:37 2019

@author: yashr
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("Google_Stock_Price_Train.csv") #Data is fetched from a file
train = data.iloc[:,1:2].values # 

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
train_scaled = sc.fit_transform(train)

X_train = []
y_train = [] 
for i in range(60, 1258):
    X_train.append(train_scaled[i-60:i, 0])
    y_train.append(train_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()
regressor.add(LSTM(units = 60, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 60, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 60, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 60))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
regressor.fit(X_train, y_train, epochs = 120, batch_size = 32)


data1 = pd.read_csv("Google_Stock_Price_Test.csv")
test = data1.iloc[:,1:2].values


X_test=[]
data2 = pd.concat((data, data1))
data2 = data2['Open']
inp = data2[1198:].values
inp = inp.reshape(-1,1)
test_scaled = sc.fit_transform(inp)
for i in range(60, 80):
    X_test.append(test_scaled[i-60:i, 0])   
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
prediction = regressor.predict(X_test)
prediction = sc.inverse_transform(prediction)

plt.plot(test, color = 'blue', label = 'Real Stock Price')
plt.plot(prediction, color = 'green', label = 'Predicted Stock Price')
plt.title("Stock Prices real and predicted")
plt.xlabel("Time")
plt.ylabel('Stock Prices')
plt.legend()
plt.show()











