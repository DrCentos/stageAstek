# -*- coding: utf-8 -*-
"""main (1).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-caaiUcwUVi6nccbyvcRYS7OLdLNNoQU
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install numpy 
# If you are using a Conda environment (such as with the Data Science Kernel), you may prefer to use conda instead of pip:
# %conda install numpy

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("./"))

# Any results you write to the current directory are saved as output.

import pandas as pd
import io
import boto3

s3c = boto3.client('s3', region_name="eu-west-3")
obj = s3c.get_object(Bucket="nvabucket",Key="coinbaseUSD_1-min_data_2014-12-01_to_2018-11-11.csv")
df = pd.read_csv(io.BytesIO(obj["Body"].read()))

import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
# from keras.wrappers.scikit_learn import KerasRegressor
# from sklearn.model_selection import GridSearchCV
import numpy as np 
import pandas as pd 


df['date'] = pd.to_datetime(df['Timestamp'], unit='s').dt.date
group = df.groupby('date')
Real_Price = group['Weighted_Price'].mean()

prediction_days = 30
df_train = Real_Price[:len(Real_Price)-prediction_days]
df_test = Real_Price[len(Real_Price)-prediction_days:]

# Adaptation to the RNN
training_set = df_train.values
training_set = np.reshape(training_set, (len(training_set), 1))
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)
X_train = training_set[0:len(training_set)-1]
y_train = training_set[1:len(training_set)]
X_train = np.reshape(X_train, (len(X_train), 1, 1))

# Commented out IPython magic to ensure Python compatibility.
# %pip install keras

# Commented out IPython magic to ensure Python compatibility.
# %pip install tensorflow

# Building the RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
# from keras.layers import Dropout

def create_model(units=32, activation='relu', optimizer='adam', dropout_rate=0.0):
    regressor = Sequential()
    regressor.add(LSTM(units=units, activation=activation, input_shape=(None, 1)))
    # regressor.add(Dropout(dropout_rate))
    regressor.add(Dense(units=1))
    regressor.compile(optimizer=optimizer, loss='mean_squared_error')
    return regressor

regressor = create_model()
regressor.fit(X_train, y_train, batch_size=5, epochs=20)

# Predicting the future
test_set = df_test.values
inputs = np.reshape(test_set, (len(test_set), 1))
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (len(inputs), 1, 1))
predicted_BTC_price = regressor.predict(inputs)
predicted_BTC_price = sc.inverse_transform(predicted_BTC_price)

# Get model Root Mean Squared Error
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(test_set, predicted_BTC_price))
print('Root Mean Squared Error: ', rmse)

# Get model R2 score
from sklearn.metrics import r2_score
r2 = r2_score(test_set, predicted_BTC_price)
print('R2 score: ', r2)

# Explained variance score
from sklearn.metrics import explained_variance_score
evs = explained_variance_score(test_set, predicted_BTC_price)
print('Explained variance score: ', evs)

import joblib
# Persist the model
path = "./scaler.bin"
joblib.dump(sc, path ,compress=True)
print("scaler persisted at " + path)

s3 = boto3.client('s3')
s3.put_object(Bucket='nvabucket', Key='scaler.bin',Body=path)

regressor.save('./model')

s3 = boto3.client('s3')
s3.put_object(Bucket='nvabucket', Key='model',Body='./model')