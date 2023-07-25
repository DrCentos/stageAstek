# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from joblib import dump
from google.cloud import storage

import os

# Any results you write to the current directory are saved as output.

# First step, import libraries.
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

storage_client = storage.Client()

BUCKET_ROOT='/gcs/nvallot_bucket'
DATA_DIR = f'{BUCKET_ROOT}/data.csv'

df = pd.read_csv(DATA_DIR)


df['date'] = pd.to_datetime(df['Timestamp'],unit='s').dt.date
group = df.groupby('date')
Real_Price = group['Weighted_Price'].mean()

# split data
prediction_days = 30
df_train= Real_Price[:len(Real_Price)-prediction_days]
df_test= Real_Price[len(Real_Price)-prediction_days:]

# Data preprocess
training_set = df_train.values
training_set = np.reshape(training_set, (len(training_set), 1))
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)
X_train = training_set[0:len(training_set)-1]
y_train = training_set[1:len(training_set)]
X_train = np.reshape(X_train, (len(X_train), 1, 1))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Initializing the RNN
regressor = Sequential()

# Adding the input layers
regressor.add(LSTM(units=4, activation='sigmoid', input_shape=(None, 1)))

# Adding the output layers
regressor.add(Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer='Adam', loss='mean_squared_error')

# Fitting the RNN to the training set
regressor.fit(X_train, y_train, batch_size=5, epochs=100)

# SAVE MODEL on bucket
regressor.save(f'{BUCKET_ROOT}/output')

# Save the model to a local file
dump(regressor, "model.joblib")

# Upload the saved model file to GCS
bucket = storage_client.get_bucket("YOUR_GCS_BUCKET")
model_directory = os.environ["AIP_MODEL_DIR"]
storage_path = os.path.join(model_directory, "model.joblib")
blob = storage.blob.Blob.from_string(storage_path, client=storage_client)
blob.upload_from_filename("model.joblib")