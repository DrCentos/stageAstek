from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd


from tensorflow import keras
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

from datetime import datetime
from os import path, getenv, makedirs, listdir

# Importing the dataset
df = pd.read_csv('gs://nvallot_bucket/data.csv')
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


BUCKET_PATH = getenv('AIP_MODEL_DIR')
SC_PATH = path.join(BUCKET_PATH, 'scaler.bin')
MODEL_PATH = path.join(BUCKET_PATH, 'rnn-bitcoin-pred')
print("BUCKET_PATH: ", BUCKET_PATH)

SC_PATH = SC_PATH.replace('gs://nvallot_bucket/', '')

# Create output directory
makedirs(BUCKET_PATH, exist_ok=True)
makedirs(MODEL_PATH, exist_ok=True)
# Save model
regressor.save(MODEL_PATH)


#store scaler
import joblib
file_sc = joblib.dump(sc, 'std_scaler.bin', compress=True)


#store on bucket
import google.cloud.storage
from google.cloud import storage



client=storage.Client("glossy-precinct-371813");
bucket=client.get_bucket('nvallot_bucket');
blob=bucket.blob(SC_PATH)
blob.upload_from_filename('std_scaler.bin')

