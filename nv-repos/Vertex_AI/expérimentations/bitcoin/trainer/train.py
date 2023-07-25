from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd


from tensorflow import keras
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

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

#store scaler
import joblib
file_sc = joblib.dump(sc, 'std_scaler.bin', compress=True)


#store on bucket
import google.cloud.storage
from google.cloud import storage

"""Write and read a blob from GCS using file-like IO"""
# The ID of your GCS bucket
# bucket_name = "your-bucket-name"

# The ID of your new GCS object
# blob_name = "storage-object-name"

client=storage.Client("glossy-precinct-371813");
bucket=client.get_bucket('nvallot_bucket');
blob=bucket.blob('mysc.bin')
blob.upload_from_filename('std_scaler.bin')


#pour le predict ...
# Initialise a client
storage_client=storage.Client("glossy-precinct-371813");
# Create a bucket object for our bucket
bucket = storage_client.get_bucket('nvallot_bucket')
# Create a blob object from the filepath
blob = bucket.blob("mysc.bin")
# Download the file to a destination
blob.download_to_filename('std_scaler2.bin')

#load
scload=joblib.load('std_scaler2.bin')



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

# model = KerasRegressor(build_fn=create_model, epochs=100, batch_size=5, verbose=0)

# param_grid = {
#     'units': [32, 64],
#     'activation': ['relu', 'tanh', 'sigmoid'],
#     'optimizer': ['adam', 'RMSprop', 'SGD']
#     # 'dropout_rate': [0.0, 0.1, 0.2]
# }

regressor = create_model()
regressor.fit(X_train, y_train, batch_size=5, epochs=20)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1)
# grid_result = grid.fit(X_train, y_train)

# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# # Open the model
# from keras.models import load_model
# regressor = load_model('../output/rnn-bitcoin-pred.h5')
# # Open the Scaler
# import pickle
# sc = pickle.load(open('../output/scaler.pkl', 'rb'))

# regressor = grid_result.best_estimator_.model

# Predicting the future
test_set = df_test.values
inputs = np.reshape(test_set, (len(test_set), 1))
inputs = scload.transform(inputs)
inputs = np.reshape(inputs, (len(inputs), 1, 1))
predicted_BTC_price = regressor.predict(inputs)
predicted_BTC_price = scload.inverse_transform(predicted_BTC_price)

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

# Visualising the results
import matplotlib.pyplot as plt
plt.plot(test_set, color='red', label='Real Bitcoin Price')
plt.plot(predicted_BTC_price, color='blue', label='Predicted Bitcoin Price')
plt.title('Bitcoin Price Prediction')
plt.xlabel('Time')
plt.ylabel('Bitcoin Price')
plt.legend()
plt.show()

BUCKET_NAME = 'europe-west1-docker.pkg.dev/glossy-precinct-371813/nvallot_bucket'
from os import path

# Create output directory
#import os
#os.makedirs('./output', exist_ok=True)
# Save model
regressor.save('gs://nvallot_bucket/job_outputs/model')
# Save scaler
#import pickle
#pickle.dump(sc, open(path.join(BUCKET_NAME, 'Scaler.pkl'), 'wb'))
