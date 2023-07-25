# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import mlflow
import mlflow.keras
import mlflow.tensorflow
import numpy as np
import argparse
import os
import glob

import matplotlib.pyplot as plt

import pandas as pd
import keras
import joblib
from keras.models import Sequential, model_from_json
from keras.layers import Dense, LSTM
from keras.optimizers import RMSprop
from keras.callbacks import Callback

import tensorflow as tf

#from azureml.core import Run
from sklearn.preprocessing import MinMaxScaler

def create_model(units=32, activation='relu', optimizer='adam', dropout_rate=0.0):
    # Building the RNN
    regressor = Sequential()
    regressor.add(LSTM(units=units, activation=activation, input_shape=(None, 1)))
    regressor.add(Dense(units=1))
    regressor.compile(optimizer=optimizer, loss='mean_squared_error')
    return regressor



class LogRunMetrics(Callback):
    # callback at the end of every epoch
    def on_epoch_end(self, epoch, log):
        # log a value repeated which creates a list
        print('log :', log)
        #run.log('Loss', log['val_loss'])
        #run.log('Accuracy', log['val_accuracy'])

print("Keras version:", keras.__version__)
print("Tensorflow version:", tf.__version__)

# Start Logging
mlflow.start_run()

# enable autologging
mlflow.sklearn.autolog()

# start an Azure ML run
#run = Run.get_context()

OUTPUT_DIR = './outputs/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

#input_file = path.join(input_path, 'coinbaseUSD.csv')
input_file = 'https://nvastockage.blob.core.windows.net/nvabucket/coinbaseUSD.csv'
df = pd.read_csv(input_file)
df['date'] = pd.to_datetime(df['Timestamp'], unit='s').dt.date
group = df.groupby('date')
Real_Price = group['Weighted_Price'].mean()

prediction_days = 30
df_train = Real_Price[:len(Real_Price)-prediction_days]

# Adaptation to the RNN
training_set = df_train.values
training_set = np.reshape(training_set, (len(training_set), 1))
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)
X_train = training_set[0:len(training_set)-1]
y_train = training_set[1:len(training_set)]
X_train = np.reshape(X_train, (len(X_train), 1, 1))

regressor = create_model()
regressor.fit(X_train, y_train, batch_size=5, epochs=20,callbacks=[LogRunMetrics()])



# write scaler as a pickle file for later
'''scaler_name = 'scaler.save'
with open(scaler_name, 'wb') as file:
    joblib.dump(value=sc, filename = os.path.join(OUTPUT_DIR, scaler_name))
run.upload_file(scaler_name, os.path.join(OUTPUT_DIR, scaler_name))
'''

# save model for use outside the script
'''model_file_name = 'log_reg.h5'
with open(model_file_name, 'wb') as file:
    joblib.dump(value=regressor, filename=os.path.join(OUTPUT_DIR, model_file_name))
'''
# Sauvegardez votre modèle Keras sur le disque
#regressor.save(os.path.join(OUTPUT_DIR, model_file_name))


# register the model with the model management service for later use
'''run.upload_file('original_model.h5', os.path.join(OUTPUT_DIR, model_file_name))
original_model = run.register_model(model_name='nvamlcompute_deploy_model',
                                    model_path='original_model.h5')'''



# create a ./outputs/model folder in the compute target
# files saved in the "./outputs" folder are automatically uploaded into run history
os.makedirs('./outputs/model', exist_ok=True)
registered_model_name ='credit_defaults_model'

'''
print("Registering the model via MLFlow")
mlflow.keras.log_model(
    regressor,
    registered_model_name=registered_model_name,
    artifact_path=registered_model_name
)
'''

print("Registering the model via MLFlow")
mlflow.keras.log_model(
    regressor,
    registered_model_name=registered_model_name,
    artifact_path=registered_model_name
)

# Saving the model to a file
mlflow.keras.save_model(
    regressor,
    path=os.path.join(registered_model_name, "trained_model"),
)


'''
# serialize NN architecture to JSON
model_json = regressor.to_json()
# save model JSON
with open('./outputs/model/model.json', 'w') as f:
    f.write(model_json)
# save model weights
regressor.save_weights('./outputs/model/model.h5')
print("model saved in ./outputs/model folder")
'''


'''
scaler_name = 'scaler.save'
with open(scaler_name, 'wb') as file:
    joblib.dump(value=sc, filename = os.path.join(registered_model_name, scaler_name))
print("scaler saved in ./outputs/model folder")
'''

from azureml.core.run import Run

# get the run this was submitted from to interact with run history
run = Run.get_context()

# write scaler as a pickle file for later
scaler_name = 'scaler.save'
with open(scaler_name, 'wb') as file:
    joblib.dump(value=sc, filename = 'scaler.save')
run.upload_file(scaler_name, 'scaler.save')

from azure.storage.blob import BlobServiceClient
connectionString = "DefaultEndpointsProtocol=https;AccountName=nvastockage;AccountKey=vCMtCTE4Au52QUOtOd25VJSWeWoPzBQwDgSa2bpV9gbaL+h8k3yyFtCixV3zeUozZAeKr5NmwzmU+AStq6d7Lg==;EndpointSuffix=core.windows.net"
# Connexion au compte de stockage et recuperation du contenu du blob
blob_service_client = BlobServiceClient.from_connection_string(connectionString)
blob_client = blob_service_client.get_blob_client('nvabucket', scaler_name)
with open(file='scaler.save', mode="rb") as data:
    blob_client.upload_blob(data)

