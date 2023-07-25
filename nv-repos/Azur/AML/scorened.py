import json
import numpy as np
import os
from keras.models import model_from_json
import mlflow
import mlflow.keras
import mlflow.tensorflow
from azureml.core.model import Model
import joblib
import pickle
from azure.storage.blob import BlobServiceClient
from os import path


def download_blob_to_file(blob_service_client: BlobServiceClient, container_name):
    current_folder = path.dirname(path.abspath(__file__))
    blob_client = blob_service_client.get_blob_client(container=container_name, blob="scaler.save")
    with open(file=os.path.join(current_folder, 'scaler.save'), mode="wb") as sample_blob:
        download_stream = blob_client.download_blob()
        sample_blob.write(download_stream.readall())

def init():
    global model
    global scaler

    model_root = Model.get_model_path('credit_defaults_model')
    print('model_root : ',model_root)
    model = mlflow.keras.load_model(model_root)

    '''
    #code si on a le scaler en local 
    new_path = (os.path.join(model_root, "scaler.save"))
    print('new_path :', new_path)
    scaler = joblib.load(new_path)
    '''

    current_folder = path.dirname(path.abspath(__file__))
    blob_service_client = BlobServiceClient.from_connection_string(
        "DefaultEndpointsProtocol=https;AccountName=nvastockage;AccountKey=vCMtCTE4Au52QUOtOd25VJSWeWoPzBQwDgSa2bpV9gbaL+h8k3yyFtCixV3zeUozZAeKr5NmwzmU+AStq6d7Lg==;EndpointSuffix=core.windows.net")
    container_client = blob_service_client.get_container_client(container= 'nvabucket')

    with open(file='scaler.save', mode="wb") as download_file:
        download_file.write(container_client.download_blob('scaler.save').readall())
    scaler = joblib.load('scaler.save')







    # load json and create model
    #json_file = open(os.path.join(model_root, 'model.json'), 'r')
    #model_json = json_file.read()
    #json_file.close()
    #model = model_from_json(model_json)
    # load weights into new model
    #model.load_weights(os.path.join(model_root, "model.h5"))
    #model.compile(optimizer='adam', loss='mean_squared_error')

    #scaler = joblib.load('scaler.save')


def run(raw_data):
    raw_data = raw_data.replace('"', '')
    data = scaler.transform([[float(i)] for i in raw_data.split(",")])
    predictions = model.predict(data).flatten().tolist()
    predictions = scaler.inverse_transform([[i] for i in predictions]).flatten().tolist()
    return {'predictions': predictions}


if __name__ == '__main__':
    init()
    print(run("0.1, 0.2, 0.3, 0.4"))
