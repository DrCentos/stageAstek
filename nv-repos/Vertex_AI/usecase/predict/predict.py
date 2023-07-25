from keras.models import load_model
from os import path, getenv
import numpy as np
import joblib
from flask import Flask, jsonify, request
from google.cloud import storage
import subprocess
from os import path, getenv, listdir, environ
BUCKET_PATH = getenv('AIP_STORAGE_URI')
MODEL_PATH = path.join(BUCKET_PATH, 'rnn-bitcoin-pred')
SC_PATH = path.join(BUCKET_PATH, 'scaler.bin')

app = Flask(__name__)

'''
# Open the model
regressor = load_model(MODEL_PATH)
# Open the Scaler
storage_client=storage.Client("glossy-precinct-371813");
bucket = storage_client.get_bucket('nvallot_bucket')
blob = bucket.blob("mysc.bin")
blob.download_to_filename('std_scaler2.bin')
sc=joblib.load('std_scaler2.bin')'''


# Loading the model from the pickle file
command = f"gcloud storage cp -r {BUCKET_PATH} ."
subprocess.run(command, shell=True, stdout=subprocess.PIPE)
#print(listdir("./artifacts"))

app = Flask(__name__)

regressor = load_model("./artifacts/rnn-bitcoin-pred")
sc=joblib.load("./artifacts/scaler.bin")

def predict(inputs):
    inputs = np.reshape(inputs, (len(inputs), 1))
    inputs = sc.transform(inputs)
    inputs = np.reshape(inputs, (len(inputs), 1, 1))
    pred = regressor.predict(inputs).tolist()
    pred = sc.inverse_transform(pred)

    return pred

@app.route(getenv('AIP_HEALTH_ROUTE'), methods=['GET'])
def health_check():
   return {"status": "healthy"}


@app.route(getenv('AIP_PREDICT_ROUTE'), methods=['POST'])
def add_income():
    request_json = request.json
    request_instances = request_json['instances']
    response = predict(request_instances)
    response = response.tolist()
    return jsonify({"predictions": response})


if __name__ == '__main__':
   app.run(host='0.0.0.0', port=int(getenv("AIP_HTTP_PORT")))