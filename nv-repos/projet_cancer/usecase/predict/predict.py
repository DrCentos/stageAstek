from keras.models import load_model
from os import path, getenv
import numpy as np
import joblib
from flask import Flask, jsonify, request
from google.cloud import storage
import subprocess
from os import path, getenv, listdir, environ
from google.cloud import aiplatform
from google.protobuf import json_format
from PIL import Image
import google.cloud.storage
import base64
import io
import os
import logging

BUCKET_PATH = getenv('AIP_STORAGE_URI')
MODEL_PATH = path.join(BUCKET_PATH, 'model')
IMAGE_PATH= getenv("AIP_BUCKET_PRED")
BUCKET_IMAGE = (IMAGE_PATH.split("//")[1]).split("/")[0]
FILE_IMAGE = (IMAGE_PATH.split("//")[1]).split("/")[1]
app = Flask(__name__)


print("listdir before bucket : ", os.listdir('./'))

# Loading the model from the pickle file
command = f"gcloud storage cp -r {BUCKET_PATH} ."
subprocess.run(command, shell=True, stdout=subprocess.PIPE)
#print(listdir("./artifacts"))

app = Flask(__name__)


model = load_model("./artifacts/model")





"""
command = f"mkdir pred"
subprocess.run(command, shell=True, stdout=subprocess.PIPE)

print("listdir apres mkdir: ", os.listdir('./'))


print("BUCKET_PATH :", BUCKET_PATH)
#command = "gsutil -m cp -r gs://nvabucket/pred/* ./pred"
command = f"gcloud storage cp -r gs://nvabucket/pred/* ."
subprocess.run(command, shell=True, stdout=subprocess.PIPE)

print("IMAGE_PATH :", IMAGE_PATH)
print("listdir after bucket : ", os.listdir('./'))

print("listdir pred : ", os.listdir('./pred'))

pred_dir = './'
file = []
for filename in os.listdir(pred_dir):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        # Si le fichier est une image, ouvrir le fichier et ajouter son contenu au tableau
        with open(os.path.join(pred_dir, filename), "rb") as f:
            image_content = f.read()
            file.append(image_content)

print(f"Nombre d'images : {len(file)}")
"""

def predict(request):



    """
    file = []
    path = './artifacts/{}'.format(FILE_IMAGE)
    for root, dirs, files in os.walk(path):
        for name in files:
            logging.info('file_path :', file_path)
            file_path = os.path.join(root, name)
            if os.path.isfile(file_path):
                file.append(file_path)
    """




    """
    ims_malignant = []
    for blob in file:
        print("dans la boucle malignant load")
        with Image.open(blob) as img:
            img = img.resize((224, 224))
            ims_malignant.append(np.array(img))
    print('ims_malignant :', ims_malignant)
    logging.info('ims_malignant :', ims_malignant)
    print("start")
    

    # Prétraitement de l'image
    client = storage.Client()
    print("BUCKET_IMAGE :", BUCKET_IMAGE)
    bucket = client.get_bucket("nvabucket")#client.get_bucket(BUCKET_IMAGE)
    print("FILE_IMAGE :", FILE_IMAGE)
    blobs = bucket.list_blobs(prefix='{}/'.format(FILE_IMAGE), delimiter='/')
    
    file = []
    for blob in blobs:
        if blob.name.endswith('.jpg'):
            print("blob.name :", blob.name)
            file.append(blob.name)

    ims_malignant = []
    for blob in file:
        print("dans la boucle malegnant load")
        blobs = bucket.blob(blob)
        with io.BytesIO(blobs.download_as_bytes()) as f:
            with Image.open(f) as img:
                img = img.resize((224, 224))
                ims_malignant.append(np.array(img))
    """
    # version image
    ims_malignant = []
    image_file = request
    print('image_file :', image_file)
    img = Image.open(io.BytesIO(image_file))
    img = img.resize((224, 224))
    ims_malignant.append(np.array(img))


    X_test = np.array(ims_malignant, dtype='uint8')
    X_test = X_test / 255
    logging.info('X_test :', X_test)
    y_pred = model.predict(X_test)
    print(y_pred[0])

    if (y_pred[0][0] > y_pred[0][1]):
        print("benign")
        pred = "benign"
    else:
        print("malignant")
        pred = "malignant"
    print("end")

    #for blob in file:
    #    blobs = bucket.blob(blob)
        # Supprimer le fichier il faudra sans doute le faire autre pars car la j'ai pas le perms
    #    blobs.delete()
    return pred


@app.route(getenv('AIP_HEALTH_ROUTE'), methods=['GET'])
def health_check():
   return {"status": "healthy"}


@app.route(getenv('AIP_PREDICT_ROUTE'), methods=['POST'])
def add_income():
    request_json = request.get_json()
    print('request_json :', request_json)
    if 'instances' not in request_json:
        return 'Error: Request does not contain instances'
    instances = request_json['instances']
    if not instances:
        return 'Error: instances is empty'
    instance = instances[0]
    print('instance[0] :', instance)
    if 'image_bytes' not in instance:
        return 'Error: instance does not contain image_bytes'
    image_bytes = instance['image_bytes']['b64']
    print('image_bytes :', image_bytes)
    image_data = base64.b64decode(image_bytes)
    ims_malignant = []
    image_file = image_data
    print('image_file :', image_file)
    img = Image.open(io.BytesIO(image_file))
    img = img.resize((224, 224))
    ims_malignant.append(np.array(img))
    x_test = np.array(ims_malignant, dtype='uint8')
    x_test = x_test / 255
    print('x_test :', x_test)
    y_pred = model.predict(x_test)
    print(y_pred[0])
    #response = predict(image_data)

    if (y_pred[0][0] > y_pred[0][1]):
        print("benign")
        pred = "benign"
    else:
        print("malignant")
        pred = "malignant"
    print("end")
    response = y_pred.tolist()
    return jsonify({"predictions": response, "resultat" : pred})

    #on va tester de prendre en entré le chemin d'une image et non un json
    #request_json = request.json
    #request_instances = request_json['instances']
    #response = predict()
    #return jsonify({"predictions": response})


if __name__ == '__main__':
   app.run(host='0.0.0.0', port=int(getenv("AIP_HTTP_PORT")),debug=True)