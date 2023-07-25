import sys
from os import path

current_folder = path.dirname(path.abspath(__file__))
import subprocess
subprocess.check_call([sys.executable, "-m", "pip","install","--upgrade","mlflow", "-t", current_folder])



import numpy as np
#le pb vien de mlflow les 2 lignes la passent pas
#import mlflow
import mlflow.keras as mlk

from azureml.core.model import Model
from PIL import Image

import base64
import io
import json


def init():
    global model

    model_root = Model.get_model_path('credit_defaults_model')
    print('model_root : ',model_root)
    model = mlk.load_model(model_root)



def predict(request):

    # version image
    ims_malignant = []
    image_file = request
    print('image_file :', image_file)
    img = Image.open(io.BytesIO(image_file))
    img = img.resize((224, 224))
    ims_malignant.append(np.array(img))


    X_test = np.array(ims_malignant, dtype='uint8')
    X_test = X_test / 255
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

def run(raw_data):
    print('request_json :', raw_data)
    raw_data = json.loads(raw_data)
    if 'instances' not in raw_data:
        return 'Error: Request does not contain instances'
    instances = raw_data['instances']
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
    return {'predictions': response}

"""
if __name__ == '__main__':
    init()
    print(model)
    print("end")
"""
