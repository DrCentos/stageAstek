from google.cloud import aiplatform
from PIL import Image
import numpy as np
import json
import google.cloud.storage
from google.cloud import storage
from tensorflow.keras.models import load_model
import io
import base64
"""
print("start")
# Prétraitement de l'image
client = storage.Client()
bucket = client.get_bucket("nvabucket")
blobs = bucket.list_blobs(prefix='{}/'.format("pred"), delimiter='/')
file = []
for blob in blobs:
    if blob.name.endswith('.jpg'):
        file.append(blob.name)

ims_malignant = []
for blob in file:
    print("dans la boucle malegnant load")
    blobs = bucket.blob(blob)
    with io.BytesIO(blobs.download_as_bytes()) as f:
        with Image.open(f) as img:
            img = img.resize((224, 224))
            ims_malignant.append(np.array(img))

X_test = np.array(ims_malignant, dtype='uint8')
X_test = X_test / 255

model = load_model("model")
print("load ok")
y_pred = model.predict(X_test)
print(y_pred[0])

if(y_pred[0][0] > y_pred[0][1]):
    print("benin")
else:
    print("malignant")
print("end")

#je supprime les photos dans le bucket possiblement faudra le faire autre part apres
# Sélectionner le fichier à supprimer
for blob in file:
    blobs = bucket.blob(blob)
    # Supprimer le fichier
    blobs.delete()

"""

client = storage.Client()

bucket = client.get_bucket("nvabucket")  # client.get_bucket(BUCKET_IMAGE)
blobs = bucket.list_blobs(prefix='{}/'.format("pred"), delimiter='/')


file = []
for blob in blobs:
    if blob.name.endswith('.jpg'):
        print("blob.name :", blob.name)
        file.append(blob.name)


REGION = 'europe-west1'
PROJECT_ID = 'glossy-precinct-371813'

staging_bucket = 'gs://nvabucket/model'  # Should be same as AIP_STORAGE_URI specified in docker file
aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=staging_bucket)




# Chargement du modèle déployé
#récupérer le numéro du endpoint
endpoint_id = "2528498511884845056"
endpoint = aiplatform.Endpoint(endpoint_id)

with open("melanoma_0.jpg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

payload = {
  "instances": [
    {
      "image_bytes": {
        "b64": encoded_string
      }
    }
  ]
}


prediction = endpoint.predict(instances=payload["instances"])

if (prediction[0][0][0] > prediction[0][0][1]):
    print("benign")
    pred = "benign"
else:
    print("malignant")
    pred = "malignant"
print("end")

print(prediction[0][0])

"""
for blob in file:
    print("dans la boucle malegnant load")
    blobs = bucket.blob(blob)
    with io.BytesIO(blobs.download_as_bytes()) as f:
        with Image.open(f) as img:
            prediction = endpoint.predict(instances=payload)
"""
#{"instances":[1,2,3]}

"""
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value, ListValue
from google.protobuf.struct_pb2 import Struct
# Envoi de la requête de prédiction
instances = [img_array.tolist()]
inputs = Struct(fields={"instances": Value(list_value=ListValue(values=instances))})
response = endpoint.predict(instances=inputs)
print(response)"""