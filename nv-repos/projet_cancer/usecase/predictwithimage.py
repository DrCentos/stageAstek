import requests
import google.cloud.storage
from google.cloud import storage
import io
from PIL import Image
from google.cloud import aiplatform


from google.cloud import aiplatform

project_id = 'glossy-precinct-371813'
endpoint_id = '7234760122487013376	'
loca = 'europe-west1'


headers = {'Authorization': 'Bearer YOUR_ACCESS_TOKEN'}

client_options = {"api_endpoint": "europe-west1-aiplatform.googleapis.com"}
client = aiplatform.gapic.EndpointServiceClient(client_options=client_options)

endpoint = client.get_endpoint(name=f"projects/{project_id}/locations/{loca}/endpoints/{endpoint_id}")

print(endpoint.deployed_models[0].model)

client = storage.Client()

bucket = client.get_bucket("nvabucket")  # client.get_bucket(BUCKET_IMAGE)
blobs = bucket.list_blobs(prefix='{}/'.format("pred"), delimiter='/')


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
            response = requests.post('https://{}'.format(endpoint.deployed_models[0].model), headers=headers, files={'file': img.tobytes()})
            print(response.json())


import os
url = 'https://{}'.format(endpoint.deployed_models[0].model)
path_img = "melanoma_0.jpg"
with open(path_img, 'rb') as img:
  name_img= os.path.basename(path_img)
  files= {'image': (name_img, img, 'multipart/form-data',{'Expires': '0'}) }
  with requests.Session() as s:
    r = s.post(url, files=files)
    print(r.status_code)








