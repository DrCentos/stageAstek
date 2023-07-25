import requests
import google.cloud.storage
from google.cloud import storage
import io
from PIL import Image
from google.cloud import aiplatform

client = storage.Client()

bucket = client.get_bucket("nvabucket")
blobs = bucket.list_blobs(prefix='{}/'.format("pred"), delimiter='/')
for blob in blobs:
    print(blob.name)