import os
from google.cloud import storage
from google.api_core.exceptions import NotFound
"""
original_string = "gs://nvabucket/model"
suffix = original_string.split("//")[1]
bucket_name  = suffix.split("/")[0]
folder_name = suffix.split("/")[1] + "/"

storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)

delimiter="/"

blobs = storage_client.list_blobs(bucket_name, prefix=folder_name)#,delimiter=delimiter)
blob_names = [blob.name for blob in blobs]
"""

"""
if delimiter:
    print("Prefixes:")
    for prefix in blobs.prefixes:
        print(prefix)
"""
"""
if len(blob_names) > 1:
    print("on charge un model déjà existant")
    latest_model_folder = sorted(blob_names)[-1]
    #folder_model = latest_model_folder.split("/")[1]
    folder_model = original_string.replace(MODEL_PATH, latest_model_folder.split("/")[1])
    #folder_model = latest_model_folder.split("/")[1] + "/" + latest_model_folder.split("/")[2] + "/" + latest_model_folder.split("/")[3] + "/" + latest_model_folder.split("/")[4] + "/"
    print(folder_model)
else:
    print("pas de model")
"""
PROJECT_ID = 'glossy-precinct-371813'
def getendpointormodelfrombucket(dest,txt):
    try:
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket("nvabucket")
        blob = bucket.blob(dest + txt)
        # Télécharger le contenu du fichier
        content = blob.download_as_string()
        endpoint_id = content.decode('utf-8')

        # Afficher le contenu du fichier
        print(endpoint_id)
    except NotFound:
        print(f"Le fichier ou la bucket {dest + txt} demandé n'existe pas.")
        endpoint_id = "0000"
    except Exception as e:
        # Gérez les exceptions possibles lors de l'interaction avec l'API de Vertex AI
        print(f"Une erreur s'est produite : {str(e)}")
        endpoint_id = "0000"
    #endpoint_id="133232312"
    return  endpoint_id


dest_model = "modelid/"
getendpointormodelfrombucket(dest_model,"aaak.txt")
print("afterere")

#blob = bucket.blob(dest + "endpointid.txt")
#blob.upload_from_string('4980708508988080128')

"""


blob = bucket.blob(dest + "endpointid.txt")

# Télécharger le contenu du fichier
content = blob.download_as_string()
content_str = content.decode('utf-8')
# Afficher le contenu du fichier
print(content_str)
"""