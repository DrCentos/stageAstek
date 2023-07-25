import os
from google.cloud import aiplatform, storage

def predict_on_uploaded_image(file_name, project_id, region, endpoint_id):
    # Charger le fichier depuis le bucket GCP
    storage_client = storage.Client()
    bucket = storage_client.bucket('nvabucket')
    blob = bucket.blob(file_name)
    image_content = blob.download_as_bytes()

    # Préparer les données pour la prédiction
    instance = [1,2,3]

    # Récupérer la référence à l'endpoint
    endpoint = aiplatform.Endpoint(endpoint_id)

    # Effectuer la prédiction
    response = endpoint.predict(instances=[instance]).predictions

    # Afficher la réponse de la prédiction
    print(response)

def upload_trigger(event, context):
    # Récupérer le nom du fichier depuis l'événement
    file_name = event['name']

    # Récupérer les variables d'environnement nécessaires
    #bucket = 'gs://nvabucket/model'  # Should be same as AIP_STORAGE_URI specified in docker file
    region = 'europe-west1'
    project_id = 'glossy-precinct-371813'
    endpoint_id = '2061250050545156096'

    # Effectuer la prédiction sur le fichier
    predict_on_uploaded_image(file_name, project_id, region, endpoint_id)
