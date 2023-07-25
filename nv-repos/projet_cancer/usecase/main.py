from os import path, getenv, makedirs, listdir
from google.cloud import storage
from google.cloud import aiplatform


def getendpointbucket():
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(RETRAIN_BUCKET)
    blob = bucket.blob(dest_endpoint + "endpointid.txt")
    # Télécharger le contenu du fichier
    content = blob.download_as_string()
    endpoint_id = content.decode('utf-8')

    # Afficher le contenu du fichier
    print(endpoint_id)
    #endpoint_id="133232312"
    return  endpoint_id

def check_and_delete_endpoint(endpoint_id):

    try:
        # Vérifiez si l'endpoint existe

        aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=bucket_staging)
        endpoint = aiplatform.Endpoint(endpoint_id)
        print(endpoint)
        if endpoint:
            # Si l'endpoint existe, supprimez-le
            client.delete_endpoint(endpoint_id)

            print(f"L'endpoint avec l'ID {endpoint_id} a été supprimé.")
        else:
            print(f"L'endpoint avec l'ID {endpoint_id} n'existe pas.")

    except NotFound:
        # L'endpoint n'existe pas
        print(f"L'endpoint avec l'ID {endpoint_id} n'existe pas.")

    except Exception as e:
        # Gérez les exceptions possibles lors de l'interaction avec l'API de Vertex AI
        print(f"Une erreur s'est produite : {str(e)}")

    print("fin check endpoint")

RETRAIN_BUCKET = "nvabucket"
REGION = 'europe-west1'
PROJECT_ID = 'glossy-precinct-371813'
bucket_staging = 'gs://nvabucket/model' # Should be same as AIP_STORAGE_URI specified in docker file
container_uri='europe-west1-docker.pkg.dev/glossy-precinct-371813/nvrep/nvtrain:latest'
model_serving_container_image_uri='europe-west1-docker.pkg.dev/glossy-precinct-371813/nvrep/nvpred:latest'
display_name='Custom Job from Code'
client = storage.Client(project=PROJECT_ID)
bucket = client.bucket(RETRAIN_BUCKET)
aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=bucket_staging)

# Create CustomContainerTrainingJob and start run with some service account
job = aiplatform.CustomContainerTrainingJob(
display_name=display_name,
container_uri=container_uri,
model_serving_container_image_uri=model_serving_container_image_uri,
)
#voir quel machine est assez grande
model = job.run( model_display_name=display_name, machine_type='n1-standard-16')

# Finally deploy the model to endpoint
endpoint = model.deploy(
deployed_model_display_name=display_name, sync=True
)

# Get the ID of the deployed model
deployed_model_id = endpoint.resource_name.split("/")[-1]
print(f"Deployed model ID: {deployed_model_id}")
#sauvegarder dans un bucket au format .txt en overwrite

dest_endpoint = "endpoint/"
# Get the ID of the deployed model
endpoint_id = deployed_model_id

# supprimer l'ancien endpoint s'il existe
old_endpoint_id = getendpointbucket()
check_and_delete_endpoint(old_endpoint_id)

#sauvegarder le nouveau endpoint id dans un bucket au format .txt en overwrite ou var d'env
blob = bucket.blob(dest_endpoint + "endpointid.txt")
blob.upload_from_string(endpoint_id)
