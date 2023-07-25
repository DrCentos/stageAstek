from google.cloud import aiplatform  # Importez votre bibliothèque ou SDK pour interagir avec l'API de Vertex AI
import google.cloud.storage
from google.cloud import storage
from google.api_core.exceptions import NotFound

RETRAIN_BUCKET = "nvabucket"
PROJECT_ID = 'glossy-precinct-371813'
dest_endpoint = "endpoint/"
bucket_staging = 'gs://nvabucket/model'
REGION = 'europe-west1'

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

# Exemple d'utilisation
endpoint_id = getendpointbucket
check_and_delete_endpoint(endpoint_id)
