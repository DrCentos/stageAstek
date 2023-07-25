import os
import glob

def afficher_elements_repertoire_courant():
    # Obtenir le chemin du répertoire courant
    chemin_repertoire_courant = os.getcwd()

    # Utiliser glob pour lister les fichiers et dossiers dans le répertoire courant
    elements_courants = glob.glob('*')

    # Afficher les éléments du répertoire courant
    for element in elements_courants:
        print(element)

# Appeler la fonction pour afficher les éléments du répertoire courant
#afficher_elements_repertoire_courant()

from azure.storage.blob import BlobServiceClient
connection_string = "DefaultEndpointsProtocol=https;AccountName=nvastockage;AccountKey=BhpwOnQK0S4VBbksgH6ptRk7Jo9Db14tXCFoEWA6cp1iKQ6VdcQ07pcB0+ew4lDQ6lx5b314i7vX+AStPgJcTg==;EndpointSuffix=core.windows.net"
nom_conteneur = "nvabucket"
model_name = 'credit_defaults_model'
file_name = "endpoint.txt"
def get_endpoint_id_from_bucket():
    endpoint_file = "endpoint.txt"


    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    # Récupérer le conteneur spécifié
    container_client = blob_service_client.get_container_client(nom_conteneur)

    # Get a reference to the blob (file) in the container
    blob_client = container_client.get_blob_client(endpoint_file)

    # Download the contents of the blob as a byte array
    content = blob_client.download_blob().readall()

    # Decode the byte array into a string assuming it's UTF-8 encoded
    endpoint_name = content.decode("utf-8")

    return endpoint_name

from azure.identity import DefaultAzureCredential
from azure.mgmt.machinelearningservices import MachineLearningServicesMgmtClient
def delete_endpoint_from_azure():


    subscription_id = "d9414828-c496-44ab-b6f9-e903bce40957"
    resource_group = "mlops"
    workspace_name = "nvml"
    endpoint_name = get_endpoint_id_from_bucket()
    print("endpoint_name :", endpoint_name)

    client = MachineLearningServicesMgmtClient(
        credential=DefaultAzureCredential(),
        subscription_id=subscription_id,
    )

    client.online_endpoints.begin_delete(
        resource_group_name=resource_group,
        workspace_name=workspace_name,
        endpoint_name=endpoint_name,
    ).result()
    return


delete_endpoint_from_azure()