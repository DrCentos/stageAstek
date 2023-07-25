from google.api_core.exceptions import NotFound
from google.cloud import aiplatform
from google.api_core.exceptions import NotFound
import google.cloud.storage
from google.cloud import storage

bucket_staging = 'gs://nvabucket/model'  # Should be same as AIP_STORAGE_URI specified in docker file
REGION = 'europe-west1'
PROJECT_ID = 'glossy-precinct-371813'
def check_and_delete_endpoint(endpoint_id):

    try:
        # Vérifiez si l'endpoint existe
        client = storage.Client(project=PROJECT_ID)
        aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=bucket_staging)
        endpoint = aiplatform.Endpoint(endpoint_id)
        print(endpoint)
        if endpoint:
            # Si l'endpoint existe, supprimez-le
            endpoint.undeploy_all()
            endpoint.delete()

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
def check_and_delete_model(model_id):

    try:
        # Vérifiez si le modèle existe
        aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=bucket_staging)
        model = aiplatform.Model(model_name=model_id)

        if model:
            # Si le modèle existe, supprimez-le
            model.delete()
            print(f"Le modèle avec l'ID {model_id} a été supprimé.")
        else:
            print(f"Le modèle avec l'ID {model_id} n'existe pas.")

    except NotFound:
        # Le modèle n'existe pas
        print(f"Le modèle avec l'ID {model_id} n'existe pas.")

    except Exception as e:
        # Gérez les exceptions possibles lors de l'interaction avec l'API AI Platform
        print(f"Une erreur s'est produite : {str(e)}")

    print("Fin de la vérification du modèle")

#passer l'endpoint du bucket
def check_endpoint(endpoint_id):
    try:
        # Vérifiez si l'endpoint existe
        client = storage.Client(project=PROJECT_ID)
        aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=bucket_staging)
        endpoint = aiplatform.Endpoint(endpoint_id)
        print(endpoint)
        if endpoint:
            print(f"L'endpoint avec l'ID {endpoint_id} a été supprimé.")
            return endpoint_id
        else:
            print(f"L'endpoint avec l'ID {endpoint_id} n'existe pas.")
            return "000"
    except NotFound:
        # L'endpoint n'existe pas
        print(f"L'endpoint avec l'ID {endpoint_id} n'existe pas.")

    except Exception as e:
        # Gérez les exceptions possibles lors de l'interaction avec l'API de Vertex AI
        print(f"Une erreur s'est produite : {str(e)}")

    print("fin check endpoint")

#prend en entré l'id du model recup dans le bucket
def check_and_deploy_model(model_id):

    try:
        # Vérifiez si le modèle existe
        aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=bucket_staging)
        model = aiplatform.Model(model_name=model_id)

        if model:
            dest_endpoint = "endpoint/"
            dest_model = "modelid/"
            txt_endpoint = "endpointid.txt"
            txt_model = "modelid.txt"
            # vérifier s'il n'y a pas deja un endpoint avec ce model
            old_endpoint_id = getendpointormodelfrombucket(dest_endpoint, txt_endpoint)
            #si lendpoint existe déja
            if(check_endpoint(old_endpoint_id)!="000"):
                print("l'endpoint existe déjà avec l'id : ",old_endpoint_id)
                flash("l'endpoint existe déjà avec l'id : ",old_endpoint_id)
                #TODO : voir pour faire une redirection ou mettre a jour la page avec ce message
            else:
                endpoint = model.deploy(
                    deployed_model_display_name=display_name, machine_type="n1-standard-16", sync=True
                )
                # Get the ID of the deployed model
                deployed_model_id = endpoint.resource_name.split("/")[-1]
                print(f"Deployed model ID: {deployed_model_id}")

                # sauvegarder dans un bucket au format .txt en overwrite
                # Get the ID of the deployed model
                endpoint_id = deployed_model_id
                # sauvegarder le nouveau endpoint id dans un bucket au format .txt en overwrite ou var d'env
                blob = bucket.blob(dest_endpoint + txt_endpoint)
                blob.upload_from_string(endpoint_id)

                flash("fin du deploy de l'endpoint")
                print("end")
        else:
            print(f"Le modèle avec l'ID {model_id} n'existe pas.")
            flash(f"Le modèle avec l'ID {model_id} n'existe pas.")

    except NotFound:
        # Le modèle n'existe pas
        print(f"Le modèle avec l'ID {model_id} n'existe pas.")
        flash(f"Le modèle avec l'ID {model_id} n'existe pas.")
        # TODO : voir pour faire une redirection ou mettre a jour la page avec ce message

    except Exception as e:
        # Gérez les exceptions possibles lors de l'interaction avec l'API AI Platform
        print(f"Une erreur s'est produite : {str(e)}")

    print("Fin de la vérification du modèle")


def delete_endpoint(endpoint_id):

    try:
        # Vérifiez si l'endpoint existe
        client = storage.Client(project=PROJECT_ID)
        aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=bucket_staging)
        endpoint = aiplatform.Endpoint(endpoint_id)
        print(endpoint)
        if endpoint:
            # Si l'endpoint existe, supprimez-le
            endpoint.undeploy_all()
            endpoint.delete()

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


check_and_delete_endpoint("325816081635606528")
# Utilisation de la fonction pour vérifier et supprimer un modèle
model_id = "5329209714527961088"  # Remplacez par l'ID du modèle que vous souhaitez vérifier et supprimer
check_and_delete_model(model_id)
