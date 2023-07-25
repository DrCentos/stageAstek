from flask import Flask, render_template, request,jsonify, url_for
from google.cloud import storage
import base64
from google.cloud import aiplatform
import google.cloud.storage
from google.cloud import storage
from google.cloud import aiplatform
from os import path, getenv, makedirs, listdir
import threading
from google.cloud import tasks_v2
from google.protobuf import timestamp_pb2
from google.api_core.exceptions import NotFound
from datetime import timedelta

RETRAIN_BUCKET = "nvabucket"
REGION = 'europe-west1'
PROJECT_ID = 'glossy-precinct-371813'
app = Flask(__name__, template_folder='template')
bucket_staging = 'gs://nvabucket/model'  # Should be same as AIP_STORAGE_URI specified in docker file
container_uri = 'europe-west1-docker.pkg.dev/glossy-precinct-371813/nvrep/nvtrain:latest'
model_serving_container_image_uri = 'europe-west1-docker.pkg.dev/glossy-precinct-371813/nvrep/nvpred:latest'
display_name = 'Custom Job from Code'
aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=bucket_staging)
dest_endpoint = "endpoint/"
dest_model = "modelid/"


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
    #gestion des erreurs si le chemin demandé n'existe pas
    except NotFound:
        print(f"Le fichier ou la bucket {dest + txt} demandé n'existe pas.")
        endpoint_id = "000"
    except Exception as e:
        # Gérez les exceptions possibles lors de l'interaction avec l'API de Vertex AI
        print(f"Une erreur s'est produite : {str(e)}")
        endpoint_id = "000"
    #endpoint_id="133232312"
    return  endpoint_id

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



def check_endpoint(endpoint_id):
    try:
        # Vérifiez si l'endpoint existe
        client = storage.Client(project=PROJECT_ID)
        aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=bucket_staging)
        endpoint = aiplatform.Endpoint(endpoint_id)
        print(endpoint)
        if endpoint:
            print(f"L'endpoint avec l'ID {endpoint_id} existe.")
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


def deployendpoint(model, dest_endpoint, txt_endpoint):
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
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(RETRAIN_BUCKET)
    blob = bucket.blob(dest_endpoint + txt_endpoint)
    blob.upload_from_string(endpoint_id)

    print("end")

#penser à appeler avant de lancer : getendpointormodelfrombucket avec comme parametre modelid/ et modelid.txt
@app.route("/deploy", methods=["POST"])
def deploy():

    try:
        dest_endpoint = "endpoint/"
        dest_model = "modelid/"
        txt_endpoint = "endpointid.txt"
        txt_model = "modelid.txt"
        model_id = getendpointormodelfrombucket(dest_model, txt_model)
        # Vérifiez si le modèle existe
        aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=bucket_staging)
        model = aiplatform.Model(model_name=model_id)

        if model:

            # vérifier s'il n'y a pas deja un endpoint avec ce model
            old_endpoint_id = getendpointormodelfrombucket(dest_endpoint, txt_endpoint)
            #si lendpoint existe déja
            if(check_endpoint(old_endpoint_id)=="000"):
                print("l'endpoint existe déjà avec l'id : ",old_endpoint_id)
                res = "un endpoint existe déjà supprimez le avant d'en recréer un"
                #TODO : voir pour faire une redirection ou mettre a jour la page avec ce message
                return render_template('deploy.html', result=res)
            else:
                print("avant le thread du deploy")
                thread = threading.Thread(target=deployendpoint,args=(model, dest_endpoint, txt_endpoint))
                thread.start()
                res = "l'endpoint a bien ete deploy"
                return render_template('deploy.html', result=res)
        else:
            res = "aucun model n'existe il faut faire un train"
            print(f"Le modèle avec l'ID {model_id} n'existe pas.")
            return render_template('deploy.html', result=res)

    except NotFound:
        # Le modèle n'existe pas
        print(f"Le modèle avec l'ID {model_id} n'existe pas.")
        res = "aucun model n'existe il faut faire un train"
        # TODO : voir pour faire une redirection ou mettre a jour la page avec ce message

    except Exception as e:
        # Gérez les exceptions possibles lors de l'interaction avec l'API AI Platform
        print(f"Une erreur s'est produite : {str(e)}")

    print("Fin de la vérification du modèle")

#penser à appeler avant de lancer : getendpointormodelfrombucket avec comme parametre endpoint/ et endpointid.txt
@app.route("/delete", methods=["POST"])
def delete():
    try:
        dest_endpoint = "endpoint/"
        txt_endpoint = "endpointid.txt"
        endpoint_id = getendpointormodelfrombucket(dest_endpoint, txt_endpoint)
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
            res = "la suppression de l'endpoint est termine"
            return render_template('deploy.html', result=res)
        else:
            print(f"L'endpoint avec l'ID {endpoint_id} n'existe pas.")
            res = "il n'y a aucun endpoint à supprimer"
            return render_template('deploy.html', result=res)

    except NotFound:
        # L'endpoint n'existe pas
        print(f"L'endpoint avec l'ID {endpoint_id} n'existe pas.")
        res = "il n'y a aucun endpoint à supprimer"
        return render_template('deploy.html', result=res)
    except Exception as e:
        # Gérez les exceptions possibles lors de l'interaction avec l'API de Vertex AI
        print(f"Une erreur s'est produite : {str(e)}")

    print("fin check endpoint")

@app.route("/trainfunc", methods=["POST"])
def trainfunc():
    from os import path, getenv, makedirs, listdir
    from google.cloud import storage
    from google.cloud import aiplatform
    RETRAIN_BUCKET = "nvabucket"
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(RETRAIN_BUCKET)
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=bucket_staging)

    # Create CustomContainerTrainingJob and start run with some service account
    # Create CustomContainerTrainingJob and start run with some service account
    job = aiplatform.CustomContainerTrainingJob(
        display_name=display_name,
        container_uri=container_uri,
        model_serving_container_image_uri=model_serving_container_image_uri,
    )
    model = job.run(model_display_name=display_name, machine_type='n1-standard-16')
    # Récupération de l'ID du modèle créé
    model_id = model.resource_name.split("/")[-1]
    print("apres le job model : ", model)
    print("apres le job model_id: ", model_id)
    # Finally deploy the model to endpoint
    endpoint = model.deploy(
        deployed_model_display_name=display_name,machine_type="n1-standard-16", sync=True
    )

    # Get the ID of the deployed model
    deployed_model_id = endpoint.resource_name.split("/")[-1]
    print(f"Deployed model ID: {deployed_model_id}")

    dest_endpoint = "endpoint/"
    dest_model = "modelid/"
    txt_endpoint = "endpointid.txt"
    txt_model = "modelid.txt"


    #supprimer l'ancien endpoint s'il existe
    old_endpoint_id = getendpointormodelfrombucket(dest_endpoint, txt_endpoint)
    check_and_delete_endpoint(old_endpoint_id)

    #supprimer l'ancien model s'il existe
    old_model_id = getendpointormodelfrombucket(dest_model, txt_model)
    check_and_delete_model(old_model_id)

    # sauvegarder dans un bucket au format .txt en overwrite
    # Get the ID of the deployed model
    endpoint_id = deployed_model_id
    # sauvegarder le nouveau endpoint id dans un bucket au format .txt en overwrite ou var d'env
    blob = bucket.blob(dest_endpoint + txt_endpoint)
    blob.upload_from_string(endpoint_id)

    #sauvegarder l'id du model registry pour pouvoir déployer avec
    # sauvegarder le nouveau model id dans un bucket au format .txt en overwrite ou var d'env
    blob = bucket.blob(dest_model + txt_model)
    blob.upload_from_string(model_id)

    print("end")
    #return render_template('retrain.html', result="terminé")
#@app.route("/retrainfunc", methods=["POST"])
def retrainfunc():

    """
    print("before wait")
    # Attendre que les fichiers soient téléchargés
    malignant_blob.reload()
    while not malignant_blob.exists():
        time.sleep(1)
        malignant_blob.reload()

    benign_blob.reload()
    while not benign_blob.exists():
        time.sleep(1)
        benign_blob.reload()
    print("after wait")
    """

    from os import path, getenv, makedirs, listdir
    from google.cloud import storage
    from google.cloud import aiplatform
    RETRAIN_BUCKET = "nvabucket"
    dest = 'retrain/'
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(RETRAIN_BUCKET)
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=bucket_staging)

    # Create CustomContainerTrainingJob and start run with some service account
    job = aiplatform.CustomContainerTrainingJob(
        display_name=display_name,
        container_uri=container_uri,
        model_serving_container_image_uri=model_serving_container_image_uri,
    )
    model = job.run(model_display_name=display_name, machine_type='n1-standard-16')
    model_id = model.resource_name.split("/")[-1]
    print("apres le job model : ", model)
    print("apres le job model_id: ", model_id)
    # Finally deploy the model to endpoint
    endpoint = model.deploy(
        deployed_model_display_name=display_name,machine_type="n1-standard-16", sync=True
    )
    print("apres l'endpoint : ", endpoint)
    # Get the ID of the deployed model
    deployed_model_id = endpoint.resource_name.split("/")[-1]
    print(f"Deployed model ID: {deployed_model_id}")

    dest_endpoint = "endpoint/"
    dest_model = "modelid/"
    txt_endpoint = "endpointid.txt"
    txt_model = "modelid.txt"


    #supprimer l'ancien endpoint s'il existe
    old_endpoint_id = getendpointormodelfrombucket(dest_endpoint, txt_endpoint)
    check_and_delete_endpoint(old_endpoint_id)

    #supprimer l'ancien model s'il existe
    old_model_id = getendpointormodelfrombucket(dest_model, txt_model)
    check_and_delete_model(old_model_id)


    endpoint_id = deployed_model_id

    #sauvegarder le nouveau endpoint id dans un bucket au format .txt en overwrite ou var d'env
    blob = bucket.blob(dest_endpoint + txt_endpoint)
    blob.upload_from_string(endpoint_id)

    #sauvegarder l'id du model registry pour pouvoir déployer avec
    # sauvegarder le nouveau model id dans un bucket au format .txt en overwrite ou var d'env
    blob = bucket.blob(dest_model + txt_model)
    blob.upload_from_string(model_id)

    print("before delete")
    #on vide les répertoires retrain/malignant et retrain/benign du bucket RETRAIN_BUCKET pour ne pas retrain avec la prochaine fois
    malignant_blobs = bucket.list_blobs(prefix= dest)
    for blob in malignant_blobs:
        blob.delete()

    # Supprimer les objets dans le répertoire benign
    benign_blobs = bucket.list_blobs(prefix= dest)
    for blob in benign_blobs:
        blob.delete()
    print("after delete")
    #return render_template('retrain.html', result="terminé")
    #TODO voir pour redireger vers une autre app.route pour afficher qu'on a fini
    #return render_template('retrain.html', result="terminé")

@app.route("/")
def template_test():

    return render_template('index.html', my_string="Wheeeee!", my_list=[0,1,2,3,4,5])



@app.route('/train', methods=['POST'])
def train():
    print("avant le thread du train")
    thread = threading.Thread(target=trainfunc)
    thread.start()
    return render_template('retrain.html', result="en cours")
    #return render_template('retrain.html', result=deployed_model_id)
    #return render_template('retrain.html', result="ok")
@app.route('/retrain', methods=['POST'])
def retrain():

    print("before send to bucket")
    print("malignant")
    malignant = request.files.getlist('malignant')

    print(malignant)
    benign = request.files.getlist('benign')
    print(benign)

    print("dans le retrain")
    print(malignant)
    # Ajouter les dossiers dans le bucket gcp
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(RETRAIN_BUCKET)
    dest = 'retrain/'

    # Ajouter les fichiers du dossier malignant
    for file in malignant:
        blob = bucket.blob(dest + file.filename)
        print("file :", file.filename)
        blob.upload_from_file(file)

    print("malignant load ok")
    # Ajouter les fichiers du dossier benign
    for file in benign:
        blob = bucket.blob(dest + file.filename)
        blob.upload_from_file(file)


    """
    print("after send to bucket")
    location = "europe-west3"
    queue_name = "nvqueu"
    # Créer une instance du client Cloud Tasks
    client = tasks_v2.CloudTasksClient()



    service_account_email = 'nvallot-service@glossy-precinct-371813.iam.gserviceaccount.com'  # Remplacez par l'e-mail de votre compte de service

    # Créer une tâche
    task_url = url_for('retrainfunc', _external=True)
    print(f"L'URL de la tâche Cloud est : {task_url}")


    parent = client.queue_path(PROJECT_ID, location, queue_name)
    print("queupath :",parent)
    task = {
        #'name': client.task_path(project, location, queue_name, task_id),
        'http_request': {
            'http_method': tasks_v2.HttpMethod.POST,
            'url': task_url,  # Remplacez par l'URL de votre file d'attente
        },
        'dispatch_deadline': timedelta(minutes=30),  # Délai maximum pour traiter la tâche
    }

    request2 = tasks_v2.CreateTaskRequest(parent=parent, task=task)

    # Envoyer la tâche à Cloud Tasks
    response = client.create_task(request2)
    """
    thread = threading.Thread(target=retrainfunc)
    thread.start()
    return render_template('retrain.html', result="en cours")
    #return render_template('retrain.html', result=deployed_model_id)
    #return render_template('retrain.html', result="ok")

@app.route('/upload', methods=['POST'])
def upload():
    #endpoint_id = "7266285319878606848"
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(RETRAIN_BUCKET)
    blob = bucket.blob(dest_endpoint + "endpointid.txt")
    # Télécharger le contenu du fichier
    content = blob.download_as_string()
    endpoint_id = content.decode('utf-8')
    # Afficher le contenu du fichier
    print(endpoint_id)
    endpoint = aiplatform.Endpoint(endpoint_id)

    # Récupérer le fichier envoyé dans la requête POST
    file = request.files['file']
    print(file)
    encoded_string = base64.b64encode(file.read()).decode('utf-8')

    payload = {
        "instances": [
            {
                "image_bytes": {
                    "b64": encoded_string
                }
            }
        ]
    }

    prediction = endpoint.predict(instances=payload["instances"])[0][0]
    print('prediction :', prediction)
    if prediction[0] > prediction[1]:
        result = "benign à "+ str((prediction[0])*100) +"%"
    else:
        result = "malignant à "+ str((prediction[1])*100) +"%"

    return render_template('result.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
