#az webapp log deployment show -n nvapp -g mlops
from flask import Flask, render_template, request,jsonify, url_for
import base64
import threading
import json
import time

from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration,
)

import os
# Authentication package
from azure.ai.ml.entities import AmlCompute
from azure.ai.ml.entities import Environment
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import logging

from azure.storage.blob import BlobServiceClient

import os
import glob

#TODO IL FAUT LA VERSION 2.0.0b1
from azure.mgmt.machinelearningservices import MachineLearningServicesMgmtClient

def afficher_elements_repertoire_courant():
    # Obtenir le chemin du répertoire courant
    chemin_repertoire_courant = os.getcwd()

    # Utiliser glob pour lister les fichiers et dossiers dans le répertoire courant
    elements_courants = glob.glob('*')

    # Afficher les éléments du répertoire courant
    for element in elements_courants:
        print(element)

# Appeler la fonction pour afficher les éléments du répertoire courant
afficher_elements_repertoire_courant()


connection_string = "DefaultEndpointsProtocol=https;AccountName=nvastockage;AccountKey=BhpwOnQK0S4VBbksgH6ptRk7Jo9Db14tXCFoEWA6cp1iKQ6VdcQ07pcB0+ew4lDQ6lx5b314i7vX+AStPgJcTg==;EndpointSuffix=core.windows.net"
nom_conteneur = "nvabucket"
model_name = 'credit_defaults_model'
file_name = "endpoint.txt"

score_code = """
import sys
from os import path

current_folder = path.dirname(path.abspath(__file__))
import subprocess
subprocess.check_call([sys.executable, "-m", "pip","install","--upgrade","mlflow", "-t", current_folder])



import numpy as np
#le pb vien de mlflow les 2 lignes la passent pas
#import mlflow
import mlflow.keras as mlk

from azureml.core.model import Model
from PIL import Image

import base64
import io
import json


def init():
    global model

    model_root = Model.get_model_path('credit_defaults_model')
    print('model_root : ',model_root)
    model = mlk.load_model(model_root)



def predict(request):

    # version image
    ims_malignant = []
    image_file = request
    print('image_file :', image_file)
    img = Image.open(io.BytesIO(image_file))
    img = img.resize((224, 224))
    ims_malignant.append(np.array(img))


    X_test = np.array(ims_malignant, dtype='uint8')
    X_test = X_test / 255
    y_pred = model.predict(X_test)
    print(y_pred[0])

    if (y_pred[0][0] > y_pred[0][1]):
        print("benign")
        pred = "benign"
    else:
        print("malignant")
        pred = "malignant"
    print("end")

    #for blob in file:
    #    blobs = bucket.blob(blob)
        # Supprimer le fichier il faudra sans doute le faire autre pars car la j'ai pas le perms
    #    blobs.delete()
    return pred

def run(raw_data):
    print('request_json :', raw_data)
    raw_data = json.loads(raw_data)
    if 'instances' not in raw_data:
        return 'Error: Request does not contain instances'
    instances = raw_data['instances']
    if not instances:
        return 'Error: instances is empty'
    instance = instances[0]
    print('instance[0] :', instance)
    if 'image_bytes' not in instance:
        return 'Error: instance does not contain image_bytes'
    image_bytes = instance['image_bytes']['b64']
    print('image_bytes :', image_bytes)
    image_data = base64.b64decode(image_bytes)
    ims_malignant = []
    image_file = image_data
    print('image_file :', image_file)
    img = Image.open(io.BytesIO(image_file))
    img = img.resize((224, 224))
    ims_malignant.append(np.array(img))
    x_test = np.array(ims_malignant, dtype='uint8')
    x_test = x_test / 255
    print('x_test :', x_test)
    y_pred = model.predict(x_test)
    print(y_pred[0])
    #response = predict(image_data)

    if (y_pred[0][0] > y_pred[0][1]):
        print("benign")
        pred = "benign"
    else:
        print("malignant")
        pred = "malignant"
    print("end")
    response = y_pred.tolist()
    return {'predictions': response}

"""

# Chemin du fichier à créer
chemin_fichier = "scorened3.py"

# Écriture du contenu dans le fichier
with open(chemin_fichier, "w") as fichier:
    fichier.write(score_code)

print(f"Le fichier '{chemin_fichier}' a été créé avec succès.")


conda = """
name: model-env
channels:
  - conda-forge
dependencies:
  - python=3.8.17
  - numpy=1.21.2
  - pip=21.2.4
  - setuptools==68.0.0
  - wheel==0.40.0
  - pip:
    - mlflow
    - tensorflow
    - azureml-defaults
    - azure-ai-ml
    - azureml-dataprep[pandas,fuse]
    - azure-storage-blob
    - azureml-inference-server-http
    - inference-schema[numpy-support]==1.5.1
    - xlrd==2.0.1
    - azureml-core
    - azureml-mlflow
    - psutil>=5.8,<5.9
    - tqdm>=4.59,<4.60
    - ipykernel~=6.0
    - matplotlib
    - joblib
    - cloudpickle==2.0.0
"""

# Chemin du fichier à créer
chemin_fichier = "condaenv.yaml"

# Écriture du contenu dans le fichier
with open(chemin_fichier, "w") as fichier:
    fichier.write(conda)

print(f"Le fichier '{chemin_fichier}' a été créé avec succès.")





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

#TODO A VERIFIER pb avec la lib et le nom de l'endpoint
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


def write_endpointid_to_bucket(content):

    # Create a BlobServiceClient object using the connection string
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    # Get a reference to the container
    container_client = blob_service_client.get_container_client(nom_conteneur)

    # Get a reference to the blob (file) in the container
    blob_client = container_client.get_blob_client(file_name)

    # Upload the new content to the blob (overwrite if it already exists)
    blob_client.upload_blob(content, overwrite=True)

    return


def deployendpoint():
    logger.warning("debut thread deploy")

    # Handle to the workspace
    from azure.ai.ml import MLClient
    import os
    # Authentication package
    from azure.identity import DefaultAzureCredential

    credential = DefaultAzureCredential()

    # Get a handle to the workspace
    ml_client = MLClient(
        credential=credential,
        subscription_id="d9414828-c496-44ab-b6f9-e903bce40957",
        resource_group_name="mlops",
        workspace_name="nvml",
    )
    registered_model_name = 'credit_defaults_model'

    dependencies_dir = "./dependencies"
    os.makedirs(dependencies_dir, exist_ok=True)

    custom_env_name = "aml-nico"

    from azureml.core import Workspace
    from azureml.core.model import Model

    # Nom du workspace Azure ML
    workspace_name = "nvml"

    # Nom du modèle enregistré dans Azure ML
    model_name = "credit_defaults_model"

    # Emplacement local où télécharger le modèle
    download_path = "./"

    subscription_id = "d9414828-c496-44ab-b6f9-e903bce40957"
    resource_group = "mlops"

    # Instanciation de l'objet Workspace
    ws = Workspace(subscription_id=subscription_id,
                   resource_group=resource_group,
                   workspace_name=workspace_name)

    # Récupération du modèle enregistré dans Azure ML
    model = Model(ws, name=model_name)
    logger.warning("avant le download")
    ls = afficher_elements_repertoire_courant()


    # Téléchargement du modèle sauvegardé dans l'artifact du modèle enregistré
    #TODO JE TEST DE METTRE LE NOM AU LIEU DE ./
    model.download(registered_model_name, exist_ok=True)
    logger.warning("apres le download")
    ls = afficher_elements_repertoire_courant()
    logger.warning(ls)

    # Affichage du chemin local vers le modèle téléchargé
    print(f"Le modèle a été téléchargé à l'emplacement suivant : {download_path}")


    # à tester !!!

    # TODO WAIT ICI

    import uuid

    # Creating a unique name for the endpoint
    online_endpoint_name = "credit-endpoint-" + str(uuid.uuid4())[:8]



    # create an online endpoint
    endpoint = ManagedOnlineEndpoint(
        name=online_endpoint_name,
        description="this is an online endpoint",
        auth_mode="key",
        tags={
            "training_dataset": "credit_defaults",
            "model_type": "sklearn.GradientBoostingClassifier",
        },
    )

    endpoint = ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    # Wait for the endpoint to be ready
    print(f"Endpoint {endpoint.name} provisioning state: {endpoint.provisioning_state}")

    # TODO virer ca on doit choper l'enpoint qu'on a crée avant
    # online_endpoint_name = 'credit-endpoint-ec7e258b'
    endpoint = ml_client.online_endpoints.get(name=online_endpoint_name)

    print(
        f'Endpoint "{endpoint.name}" with provisioning state "{endpoint.provisioning_state}" is retrieved'
    )

    # Let's pick the latest version of the model
    latest_model_version = max(
        [int(m.version) for m in ml_client.models.list(name=registered_model_name)]
    )

    # Create the thread and pass the argument using the 'args' parameter
    thread = threading.Thread(target=write_endpointid_to_bucket, args=(endpoint.name,))

    # Start the thread
    thread.start()

    # picking the model to deploy. Here we use the latest version of our registered model
    model = ml_client.models.get(name=registered_model_name, version=latest_model_version)

    logger.warning("avant deploy blue")
    env = Environment(
        name=custom_env_name,
        description="Custom environment for Credit Card Defaults pipeline",
        tags={"scikit-learn": "0.24.2"},
        conda_file=os.path.join(dependencies_dir, "conda.yml"),
        image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:latest",
    )
    pipeline_job_env = ml_client.environments.create_or_update(env)

    # create an online deployment.
    blue_deployment = ManagedOnlineDeployment(
        name="blue",
        endpoint_name=online_endpoint_name,
        model=model,
        instance_type="Standard_DS2_v2",
        code_configuration=CodeConfiguration(
            code="./", scoring_script="scorened3.py"
        ),
        instance_count=1,
        environment='aml-nico@latest'
    )

    blue_deployment = ml_client.begin_create_or_update(blue_deployment).result()
    logger.warning("apres le deploy blue")




app = Flask(__name__, template_folder='template')
logger = logging.getLogger(__name__)
#retourne True si le model existe sinon False

#penser à appeler avant de lancer : getendpointormodelfrombucket avec comme parametre modelid/ et modelid.txt
@app.route("/deploy", methods=["POST"])
def deploy():
    logger.warning("avant le thread du deploy")
    print("avant le thread du deploy")
    thread = threading.Thread(target=deployendpoint)
    thread.start()
    res = "l'endpoint a bien ete deploy"
    logger.warning("l'endpoint a bien ete deploy")
    return render_template('deploy.html', result=res)


#penser à appeler avant de lancer : getendpointormodelfrombucket avec comme parametre endpoint/ et endpointid.txt
@app.route("/delete", methods=["POST"])
def delete():
    thread = threading.Thread(target=delete_endpoint_from_azure)
    thread.start()
    res = "suppression en cours"
    return render_template('deploy.html', result=res)
@app.route("/trainfunc", methods=["POST"])
def trainfunc():
    logger.warning("before credential train")
    credential = DefaultAzureCredential()
    # Get a handle to the workspace
    logger.warning("after credential train")
    ml_client = MLClient(
        credential=credential,
        subscription_id="d9414828-c496-44ab-b6f9-e903bce40957",
        resource_group_name="mlops",
        workspace_name="nvml",
    )
    registered_model_name = 'credit_defaults_model'
    logger.warning("after MLCLIENT DEFAULT")
    dependencies_dir = "./dependencies"
    os.makedirs(dependencies_dir, exist_ok=True)

    custom_env_name = "aml-nico"

    # Name assigned to the compute cluster
    cpu_compute_target = "cpu-cluster"

    try:
        # let's see if the compute target already exists
        cpu_cluster = ml_client.compute.get(cpu_compute_target)
        print(
            f"You already have a cluster named {cpu_compute_target}, we'll reuse it as is."
        )

    except Exception:
        print("Creating a new cpu compute target...")

        # Let's create the Azure ML compute object with the intended parameters
        cpu_cluster = AmlCompute(
            name=cpu_compute_target,
            # Azure ML Compute is the on-demand VM service
            type="amlcompute",
            # VM Family
            size="STANDARD_DS2_V2",
            # Minimum running nodes when there is no job running
            min_instances=0,
            # Nodes in cluster
            max_instances=4,
            # How many seconds will the node running after the job termination
            idle_time_before_scale_down=180,
            # Dedicated or LowPriority. The latter is cheaper but there is a chance of job termination
            tier="Dedicated",
        )
        print(
            f"AMLCompute with name {cpu_cluster.name} will be created, with compute size {cpu_cluster.size}"
        )
        # Now, we pass the object to MLClient's create_or_update method
        cpu_cluster = ml_client.compute.begin_create_or_update(cpu_cluster)

    logger.warning("after cluster train")

    # TODO WAIT ICI

    pipeline_job_env = Environment(
        name=custom_env_name,
        description="Custom environment for Credit Card Defaults pipeline",
        tags={"scikit-learn": "0.24.2"},
        conda_file=os.path.join(dependencies_dir, "condaenv.yml"),
        image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:latest",
    )
    pipeline_job_env = ml_client.environments.create_or_update(pipeline_job_env)

    print(
        f"Environment with name {pipeline_job_env.name} is registered to workspace, the environment version is {pipeline_job_env.version}"
    )


    train_src_dir = "./src"
    os.makedirs(train_src_dir, exist_ok=True)

    from azure.ai.ml import command
    from azure.ai.ml import Input

    job = command(
        code="./src/",  # location of source code
        command="python main.py",
        environment="aml-nico@latest",
        compute=cpu_compute_target,
        experiment_name="train_model_credit_default_prediction",
        display_name="credit_default_prediction",
    )

    ml_client.create_or_update(job)
    print("end")

def retrainfunc():

    timestamp = str(int(time.time()))  # Obtenir le timestamp actuel
    nom_nv = "nv" + timestamp
    s3 = boto3.client('s3')
    s3.put_object(Body=nom_nv, Bucket=s3_bucket, Key=file_name)
    print(nom_nv)

    estimator = sagemaker.estimator.Estimator(
        image_uri='778331702232.dkr.ecr.eu-west-3.amazonaws.com/nv-train-sagemaker:latest',
        role='arn:aws:iam::778331702232:role/nvsagemaker',
        instance_count=instance_count,
        instance_type=instance_type,
        output_path='s3://nvabucket/output',
        sagemaker_session=sagemaker_session,
        max_run=24 * 60 * 60
    )

    estimator.fit(job_name=nom_nv)

    #TODO : voir ou mettre car la ça va supprimer aussi ce que je viens de créer mais
    # avant j'ai besoin pour retrain donc peu être dans le code train
    delete_model_file_from_s3(s3_bucket, 'output')

    # supprimer l'ancien endpoint s'il existe
    delete_endpoint_from_sagemaker(endpoint_name)

    # supprimer l'ancien model s'il existe
    delete_sagemaker_model(model_name)
    modeldata = "s3://nvabucket/output/" + nom_nv + "/output/model.tar.gz"
    print("modeldata :", modeldata)
    # Create the model
    model = sagemaker.Model(
        image_uri='778331702232.dkr.ecr.eu-west-3.amazonaws.com/nv-pred-sagemaker:latest',
        model_data=modeldata,
        role='arn:aws:iam::778331702232:role/nvsagemaker',
        sagemaker_session=sagemaker_session,
        name='nvamodel'
    )

    # Deploy the model
    predictor = model.deploy(
        initial_instance_count=instance_count,
        instance_type=instance_type,
        endpoint_name=endpoint_name
    )

    # Get the endpoint
    predictor = sagemaker.predictor.Predictor(
        endpoint_name=endpoint_name,
        sagemaker_session=sagemaker_session
    )
    print("end")
    dest = 'retrain/'

    #supprimer l'ancien endpoint s'il existe
    delete_endpoint_from_sagemaker(endpoint_name)

    #supprimer l'ancien model s'il existe
    delete_sagemaker_model(model_name)

    print("before delete")
    bucket = s3.Bucket(s3_bucket)
    malignant_blobs = bucket.objects.filter(Prefix=dest + 'malignant/')
    for blob in malignant_blobs:
        blob.delete()

    # Supprimer les objets dans le préfixe benign
    benign_blobs = bucket.objects.filter(Prefix=dest + 'benign/')
    for blob in benign_blobs:
        blob.delete()
    print("after delete")

@app.route("/")
def template_test():

    return render_template('index.html', my_string="Wheeeee!", my_list=[0,1,2,3,4,5])



@app.route('/train', methods=['POST'])
def train():
    logger.warning("Log Entry avant le lancement du thread")
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
    s3 = boto3.resource('s3')
    dest = 'retrain/'

    # Ajouter les fichiers du dossier malignant
    for file in malignant:
        # Charger l'image dans S3
        s3.Object(s3_bucket,dest + file.filename).put(Body=file)

    print("malignant load ok")
    # Ajouter les fichiers du dossier benign
    for file in benign:
        s3.Object(s3_bucket,dest + file.filename).put(Body=file)

    thread = threading.Thread(target=retrainfunc)
    thread.start()
    return render_template('retrain.html', result="en cours")


@app.route('/upload', methods=['POST'])
def upload():
    credential = DefaultAzureCredential()

    # Get a handle to the workspace
    ml_client = MLClient(
        credential=credential,
        subscription_id="d9414828-c496-44ab-b6f9-e903bce40957",
        resource_group_name="mlops",
        workspace_name="nvml",
    )

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


    file = request.files['file']
    print(file)
    print("avant")
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
    payload_json = json.dumps(payload)
    print("type :", type(payload_json))

    # Chemin du fichier JSON local
    json_file_path = "./payload.json"

    # Écriture du contenu JSON dans le fichier
    with open(json_file_path, "w") as json_file:
        json_file.write(payload_json)

    print("Le fichier JSON a été créé avec succès : ", json_file_path)

    # Lecture du contenu JSON depuis le fichier
    with open(json_file_path, "r") as json_file:
        json_data = json_file.read()

    # Utilisation des données JSON
    parsed_data = json.loads(json_data)
    print("json_data :", json_data)

    pred = ml_client.online_endpoints.invoke(
        endpoint_name=endpoint_name,
        request_file=json_file_path,
        deployment_name="blue",
    )

    print('prediction :', pred)

    dict_data = json.loads(pred)
    print('type :', type(dict_data))
    print('dict_data["prediction"] :', dict_data["predictions"])
    pred = dict_data["predictions"]
    print('pred[0] :', pred[0])

    if pred[0][0] > pred[0][1]:
        result = "benign à " + str((pred[0][0]) * 100) + "%"
    else:
        result = "malignant à " + str((pred[0][1]) * 100) + "%"

    print("resultat :", result)

    return render_template('result.html', result=result)

if __name__ == "__main__":
    app.run(debug=True,port=5000)
