from flask import Flask, render_template, request,jsonify
from google.cloud import storage
import base64
from google.cloud import aiplatform
import google.cloud.storage
from google.cloud import storage
from google.cloud import aiplatform
from os import path, getenv, makedirs, listdir
import google.auth
import google.auth.transport.requests
import requests

credentials, _ = google.auth.default()
credentials.refresh(google.auth.transport.requests.Request())
identity_token = credentials.id_token

auth_header = {"Authorization": "Bearer " + identity_token}
response = requests.get("${SERVICE_URL}/partners", headers=auth_header)

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


@app.route("/")
def template_test():

    return render_template('index.html', my_string="Wheeeee!", my_list=[0,1,2,3,4,5])

@app.route('/retrain', methods=['POST'])
def retrain():
    print("before send to bucket")
    malignant = request.files.getlist('malignant')
    benign = request.files.getlist('benign')

    # Ajouter les dossiers dans le bucket gcp
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(RETRAIN_BUCKET)
    dest = 'retrain/'

    # Ajouter les fichiers du dossier malignant
    for file in malignant:
        blob = bucket.blob(dest + file.filename)
        blob.upload_from_file(file)

    # Ajouter les fichiers du dossier benign
    for file in benign:
        blob = bucket.blob(dest + file.filename)
        blob.upload_from_file(file)

    print("after send to bucket")

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





    # Create CustomContainerTrainingJob and start run with some service account
    job = aiplatform.CustomContainerTrainingJob(
        display_name=display_name,
        container_uri=container_uri,
        model_serving_container_image_uri=model_serving_container_image_uri,
    )
    model = job.run(model_display_name=display_name, machine_type='n1-standard-16')
    print("apres le job : ", model)
    # Finally deploy the model to endpoint
    endpoint = model.deploy(
        deployed_model_display_name=display_name, sync=True
    )

    # Get the ID of the deployed model
    deployed_model_id = endpoint.resource_name.split("/")[-1]
    print(f"Deployed model ID: {deployed_model_id}")
    endpoint_id = deployed_model_id
    # TODO le sauvegarder le nouveau endpoint id dans un bucket au format .txt en overwrite ou var d'env
    blob = bucket.blob(dest_endpoint + "endpointid.txt")
    blob.upload_from_string(endpoint_id)

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

    return render_template('retrain.html', result=deployed_model_id)
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
        result = "benign à "+ str(prediction[0]) +"%"
    else:
        result = "malignant à "+ str(prediction[1]) +"%"

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
