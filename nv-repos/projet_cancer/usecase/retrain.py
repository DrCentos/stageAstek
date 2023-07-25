import google.cloud.storage
from google.cloud import storage
from google.cloud import aiplatform

RETRAIN_BUCKET = "nvabucket"
REGION = 'europe-west1'
PROJECT_ID = 'glossy-precinct-371813'
bucket_staging = 'gs://nvabucket/model'  # Should be same as AIP_STORAGE_URI specified in docker file
container_uri = 'europe-west1-docker.pkg.dev/glossy-precinct-371813/nvrep/nvtrain:latest'
model_serving_container_image_uri = 'europe-west1-docker.pkg.dev/glossy-precinct-371813/nvrep/nvpred:latest'
display_name = 'Custom Job from Code'
aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=bucket_staging)
dest_endpoint = "endpoint/"

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

# supprimer l'ancien endpoint s'il existe
old_endpoint_id = getendpointbucket()
check_and_delete_endpoint(old_endpoint_id)

# sauvegarder le nouveau endpoint id dans un bucket au format .txt en overwrite ou var d'env
blob = bucket.blob(dest_endpoint + "endpointid.txt")
blob.upload_from_string(endpoint_id)

print("before delete")
# on vide les répertoires retrain/malignant et retrain/benign du bucket RETRAIN_BUCKET pour ne pas retrain avec la prochaine fois
malignant_blobs = bucket.list_blobs(prefix=dest)
for blob in malignant_blobs:
    blob.delete()

# Supprimer les objets dans le répertoire benign
benign_blobs = bucket.list_blobs(prefix=dest)
for blob in benign_blobs:
    blob.delete()
print("after delete")
