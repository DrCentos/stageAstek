from google.cloud import aiplatform

REGION = 'us-central1'
PROJECT_ID = 'your-project-name'
bucket = 'gs://your-bucket-name/model' # Should be same as AIP_STORAGE_URI specified in docker file
container_uri='us-central1-docker.pkg.dev/your-project-name/testrepo/lightgbm_model'
model_serving_container_image_uri='us-central1-docker.pkg.dev/your-project-name/testrepo/lightgbm_serve'
display_name='Custom Job from Code'

aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=bucket)

# Create CustomContainerTrainingJob and start run with some service account
job = aiplatform.CustomContainerTrainingJob(
display_name=display_name,
container_uri=container_uri,
model_serving_container_image_uri=model_serving_container_image_uri,
)

model = job.run( model_display_name=display_name, service_account="")

# Finally deploy the model to endpoint
endpoint = model.deploy(
deployed_model_display_name=display_name, sync=True
)