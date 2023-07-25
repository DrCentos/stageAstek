from google.cloud import aiplatform

REGION = 'europe-west1'
PROJECT_ID = 'glossy-precinct-371813'
bucket = 'gs://nvallot_bucket/job_outputs/model' # Should be same as AIP_STORAGE_URI specified in docker file
container_uri='europe-west1-docker.pkg.dev/glossy-precinct-371813/nvrep/b_model:latest'
model_serving_container_image_uri='europe-west1-docker.pkg.dev/glossy-precinct-371813/nvrep/b_serve:latest'
display_name='Custom Job from Code'

aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=bucket)

# Create CustomContainerTrainingJob and start run with some service account
job = aiplatform.CustomContainerTrainingJob(
display_name=display_name,
container_uri=container_uri,
model_serving_container_image_uri=model_serving_container_image_uri,
)

'''
from datetime import datetime

#gs://nvallot_bucket/job_outputs/model/aiplatform-custom-training-2023-03-02-10:48:15.251

timestamp = datetime.now()
time = timestamp.isoformat()
text = time[:10] + '-' + time[11:]
delLastStr = text[:-3]
var3 = "aiplatform-custom-training-" + delLastStr
print(var3)'''

model = job.run( model_display_name=display_name)


# Finally deploy the model to endpoint
endpoint = model.deploy(
deployed_model_display_name=display_name, sync=True
)