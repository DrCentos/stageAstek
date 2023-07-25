# Import libraries
from google.cloud import aiplatform

# Initialize connection
aiplatform.init(location='europe-west1')

job = aiplatform.CustomContainerTrainingJob(
    display_name='custom_training_sdk',
    container_uri='europe-west1-docker.pkg.dev/glossy-precinct-371813/nvrep/stroke_model@sha256:20c3abb1ad9d986396b058552a455b5c0f45ac839e2cc077b0339b5e2f43f3f7',
    staging_bucket='nvallot_bucket/job_outputs'
)

model = aiplatform.Model.upload(
    project='glossy-precinct-371813',
    display_name='custom_training_nv',
    artifact_uri="gs://nvallot_bucket/mpg",
    serving_container_image_uri="europe-west1-docker.pkg.dev/glossy-precinct-371813/nvrep/mpg:v1",
    localisation='europe-west1'
)

model = aiplatform.Model(model_name='/projects/glossy-precinct-371813/locations/europe-west1/models/bitcoinnv')

'''job.run(
    replica_count=1,
    machine_type='n1-standard-4',
    args=['--data_gcs_path=gs://nvallot_bucket/healthcare-dataset-stroke-data.csv']
)'''

model = aiplatform.Model.upload(
    display_name='custom_training_nv',
    artifact_uri="gs://nvallot_bucket/job_outputs",
    serving_container_image_uri="europe-west1-docker.pkg.dev/glossy-precinct-371813/nvrep/stroke_model@sha256:20c3abb1ad9d986396b058552a455b5c0f45ac839e2cc077b0339b5e2f43f3f7",
)

endpoint = model.deploy(machine_type="n1-standard-4",
                        min_replica_count=1,
                        max_replica_count=1)


test_instance={
    'id': 80422,
    'gender': 'Male',
    'age': 52,
    'hypertension': 0,
    'hear_disease': 1,
    'ever_married': 'Yes',
    'work_type': 'Private',
    'Residence_type': 'Rural',
    'avg_glucose_level': 94.6,
    'bmi': 27.2,
    'smoking_status': 'smokes'
}

response = endpoint.predict([test_instance])

print('API response: ', response)