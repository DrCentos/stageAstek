from google.cloud import aiplatform

my_dataset = aiplatform.TabularDataset.create(location='europe-west1',
    display_name="my-dataset", gcs_source=['gs://nvallot_bucket/data.csv'])
def create_custom_job_sample(
    project: str,
    display_name: str,
    container_image_uri: str,
    location: str = "europe-west1",
    api_endpoint: str = "europe-west1-aiplatform.googleapis.com",
):
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.JobServiceClient(client_options=client_options)
    custom_job = {
        "display_name": display_name,
        "job_spec": {
            "worker_pool_specs": [
                {
                    "machine_spec": {
                        "machine_type": "n1-standard-4"
                    },
                    "replica_count": 1,
                    "container_spec": {
                        "image_uri": container_image_uri,
                        "command": [],
                        "args": [],
                    },
                }
            ]
        },
    }
    parent = f"projects/{project}/locations/{location}"
    response = client.create_custom_job(parent=parent, custom_job=custom_job)
    print("response:", response)

    model = custom_job.run(my_dataset,
                    replica_count=1,
                    machine_type="n1-standard-4")

if __name__ == "__main__":
    create_custom_job_sample('glossy-precinct-371813','nvprojet','europe-west1-docker.pkg.dev/glossy-precinct-371813/nvrep/image:latest')