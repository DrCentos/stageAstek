gcloud run deploy nvbuild  --source . --platform managed  --region europe-west1   --no-allow-unauthenticated   --project=glossy-precinct-371813  --service-account=nvallot-service@glossy-precinct-371813.iam.gserviceaccount.com



# Crée cluster
gcloud dataproc clusters create nv-cluster --region europe-west3 --zone europe-west3-a --master-machine-type n1-standard-2 --master-boot-disk-size 50 --num-workers 2 --worker-machine-type n1-standard-2 --worker-boot-disk-size 50 --image-version 2.0-debian10 --project glossy-precinct-371813

# Publier le job pyspark dans bucket
gsutil cp ./job.py  gs://nv-conso-csv/Dataproc/job.py

# Lancer le job
gcloud dataproc jobs submit pyspark --cluster=nv-cluster --region=europe-west3 gs://nv-conso-csv/Dataproc/job.py

# Crée un workflow template
gcloud dataproc workflow-templates create nv-template --region europe-west3

# Set managed cluster
gcloud dataproc workflow-templates set-managed-cluster nv-template --region europe-west3 --cluster-name nv-cluster

# Create the job
gcloud dataproc workflow-templates add-job pyspark gs://nv-conso-csv/Dataproc/job.py --step-id nv-step --workflow-template nv-template --region europe-west3

# Launch job
gcloud dataproc workflow-templates instantiate nv-template --region europe-west3



gcloud run deploy nvbuild  --source . --platform managed  --region europe-west1   --no-allow-unauthenticated
--project=glossy-precinct-371813  --service-account=nvallot-service@glossy-precinct-371813.iam.gserviceaccount.com
