gcloud run deploy nvbuild  --source . --platform managed  --region europe-west1   --no-allow-unauthenticated   --project=glossy-precinct-371813  --service-account=nvallot-service@glossy-precinct-371813.iam.gserviceaccount.com


aws cloudformation deploy --template-file mytemplate.yml --stack-name nvstack --parameter-overrides database=nicov --capabilities CAPABILITY_NAMED_IAM