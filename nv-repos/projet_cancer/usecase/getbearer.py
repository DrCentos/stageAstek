from google.oauth2 import service_account
from google.cloud import aiplatform

# Define the project ID and endpoint ID
project_id = 'glossy-precinct-371813'
endpoint_id = "your-endpoint-id"
region = "europe-west1"

# Define the path to your service account key file
key_path = "/path/to/your/service/account/key.json"

# Create a credentials object using your service account key file
credentials = service_account.Credentials.from_service_account_file(key_path)

# Create a client object using your credentials
client_options = {"api_endpoint": f"{region}-aiplatform.googleapis.com"}
client = aiplatform.gapic.EndpointServiceClient(credentials=credentials, client_options=client_options)

# Get the endpoint object
endpoint = client.get_endpoint(name=f"projects/{project_id}/locations/{region}/endpoints/{endpoint_id}")

# Extract the bearer token from the endpoint object
bearer_token = endpoint.access_token.token

# Print the bearer token
print(bearer_token)
