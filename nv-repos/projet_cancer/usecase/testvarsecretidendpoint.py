from google.cloud import secretmanager
from google.oauth2.service_account import Credentials
"""
# Spécifiez le chemin vers le fichier JSON contenant les informations d'identification de la service account
key_path = 'service_account_key.json'

# Chargez les informations d'identification de la service account à partir du fichier JSON
credentials = Credentials.from_service_account_file(key_path)

# Créez une instance du client Secret Manager
client = secretmanager.SecretManagerServiceClient(credentials=credentials)

# Remplacez {PROJECT_ID} par l'ID de votre projet GCP
project_id = "glossy-precinct-371813"

# Spécifiez l'ID de la variable d'environnement à récupérer
secret_id = "nvasecret"
version_id = "1"
name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"

# Récupérez la valeur de la variable d'environnement
response = client.access_secret_version(request={"name": name})
value = response.payload.data.decode("UTF-8")

# Utilisez la valeur de la variable d'environnement
print(f"La valeur de ma variable d'environnement est : {value}")
"""


# Replacez "PROJECT_ID" par le nom de votre projet GCP
project_id = "glossy-precinct-371813"
from google.cloud import secretmanager_v1beta1 as secretmanager

# Créer un client Secret Manager
client = secretmanager.SecretManagerServiceClient()

# Créer un secret
parent = f"projects/{project_id}"
secret_id = 'my-secret'
secret = {
    'replication': {
        'automatic': {},
    },
}
response = client.create_secret(parent=parent, secret_id=secret_id, secret=secret)

# Récupérer le contenu du secret
secret_name = f"{parent}/secrets/{secret_id}/versions/latest"
response = client.access_secret_version(name=secret_name)
content = response.payload.data.decode('UTF-8')
print(content)
