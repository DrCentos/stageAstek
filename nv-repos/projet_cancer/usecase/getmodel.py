from google.cloud import aiplatform

project_id = "glossy-precinct-371813"
location = "europe-west1"  # La région où le modèle est déployé
model_display_name  = "Custom Job from Code"

# Créer le client Vertex AI
client_options = {"api_endpoint": f"{location}-aiplatform.googleapis.com"}
client = aiplatform.gapic.ModelServiceClient(client_options=client_options)

# Récupérer le modèle
response = client.list_models(parent=f"projects/{project_id}/locations/{location}")

for model in response:
    if model.display_name == model_display_name:
        model_name = model.name
        retrieved_model = client.get_model(name=model_name)
        print(retrieved_model)
        break