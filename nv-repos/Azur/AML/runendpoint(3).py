# Handle to the workspace
from azure.ai.ml import MLClient
import os
# Authentication package
from azure.identity import DefaultAzureCredential


credential = DefaultAzureCredential()

# Get a handle to the workspace
ml_client = MLClient(
    credential=credential,
    subscription_id="d9414828-c496-44ab-b6f9-e903bce40957",
    resource_group_name="mlops",
    workspace_name="nvml2",
)
registered_model_name ='credit_defaults_model'

dependencies_dir = "./dependencies"
os.makedirs(dependencies_dir, exist_ok=True)

custom_env_name = "aml-scikit-learn"

from azureml.core import Workspace
from azureml.core.model import Model

# Nom du workspace Azure ML
workspace_name = "nvml2"

# Nom du modèle enregistré dans Azure ML
model_name = "credit_defaults_model"

# Emplacement local où télécharger le modèle
download_path = "./"

subscription_id = "d9414828-c496-44ab-b6f9-e903bce40957"
resource_group = "mlops"

# Instanciation de l'objet Workspace
ws = Workspace(subscription_id=subscription_id,
               resource_group=resource_group,
               workspace_name=workspace_name)

# Récupération du modèle enregistré dans Azure ML
model = Model(ws, name=model_name)

# Téléchargement du modèle sauvegardé dans l'artifact du modèle enregistré
model.download(download_path,exist_ok=True)

# Affichage du chemin local vers le modèle téléchargé
print(f"Le modèle a été téléchargé à l'emplacement suivant : {download_path}")



# à tester !!!

#TODO WAIT ICI

import uuid

# Creating a unique name for the endpoint
online_endpoint_name = "credit-endpoint-" + str(uuid.uuid4())[:8]

from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration,
)

# create an online endpoint
endpoint = ManagedOnlineEndpoint(
    name=online_endpoint_name,
    description="this is an online endpoint",
    auth_mode="key",
    tags={
        "training_dataset": "credit_defaults",
        "model_type": "sklearn.GradientBoostingClassifier",
    },
)

endpoint = ml_client.online_endpoints.begin_create_or_update(endpoint).result()
# Wait for the endpoint to be ready
endpoint.wait_for_deployment(show_output=True)
print(f"Endpoint {endpoint.name} provisioning state: {endpoint.provisioning_state}")

#TODO virer ca on doit choper l'enpoint qu'on a crée avant
#online_endpoint_name = 'credit-endpoint-ec7e258b'
endpoint = ml_client.online_endpoints.get(name=online_endpoint_name)

print(
    f'Endpoint "{endpoint.name}" with provisioning state "{endpoint.provisioning_state}" is retrieved'
)

# Let's pick the latest version of the model
latest_model_version = max(
    [int(m.version) for m in ml_client.models.list(name=registered_model_name)]
)


# picking the model to deploy. Here we use the latest version of our registered model
model = ml_client.models.get(name=registered_model_name, version=latest_model_version)



from azure.ai.ml.entities import Environment

env = Environment(
    name=custom_env_name,
    description="Custom environment for Credit Card Defaults pipeline",
    tags={"scikit-learn": "0.24.2"},
    conda_file=os.path.join(dependencies_dir, "conda.yml"),
    image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:latest",
)
pipeline_job_env = ml_client.environments.create_or_update(env)

# create an online deployment.
blue_deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=online_endpoint_name,
    model=model,
    instance_type="Standard_DS2_v2",
    code_configuration=CodeConfiguration(
        code="./", scoring_script="scorened.py"
    ),
    instance_count=1,
    environment='aml-scikit-learn@latest'
)

blue_deployment = ml_client.begin_create_or_update(blue_deployment).result()

