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

from azure.ai.ml.entities import AmlCompute

# Name assigned to the compute cluster
cpu_compute_target = "cpu-cluster"

try:
    # let's see if the compute target already exists
    cpu_cluster = ml_client.compute.get(cpu_compute_target)
    print(
        f"You already have a cluster named {cpu_compute_target}, we'll reuse it as is."
    )

except Exception:
    print("Creating a new cpu compute target...")

    # Let's create the Azure ML compute object with the intended parameters
    cpu_cluster = AmlCompute(
        name=cpu_compute_target,
        # Azure ML Compute is the on-demand VM service
        type="amlcompute",
        # VM Family
        size="STANDARD_DS3_V2",
        # Minimum running nodes when there is no job running
        min_instances=0,
        # Nodes in cluster
        max_instances=4,
        # How many seconds will the node running after the job termination
        idle_time_before_scale_down=180,
        # Dedicated or LowPriority. The latter is cheaper but there is a chance of job termination
        tier="Dedicated",
    )
    print(
        f"AMLCompute with name {cpu_cluster.name} will be created, with compute size {cpu_cluster.size}"
    )
    # Now, we pass the object to MLClient's create_or_update method
    cpu_cluster = ml_client.compute.begin_create_or_update(cpu_cluster)



import os



from azure.ai.ml.entities import Environment



pipeline_job_env = Environment(
    name=custom_env_name,
    description="Custom environment for Credit Card Defaults pipeline",
    tags={"scikit-learn": "0.24.2"},
    conda_file=os.path.join(dependencies_dir, "conda.yml"),
    image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:latest",
)
pipeline_job_env = ml_client.environments.create_or_update(pipeline_job_env)

#TODO WAIT ICI

#TODO A VERIFIER RECUPERER LE MODEL EN LOCAL dans credit_defaults_model
# à tester il faut peu êter le mettre dans une fonction d'un autre .py l'import et faire un appel à celui ci!!!! 

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
model.download(download_path)

# Affichage du chemin local vers le modèle téléchargé
print(f"Le modèle a été téléchargé à l'emplacement suivant : {download_path}")

#il faut remettre le scaler ici

# à tester !!!

print(
    f"Environment with name {pipeline_job_env.name} is registered to workspace, the environment version is {pipeline_job_env.version}"
)

import os

train_src_dir = "./src"
os.makedirs(train_src_dir, exist_ok=True)


from azure.ai.ml import command
from azure.ai.ml import Input




job = command(
    code="./src/",  # location of source code
    command="python main.py",
    environment="aml-scikit-learn@latest",
    compute=cpu_compute_target,
    experiment_name="train_model_credit_default_prediction",
    display_name="credit_default_prediction",
)

ml_client.create_or_update(job)

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



