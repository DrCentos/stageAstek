from azure.batch import BatchServiceClient
from azure.batch import batch_auth as batchauth
from azure.batch import models as batchmodels
from azure.batch.models import ImageReference, VirtualMachineConfiguration, ContainerConfiguration, PoolAddParameter

# Remplacez les valeurs suivantes par les vôtres
BATCH_ACCOUNT_NAME = 'batchmlop'
BATCH_ACCOUNT_KEY = 'QjKQ3Co+RiYj3/gw7dlCwryGlDqEXIvLj79Y0Ee+Tn/nT8B0KpvIxcKhk13RkojILr0/4ywdfCkf+ABaQ/XicA=='
POOL_ID = 'nvpool'
JOB_ID = 'nvjob'
TASK_ID = 'nvtask'
TASK_COMMAND_LINE = '/usr/bin/python3 /pipeline.py'
DOCKER_IMAGE_NAME = 'nvregistre.azurecr.io/usecase1:latest'
REGISTRY_SERVER = 'nvregistre.azurecr.io'
REGISTRY_USERNAME = 'nvregistre'
REGISTRY_PASSWORD = 'yg5PqEJOOWixzRMKZIrjVSY63E6Hz9IhJMx74IeLcj+ACRBzNMJE'
BATCH_ACCOUNT_URL = 'https://batchmlop.francecentral.batch.azure.com'

# Créer un client Batch
credentials = batchauth.SharedKeyCredentials(BATCH_ACCOUNT_NAME, BATCH_ACCOUNT_KEY)
batch_client = BatchServiceClient(credentials, batch_url = BATCH_ACCOUNT_URL)

# Créer une référence d'image pour l'image Docker stockée sur Azure Container Registry
image_ref_to_use = ImageReference(
    publisher='microsoft-azure-batch',
    offer='ubuntu-server-container',
    sku='20-04-lts',
    version='latest'
)

# Spécifier la configuration du conteneur
container_conf = ContainerConfiguration(
    container_image_names=[DOCKER_IMAGE_NAME],
    container_registries=[batchmodels.ContainerRegistry(
        registry_server=REGISTRY_SERVER,
        user_name=REGISTRY_USERNAME,
        password=REGISTRY_PASSWORD)])

# Créer un nouveau pool avec la configuration de l'image et du conteneur
new_pool = PoolAddParameter(
    id=POOL_ID,
    virtual_machine_configuration=VirtualMachineConfiguration(
        image_reference=image_ref_to_use,
        container_configuration=container_conf,
        node_agent_sku_id='batch.node.ubuntu 20.04'),
    vm_size='STANDARD_A2_V2',
    target_dedicated_nodes=1)

batch_client.pool.add(new_pool)

# Créer un nouveau job et l'ajouter au pool
job = batchmodels.JobAddParameter(
    id=JOB_ID,
    pool_info=batchmodels.PoolInformation(pool_id=POOL_ID))
batch_client.job.add(job)

# Créer une nouvelle tâche et l'ajouter au job
task = batchmodels.TaskAddParameter(
    id=TASK_ID,
    command_line=TASK_COMMAND_LINE,
    container_settings=batchmodels.TaskContainerSettings(
        image_name=DOCKER_IMAGE_NAME))
batch_client.task.add(job_id=JOB_ID, task=task)