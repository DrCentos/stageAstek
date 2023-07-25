from azure.batch import BatchServiceClient
from azure.batch.models import ContainerConfiguration, PoolAddParameter, VirtualMachineConfiguration, ImageReference, ContainerRegistry, TaskAddParameter, TaskContainerSettings
from azure.batch import batch_auth as batchauth
from azure.batch.models import JobAddParameter, PoolInformation

# Remplacez les valeurs suivantes par les vôtres
BATCH_ACCOUNT_NAME = 'batchmlop'
BATCH_ACCOUNT_KEY = 'QjKQ3Co+RiYj3/gw7dlCwryGlDqEXIvLj79Y0Ee+Tn/nT8B0KpvIxcKhk13RkojILr0/4ywdfCkf+ABaQ/XicA=='
POOL_ID = 'nvpool'
JOB_ID = 'nvjob'
TASK_ID = 'nvtask'
TASK_COMMAND_LINE = ''#'az acr login --name nvregistre'
DOCKER_IMAGE_NAME = 'nvregistre.azurecr.io/usecase1:latest'
REGISTRY_SERVER = 'nvregistre.azurecr.io'
REGISTRY_USERNAME = 'nvregistre'
REGISTRY_PASSWORD = 'nDolzRDnvIMq2l26qx8Bo4g0o2+3fXZURb1nHw1rDP+ACRAsulmF'
BATCH_ACCOUNT_URL = 'https://batchmlop.francecentral.batch.azure.com'

credentials = batchauth.SharedKeyCredentials(BATCH_ACCOUNT_NAME, BATCH_ACCOUNT_KEY)

batch_client = BatchServiceClient(credentials, batch_url = BATCH_ACCOUNT_URL)

container_registry = ContainerRegistry(
    registry_server=REGISTRY_SERVER,
    user_name=REGISTRY_USERNAME,
    password=REGISTRY_PASSWORD
)

# Create container configuration, prefetching Docker images from the container registry
container_conf = ContainerConfiguration(
        container_image_names = ['nvregistre.azurecr.io/usecase1:latest'],
        container_registries = [container_registry])

pool = PoolAddParameter(
    id=POOL_ID,
    vm_size='STANDARD_A2_V2',
    target_dedicated_nodes=1,
    virtual_machine_configuration=VirtualMachineConfiguration(
        image_reference=ImageReference(
            publisher='microsoft-azure-batch',
            offer='ubuntu-server-container',
            sku='20-04-lts',
            version='latest'
        ),
        container_configuration=container_conf,
        node_agent_sku_id='batch.node.ubuntu 20.04'
    )
)

batch_client.pool.add(pool)





job = JobAddParameter(
    id=JOB_ID,
    pool_info=PoolInformation(pool_id=POOL_ID)
)

batch_client.job.add(job)

'''
# Create a ContainerRegistryClient that will authenticate through Active Directory
from azure.containerregistry import ContainerRegistryClient

endpoint = "https://nvregistre.azurecr.io"
audience = "/subscriptions/d9414828-c496-44ab-b6f9-e903bce40957/resourceGroups/mlops/providers/Microsoft.ContainerRegistry/registries/nvregistre"
client = ContainerRegistryClient(endpoint, credentials, audience=audience)
'''

'''
import docker
client = docker.from_env()
registry = 'nvregistre.azurecr.io'
username = 'nvregistre'
client.login(username=username, registry=registry)'''



task_container_settings = TaskContainerSettings(
    image_name=DOCKER_IMAGE_NAME,
    container_run_options='--rm --workdir /app',
    registry=container_registry
)


task = TaskAddParameter(
    id=TASK_ID,
    command_line=TASK_COMMAND_LINE,
    container_settings=task_container_settings
)

batch_client.task.add(job_id=JOB_ID, task=task)


'''
task_id = 'sampletask'
task_container_settings = TaskContainerSettings(
    image_name='nvregistre.azurecr.io/usecase1:latest',
    container_run_options='--rm --workdir .')
task = TaskAddParameter(
    id=task_id,
    command_line='/bin/sh -c \"echo \'hello world\' > $AZ_BATCH_TASK_WORKING_DIR/output.txt\"',
    container_settings=task_container_settings
)'''

'''
# Create task to run the Docker image
task = TaskAddParameter(
    id=TASK_ID,
    command_line=TASK_COMMAND_LINE,
    container_settings=TaskContainerSettings(
        image_name=DOCKER_IMAGE_NAME,
        container_run_options='--rm',
        registry=container_registry
    )
)
'''
#batch_client.task.add(JOB_ID, task)

