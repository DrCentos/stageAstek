from azure.batch import BatchServiceClient
from azure.batch.models import ElevationLevel, AutoUserScope, AutoUserSpecification, UserIdentity, StartTask, PoolAddParameter, VirtualMachineConfiguration, ImageReference, ContainerRegistry, TaskAddParameter
from azure.batch import batch_auth as batchauth
from azure.batch.models import JobAddParameter, PoolInformation
import helpers

# Remplacez les valeurs suivantes par les vôtres
BATCH_ACCOUNT_NAME = 'batchmlop'
BATCH_ACCOUNT_KEY = 'QjKQ3Co+RiYj3/gw7dlCwryGlDqEXIvLj79Y0Ee+Tn/nT8B0KpvIxcKhk13RkojILr0/4ywdfCkf+ABaQ/XicA=='
POOL_ID = 'anvpoole'
JOB_ID = 'anvjobee'
TASK_ID = 'anvtaskee'
DOCKER_IMAGE_NAME = 'nvregistre.azurecr.io/usecase1:latest'
TASK_COMMAND = ''
TASK_COMMAND_LINE = f"/bin/bash -c 'docker run --rm {DOCKER_IMAGE_NAME} {TASK_COMMAND}'"
REGISTRY_SERVER = 'nvregistre.azurecr.io'
REGISTRY_USERNAME = 'nvregistre'
REGISTRY_PASSWORD = 'yg5PqEJOOWixzRMKZIrjVSY63E6Hz9IhJMx74IeLcj+ACRBzNMJE'
BATCH_ACCOUNT_URL = 'https://batchmlop.francecentral.batch.azure.com'

credentials = batchauth.SharedKeyCredentials(BATCH_ACCOUNT_NAME, BATCH_ACCOUNT_KEY)

batch_client = BatchServiceClient(credentials, batch_url=BATCH_ACCOUNT_URL)

task_commands = [
    'sudo groupadd docker',

    # Install pip
    'sudo usermod -aG docker $USER',
    # Install the azure-storage module so that the task script can access
    # Azure Blob storage, pre-cryptography version
    'newgrp docker']

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
        node_agent_sku_id='batch.node.ubuntu 20.04'
    ),
    start_task = StartTask(
        #command_line=helpers.wrap_commands_in_shell('linux', task_commands),
        command_line="/bin/bash -c \"cat /etc/group\"",
        wait_for_success = True,
        user_identity = UserIdentity(
            auto_user=AutoUserSpecification(
                scope=AutoUserScope.pool,
                elevation_level=ElevationLevel.admin)),
)
)

batch_client.pool.add(pool)

container_registry = ContainerRegistry(
    registry_server=REGISTRY_SERVER,
    user_name=REGISTRY_USERNAME,
    password=REGISTRY_PASSWORD
)

job = JobAddParameter(
    id=JOB_ID,
    pool_info=PoolInformation(pool_id=POOL_ID)
)

batch_client.job.add(job)

task = TaskAddParameter(
    id=TASK_ID,
    command_line=TASK_COMMAND_LINE,
)

batch_client.task.add(job_id=JOB_ID, task=task)
