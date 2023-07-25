import os
import sys
from os import path

current_folder = path.dirname(path.abspath(__file__))
import subprocess
subprocess.check_call([sys.executable, "-m", "pip","install","--upgrade","tensorflow","mlflow","protobuf==3.20.0" , "-t", current_folder])


# Authentication package
from azure.ai.ml.entities import AmlCompute
from azure.ai.ml.entities import Environment
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import logging


logger = logging.getLogger(__name__)
def trainfunc():
    logger.warning("before credential train")
    credential = DefaultAzureCredential()
    # Get a handle to the workspace
    logger.warning("after credential train")
    ml_client = MLClient(
        credential=credential,
        subscription_id="d9414828-c496-44ab-b6f9-e903bce40957",
        resource_group_name="mlops",
        workspace_name="nvml",
    )
    registered_model_name = 'credit_defaults_model'
    logger.warning("after MLCLIENT DEFAULT")
    dependencies_dir = "./dependencies"
    os.makedirs(dependencies_dir, exist_ok=True)

    custom_env_name = "aml-nico"

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
            size="Standard_F4s_v2",
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

    logger.warning("after cluster train")

    # TODO WAIT ICI

    pipeline_job_env = Environment(
        name=custom_env_name,
        description="Custom environment for Credit Card Defaults pipeline",
        tags={"scikit-learn": "0.24.2"},
        conda_file=os.path.join(dependencies_dir, "conda.yml"),
        image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:latest",
    )
    pipeline_job_env = ml_client.environments.create_or_update(pipeline_job_env)

    print(
        f"Environment with name {pipeline_job_env.name} is registered to workspace, the environment version is {pipeline_job_env.version}"
    )


    train_src_dir = "./src"
    os.makedirs(train_src_dir, exist_ok=True)

    from azure.ai.ml import command
    from azure.ai.ml import Input

    job = command(
        code="./src/",  # location of source code
        command="python main.py",
        environment="aml-nico@latest",
        compute=cpu_compute_target,
        experiment_name="train_model_credit_default_prediction",
        display_name="credit_default_prediction",
    )

    ml_client.create_or_update(job)
    print("end")


trainfunc()