# test the blue deployment with some sample data
# Handle to the workspace
from azure.ai.ml import MLClient
import os
# Authentication package
from azure.identity import DefaultAzureCredential
import json
import pandas as pd

credential = DefaultAzureCredential()

# Get a handle to the workspace
ml_client = MLClient(
    credential=credential,
    subscription_id="d9414828-c496-44ab-b6f9-e903bce40957",
    resource_group_name="mlops",
    workspace_name="nvml2",
)

online_endpoint_name = 'credit-endpoint-f1857ae3'


pred = ml_client.online_endpoints.invoke(
    endpoint_name=online_endpoint_name,
    request_file="sample-request.json", #créer un fichier avec juste 1,2,3,5
    deployment_name="blue",
)

print('pred : ', pred)
'''
ml_client.online_deployments.get_logs(
    name="blue", endpoint_name=online_endpoint_name, lines=50
)'''