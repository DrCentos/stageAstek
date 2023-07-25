from flask import Flask, render_template, request,jsonify, url_for
from google.cloud import storage
import base64
from google.cloud import aiplatform
import google.cloud.storage
from google.cloud import storage
from google.cloud import aiplatform
from os import path, getenv, makedirs, listdir
import threading
from google.cloud import tasks_v2
from google.protobuf import timestamp_pb2
import time

# Votre code existant ici

# Créer une instance du client Cloud Tasks
client = tasks_v2.CloudTasksClient()
location = "europe-west3"
PROJECT_ID = 'glossy-precinct-371813'
queue_name = "nvqueu"
# Créer une tâche différée
task_url = 'https://nvbuild-utkj5a52gq-ew.a.run.app'  # Remplacez REGION et PROJECT_ID
parent = client.queue_path(PROJECT_ID, location, queue_name)
task = {
    'http_request': {
        'http_method': tasks_v2.HttpMethod.POST,
        'url': task_url,
    },
    'schedule_time': {'seconds': int(time.time()) +5}  # Exécution différée après 5 secondes
}
request2 = tasks_v2.CreateTaskRequest(parent=parent, task=task)

# Envoyer la tâche à Cloud Tasks
response = client.create_task(request2)


