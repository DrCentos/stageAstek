
"""Example Airflow DAG that kicks off a Cloud Dataproc Template that runs a
Spark Pi Job.

This DAG relies on an Airflow variable
https://airflow.apache.org/docs/apache-airflow/stable/concepts/variables.html
* project_id - Google Cloud Project ID to use for the Cloud Dataproc Template.
"""

import datetime
import logging
from airflow import models
from airflow.contrib.operators import dataproc_operator
from airflow.utils.dates import days_ago
from airflow.models import Variable


project_id = Variable.get("project_id")  #attention ici le code n'était pas bon voir doc airflow
print('lalalalallaa ;',project_id)
logging.info('lalalalala')
logging.info(project_id)
default_args = {
    # Tell airflow to start one day ago, so that it runs as soon as you upload it
    "start_date": days_ago(1),
    "project_id": project_id,
}

# Define a DAG (directed acyclic graph) of tasks.
# Any task you create within the context manager is automatically added to the
# DAG object.
with models.DAG(
    # The id you will see in the DAG airflow page
    "dataproc_workflow_dag",
    default_args=default_args,
    # The interval with which to schedule the DAG
    schedule_interval=datetime.timedelta(days=1),  # Override to match your needs
) as dag:

    start_template_job = dataproc_operator.DataprocWorkflowTemplateInstantiateOperator(
        # The task id of your job
        task_id="dataproc_workflow_dag",
        # The template id of your workflow
        template_id="nvspa",
        project_id=project_id,
        # The region for the template
        # For more info on regions where Dataflow is available see:
        # https://cloud.google.com/dataflow/docs/resources/locations
        region="europe-west4",
    )
    print('FINITOTOTOT ')
    logging.debug('FINITOTOTOTO')