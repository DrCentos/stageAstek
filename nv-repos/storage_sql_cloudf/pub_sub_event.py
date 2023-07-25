import base64
import json
from google.cloud import bigquery


def main(event, context):
    pubsub_message = base64.b64decode(event['data']).decode('utf-8')
    data = json.loads(pubsub_message)


    bq_client = bigquery.Client()
    dataset_id = 'glossy-precinct-371813.nv_consommation'
    table_id = 'glossy-precinct-371813.nv_consommation.tab_conso'
    dataset_ref = bq_client.dataset(dataset_id)

    errors = bq_client.insert_rows_json(table_id, data)
    print(f'errors = {errors}')




