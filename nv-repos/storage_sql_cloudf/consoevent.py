from google.cloud import storage
import json
from google.cloud import bigquery


def get_storage(file):
    content = storage.Client().get_bucket(file['bucket']).get_blob(file['name']).download_as_string().decode("utf-8")
    # print(f"Processing file: {file['name']}.")
    # print(f"Processing content: {content}.")
    return content


def get_big_querry(file):
    tab_conso = bigquery.Client().dataset('glossy-precinct-371813.nv_consommation').table(
        'glossy-precinct-371813.nv_consommation.tab_conso')
    return tab_conso


def create_json(data):
    rows_to_insert = []
    for i in range(len(data)):
        _json = data[i]
        for key in _json:
            if key == "fields":
                _jsonEnd = _json[key]
        # TODO : rajouter les 3 clés
        rows_to_insert.append(_jsonEnd)
    print(f'rows_to_insert : {rows_to_insert}')

    return rows_to_insert


def main(event, context):
    file = event
    content = get_storage(file)
    content = json.loads(content)
    rows_to_insert = create_json(content)
    print("type : ", type(rows_to_insert))

    table_id = 'glossy-precinct-371813.nv_consommation.tab_conso'
    sql_request = bigquery.Client().insert_rows_json(table_id, rows_to_insert)
    print(f'errors = {sql_request}')




