import base64

def main(event, context):

    pubsub_message = base64.b64decode(event['data']).decode('utf-8')

    data_json = json.loads(pubsub_message)
    print("json",data_json)

    print(pubsub_message)
