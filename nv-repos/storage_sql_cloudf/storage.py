from google.cloud import storage

def main(event, context):
    """Triggered by a change to a Cloud Storage bucket.
    Args:
         event (dict): Event payload.
         context (google.cloud.functions.Context): Metadata for the event.
    """
    file = event
    content = storage.Client().bucket(file['bucket']).blob(file['name']).download_as_string().decode("utf-8")
    print(f"Processing file: {file['name']}.")
    print(f"Processing content: {content}.")
