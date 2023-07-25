from google.cloud import storage

def check_and_create_bucket(bucket_name):
    # Instantier le client GCS
    storage_client = storage.Client()

    # Vérifier si le bucket existe
    bucket = storage_client.bucket(bucket_name)
    if bucket.exists():
        print(f"Le bucket {bucket_name} existe déjà.")
    else:
        # Créer le bucket
        bucket = storage_client.create_bucket(bucket_name)
        print(f"Le bucket {bucket_name} a été créé avec succès.")

def main():
    # Nom du bucket à vérifier et créer
    bucket_name = 'nvafter'  # Remplacez par le nom du bucket que vous souhaitez vérifier/créer

    # Vérifier et créer le bucket si nécessaire
    check_and_create_bucket(bucket_name)

if __name__ == '__main__':
    main()
