#python resize.py --inputb benign --inputm malignant --output nvafter --setup_file ./setup.py --disk_size_gb 40 --experiments=shuffle_mode=service  --num_workers 2



import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from PIL import Image
import io



class ReadImage(beam.DoFn):

    def __init__(self, output_path, input_path):
        self.output_path = output_path
        self.input_path = input_path


    def process(self, element):
        from PIL import Image
        import io
        from google.cloud import storage
        client = storage.Client("glossy-precinct-371813")
        bucket = client.get_bucket(self.input_path)
        blob = bucket.blob(element)

        with io.BytesIO(blob.download_as_bytes()) as f:
            with Image.open(f) as img:
                img = img.resize((224, 224))
                #img.save('gs://{}/{}'.format('nvafter', element))
                img.save(f, format='JPEG')
                contents = f.getvalue()
                #data = img.tobytes()
        #print("img :", img)
        bucket = client.get_bucket(self.output_path)
        blob = bucket.blob(element)
        blob.upload_from_file(io.BytesIO(contents), content_type='image/jpeg')
        #blob.upload_from_filename(img)
        #blob.upload_from_string(base64.b64encode(data).decode('utf-8'))


def run(argv=None):
    import argparse
    from google.cloud import storage
    import apache_beam as beam
    from apache_beam.options.pipeline_options import PipelineOptions, GoogleCloudOptions, StandardOptions
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputb', dest='inputb', required=False, help='Input directory to process.', default='benign')
    parser.add_argument('--inputm', dest='inputm', required=False, help='Input directory to process.', default='malignant')
    parser.add_argument('--output', dest='output', required=False, help='Output directory to write results to.', default='nvabucket')
    parser.add_argument('--input', dest='input', required=False, help='Input directory to process.', default='nvabucketin')
    known_args, pipeline_args = parser.parse_known_args(argv)

    client = storage.Client()

    options = PipelineOptions(pipeline_args)
    google_cloud_options = options.view_as(GoogleCloudOptions)
    google_cloud_options.project = 'glossy-precinct-371813'
    google_cloud_options.job_name = 'nvresize'
    google_cloud_options.staging_location = 'gs://{}/staging'.format(known_args.input)
    google_cloud_options.temp_location = 'gs://{}/temp'.format(known_args.input)
    options.view_as(StandardOptions).runner = 'DataflowRunner'



    # Je récupère tous les fichier jpg qui sont dans le dossier malignant et le dossier benign afin de reduire leur taille
    # en 300x300 dans le cas ou on veut faire en sorte que lorsqu'un utilisateur ses datas il faudrait faire en sorte
    #qu'il ajoute dans une bucket en particulier je fait en sorte d'avoir une cloud fonction qui trigger cette cette bucket
    # la cloud function va alors lancer ce pipeline qui va regarder toutes les images dans cette bucket et faire la réduction...
    # ensuite je continue avec la phase de train...
    # dans le cas d'un simple predict peu être on peu simplement reduire l'image et la normaliser dans le predict j'utilise un StandardScaler dans je suis pas sur que j'ai besoin de le sauvegarder


    client = storage.Client()
    bucket = client.get_bucket(known_args.input)
    blobs = bucket.list_blobs(prefix='{}/'.format(known_args.inputb), delimiter='/')
    filesbenign = []
    for blob in blobs:
        if blob.name.endswith('.jpg'):
            filesbenign.append(blob.name)

    blobs = bucket.list_blobs(prefix='{}/'.format(known_args.inputm), delimiter='/')
    filesmalignant= []
    for blob in blobs:
        if blob.name.endswith('.jpg'):
            filesmalignant.append(blob.name)

    with beam.Pipeline(options=options) as p:
        pimages = (p
                   | 'Read Images benign' >> beam.Create(filesbenign)
                   | 'Process Images benign' >> beam.ParDo(ReadImage(known_args.output, known_args.input))
                   )

        pbenign = (p
                   | 'Read Images malignant' >> beam.Create(filesmalignant)
                   | 'Process Images malignant' >> beam.ParDo(ReadImage(known_args.output, known_args.input))
                   )

if __name__ == '__main__':
    run()