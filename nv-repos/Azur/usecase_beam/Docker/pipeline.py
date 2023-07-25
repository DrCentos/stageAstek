#import sys
#from os import path

#current_folder = path.dirname(path.abspath(__file__))
#import subprocess
#subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "pip","-t", current_folder])
#subprocess.check_call([sys.executable, "-m", "pip3", "install", "apache_beam", "boto3","pandas","pymysql","s3fs","fsspec", "-t", current_folder])

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import pandas as pd
import io
import json
import logging
import pymysql
from azure.storage.blob import BlobServiceClient

my_host = "nvadatabase.mysql.database.azure.com"
user_name = "nicolasvallot"
password = "nvallot1!"
db_name = "nvadatabase"
port=3306
ssl_ca="DigiCertGlobalRootG2.crt.pem"
table = 'usecase'
tableall = 'jointure'


pipeline_options = PipelineOptions(
    runner="DirectRunner"
)

class CsvToJsonConvertion(beam.DoFn):

    def __init__(self, inputFilePath, outputFilePath):
        self.inputFilePath = inputFilePath
        self.outputFilePath = outputFilePath


    def process(self, something):
        import csv
        import pandas as pd
        import json
        from azure.storage.blob import BlobClient
        #print('DANS LE PROCESS CSV TO JSON')

        # Define parameters
        connectionString = "DefaultEndpointsProtocol=https;AccountName=nvastockage;AccountKey=vCMtCTE4Au52QUOtOd25VJSWeWoPzBQwDgSa2bpV9gbaL+h8k3yyFtCixV3zeUozZAeKr5NmwzmU+AStq6d7Lg==;EndpointSuffix=core.windows.net"

        # Connexion au compte de stockage et recuperation du contenu du blob
        blob_service_client = BlobServiceClient.from_connection_string(connectionString)
        '''blob_client = blob_service_client.get_blob_client(container="input", blob=self.inputFilePath)
        with open('conso.csv', "wb") as my_blob:
            download_stream = blob_client.download_blob()
            my_blob.write(download_stream.readall())'''
        #df = pd.read_csv('https://nvastockage.blob.core.windows.net/input/conso-elec-gaz-annuelle-par-naf-agregee-region.csv', delimiter=';')
        df = pd.read_csv(self.inputFilePath, delimiter=';')
        df = df.fillna(0)
        myjson = df.to_json(orient='index')
        #upload du fichier json dans le container output blob j.json
        blob_client = blob_service_client.get_blob_client(self.outputFilePath, 'j.json')
        #print(f'Uploading file myjson to container ouput...')
        blob_client.upload_blob(myjson, overwrite=True)
        data = df.to_dict(orient='index')
        print("The length of the dictionary is {}".format(len(data)))
        # on veut que les 100 premier car sinon trop long mais normalement on prend tout
        for i in range (0,100):
            yield (data.get(i))





def read_naf(input):

    import csv
    from dataclasses import asdict
    from dataclasses import dataclass

    @dataclass
    class Naf:
        line: str
        code: str
        int_vf: str
        int_v2_65c: str
        int_v2_40c: str

    my_csv = csv.reader([input], delimiter =',')

    for col in my_csv:
        #print('type col :',type(col))
        #print (asdict(data_c.Naf(*col))
        #print("nafnafnaf :", asdict(Naf(*col)))
        return asdict(Naf(*col))






def jointure_naf_conso(conso,naf):
    from dataclasses import dataclass
    from dataclasses import asdict
    @dataclass
    class ConsoAndNaf:
        operateur: str
        annee: int
        filiere: str
        code_categorie_consommation: str
        libelle_categorie_consommation: str
        code_grand_secteur: str
        libelle_grand_secteur: str
        code_naf: str
        libelle_secteur_naf2: str
        conso: float
        pdl: int
        indqual: float
        nombre_mailles_secretisees: int
        code_region: int
        libelle_region: str
        naf_code: str
        naf_int_vf: str
        naf_int_v2_65c: str
        naf_int_v2_40c: str
    #print('conso[code_naf]', conso['code_naf'])
    str_codenaf_conso = repr(conso['code_naf'])
    print(type(str_codenaf_conso), str_codenaf_conso)

    if str_codenaf_conso in naf.keys():
        print('erroorororro :', naf[str_codenaf_conso])
        output = naf[str_codenaf_conso]
        output = {
            "naf_code": output["code"],
            "naf_int_vf": output["int_vf"],
            "naf_int_v2_65c": output["int_v2_65c"],
            "naf_int_v2_40c": output["int_v2_40c"]
        }
    else:
        output = {
            "naf_code": "",
            "naf_int_vf": "",
            "naf_int_v2_65c": "",
            "naf_int_v2_40c": ""
        }
    #print("cococo:", asdict(ConsoAndNaf(**conso,**output)))
    return asdict(ConsoAndNaf(**conso,**output))

class WriteToMYSQLBIS(beam.DoFn):
    def __init__(self, host, user, password, database, table):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.table = table
        self.connection = None
        self.count = 0

    def start_bundle(self):
        self.connection = pymysql.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor,
            port=3306,
            ssl_ca="DigiCertGlobalRootG2.crt.pem"
        )

    def process(self, element):
        with self.connection.cursor() as cursor:
            keys = list(element.keys())
            values = list(element.values())
            # Loop through all values and truncate if they are longer than 100 characters
            for i in range(len(values)):
                if isinstance(values[i], str) and len(values[i]) > 100:
                    values[i] = values[i][:100]
            columns = ', '.join(keys)
            placeholders = ', '.join(['%s'] * len(keys))
            sql = f"INSERT INTO {self.table} ({columns}) VALUES ({placeholders})"
            cursor.execute(sql, values)
            self.connection.commit()
            self.count = self.count + 1
            print('self.count : ', self.count)

    def finish_bundle(self):
        self.connection.close()

def getnaf():
    from os import path
    import os

    current_folder = path.dirname(path.abspath(__file__))
    # Connexion au compte de stockage et recuperation du contenu du blob
    blob_service_client = BlobServiceClient.from_connection_string(
        "DefaultEndpointsProtocol=https;AccountName=nvastockage;AccountKey=vCMtCTE4Au52QUOtOd25VJSWeWoPzBQwDgSa2bpV9gbaL+h8k3yyFtCixV3zeUozZAeKr5NmwzmU+AStq6d7Lg==;EndpointSuffix=core.windows.net")

    '''blob_client = blob_service_client.get_blob_client(container="nvabucket", blob="nafdata.csv")
    with open(file='/filename.csv', mode="wb") as sample_blob:
        download_stream = blob_client.download_blob()
        file = sample_blob.write(download_stream.readall())'''
    container_client = blob_service_client.get_container_client("nvabucket")
    blob_client = container_client.get_blob_client("nafdata.csv")
    mystr = blob_client.download_blob().readall().decode('utf-8')# read blob content as string
    return mystr


def run():
    from apache_beam.io.fileio import MatchFiles, ReadMatches
    write_to_sql = WriteToMYSQLBIS(host=my_host, user=user_name, password=password, database=db_name, table=table)
    write_to_sql_all = WriteToMYSQLBIS(host=my_host, user=user_name, password=password, database=db_name, table=tableall)
    file_pattern = getnaf()
    print('file_pattern : ',type(file_pattern))
    with beam.Pipeline(options=pipeline_options) as pipeline:
        pconso = (
                pipeline
                | 'Start conso' >> beam.Create([None])
                | 'write bucket' >> beam.ParDo(CsvToJsonConvertion('https://nvastockage.blob.core.windows.net/nvabucket/conso.csv','output'))
        )
        pconso | "Write to MYSQL" >> beam.ParDo(write_to_sql)

        pnaf = (
                pipeline
                #| 'Start naf' >> beam.Create([None])
                #| "Get csv" >> beam.Map(getnaf)
                #| 'Start Naf' >> beam.io.ReadFromText(file_pattern=local_file_path)
                | beam.io.fileio.MatchFiles(file_pattern=file_pattern)
                | beam.io.fileio.ReadMatches()
                | beam.Map(lambda file: file.read_utf8())
                | "Read Naf" >> beam.Map(read_naf)
                | "get key naf" >> beam.Map(lambda x: (x["code"], x))
        )
        naf_dict = beam.pvalue.AsDict(pnaf)

        pjoin = (
                pconso
                | 'Jointure' >> beam.Map(jointure_naf_conso, naf_dict)
                | "Write to SQL ALL" >> beam.ParDo(write_to_sql_all)
        )


if __name__ == '__main__':
    #getnaf()
    logging.getLogger().setLevel(logging.INFO)
    run()
    print("END")