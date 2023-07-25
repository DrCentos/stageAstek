import sys
from os import path

current_folder = path.dirname(path.abspath(__file__))
import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "apache_beam", "boto3","pandas","pymysql","s3fs","fsspec", "-t", current_folder])

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import pandas as pd
import io
import json
import base64
import logging
import boto3
import pymysql


rds_host = "nvadatabase.c96r5xa5f7rh.eu-west-3.rds.amazonaws.com"
user_name = "nicolasvallot"
password = "nicolasvallot"
db_name = "nvadatabase"
table = 'usecase'
tableall = 'jointure'

pipeline_options = PipelineOptions(
    runner="DirectRunner"
)



client = boto3.client('kinesis',region_name="eu-west-3")

s3 = boto3.resource('s3')
s3_object = s3.Object('nvabucket', 'nico3.csv')
csv_data = s3_object.get()['Body'].read().decode('utf-8')
df = pd.read_csv(io.StringIO(csv_data), error_bad_lines=False)



class CsvToJsonConvertion(beam.DoFn):

    def __init__(self, inputFilePath, outputFilePath):
        self.inputFilePath = inputFilePath
        self.outputFilePath = outputFilePath


    def process(self, something):
        import csv
        import pandas as pd
        import json
        #print('DANS LE PROCESS CSV TO JSON')
        df = pd.read_csv(self.inputFilePath, delimiter=';')
        df = df.fillna(0)
        df.to_json(self.outputFilePath, orient='index')
        data = df.to_dict(orient='index')
        #output = []
        for index, row in df.iterrows():
            #print(data.get(index))
            #print('datatatata :' ,data.get(index))
            yield (data.get(index))
            #output.append(data.get(index))
        #return output

class WriteToRDS(beam.DoFn):
    def __init__(self, host, user, password, database, table):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.table = table

    def start_bundle(self):
        import pymysql
        self.connection = pymysql.connect(host=self.host, user=self.user, password=self.password, db=self.database)
        self.cursor = self.connection.cursor()

    def process(self, element):
        placeholders = ', '.join(['%s'] * len(element))
        columns = ', '.join(element.keys())
        sql = "INSERT INTO %s (`%s`) VALUES (%s)" % (self.table, columns, placeholders)
        self.cursor.execute(sql, list(element.values()))
        self.connection.commit()

    def finish_bundle(self):
        self.cursor.close()
        self.connection.close()


def read_naf(input : str):

    import csv
    from dataclasses import asdict
    from dataclasses import dataclass

    @dataclass
    class Naf:
        ligne: str
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






class WriteToAWSRDSALL(beam.DoFn):
    import pymysql
    def __init__(self, host, user, password, database, table):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.table = table
        self.connection = None

    def start_bundle(self):
        self.connection = pymysql.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )

    def process(self, element):
        with self.connection.cursor() as cursor:
            sql = f"INSERT INTO {self.table} (operateur, annee, filiere, code_categorie_consommation, libelle_categorie_consommation, code_grand_secteur, libelle_grand_secteur, code_naf, libelle_secteur_naf2, conso, pdl, indqual, nombre_mailles_secretisees, code_region, libelle_region, naf_code, naf_int_vf, naf_int_v2_65c, naf_int_v2_40c) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            values = (element['operateur'], element['annee'], element['filiere'], element['code_categorie_consommation'], element['libelle_categorie_consommation'], element['code_grand_secteur'], element['libelle_grand_secteur'], element['code_naf'], element['libelle_secteur_naf2'], element['conso'], element['pdl'], element['indqual'], element['nombre_mailles_secretisees'], element['code_region'], element['libelle_region'],element['naf_code'],element['naf_int_vf'],element['naf_int_v2_65c'],element['naf_int_v2_40c'])
            cursor.execute(sql, values)
            self.connection.commit()

    def finish_bundle(self):
        self.connection.close()


class WriteToAWSRDS(beam.DoFn):
    def __init__(self, host, user, password, database, table):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.table = table
        self.connection = None

    def start_bundle(self):
        self.connection = pymysql.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )

    def process(self, element):
        with self.connection.cursor() as cursor:
            sql = f"INSERT INTO {self.table} (operateur, annee, filiere, code_categorie_consommation, libelle_categorie_consommation, code_grand_secteur, libelle_grand_secteur, code_naf, libelle_secteur_naf2, conso, pdl, indqual, nombre_mailles_secretisees, code_region, libelle_region) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            values = (element['operateur'], element['annee'], element['filiere'], element['code_categorie_consommation'], element['libelle_categorie_consommation'], element['code_grand_secteur'], element['libelle_grand_secteur'], element['code_naf'], element['libelle_secteur_naf2'], element['conso'], element['pdl'], element['indqual'], element['nombre_mailles_secretisees'], element['code_region'], element['libelle_region'])
            cursor.execute(sql, values)
            self.connection.commit()

    def finish_bundle(self):
        self.connection.close()

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
    if conso['code_naf'] in naf.keys():
        output = naf[conso['code_naf']]
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

class WriteToAWSRDSBIS(beam.DoFn):
    def __init__(self, host, user, password, database, table):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.table = table
        self.connection = None

    def start_bundle(self):
        self.connection = pymysql.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )

    def process(self, element):
        with self.connection.cursor() as cursor:
            keys = list(element.keys())
            #print("keykey :",keys)
            values = tuple(element.values())
            #print("valuesvalues :", values)
            columns = ', '.join(keys)
            placeholders = ', '.join(['%s'] * len(keys))
            sql = f"INSERT INTO {self.table} ({columns}) VALUES ({placeholders})"
            cursor.execute(sql, values)
            self.connection.commit()

    def finish_bundle(self):
        self.connection.close()

def run():

    write_to_rds = WriteToAWSRDSBIS(host=rds_host, user=user_name, password=password, database=db_name, table=table)
    write_to_rds_all = WriteToAWSRDSBIS(host=rds_host, user=user_name, password=password, database=db_name, table=tableall)

    with beam.Pipeline(options=pipeline_options) as pipeline:
        pconso = (
                pipeline
                | 'Start conso' >> beam.Create([None])
                | 'write bucket' >> beam.ParDo(CsvToJsonConvertion('s3://nvabucket/conso-elec-gaz-annuelle-par-naf-agregee-region.csv','s3://ave-bucket-1/n.json'))
        )
        pconso | "Write to RDS" >> beam.ParDo(write_to_rds)
        pnaf = (
                pipeline
                | 'Start Naf' >> beam.io.ReadFromText('s3://nvabucket/nafdata.csv')
                | "Read Naf" >> beam.Map(read_naf)
                | "get key naf" >> beam.Map(lambda x: (x["code"], x))
        )
        naf_dict = beam.pvalue.AsDict(pnaf)
        pjoin = (
                pconso
                | 'Jointure' >> beam.Map(jointure_naf_conso, naf_dict)
                | "Write to RDS ALL" >> beam.ParDo(write_to_rds_all)
        )


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()