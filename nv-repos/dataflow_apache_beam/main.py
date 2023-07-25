#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""A word-counting workflow."""
import csv
from dataclasses import asdict

# pytype: skip-file

# beam-playground:
#   name: WordCount
#   description: An example that counts words in Shakespeare's works.
#   multifile: false
#   pipeline_options: --output output.txt
#   context_line: 44
#   categories:
#     - Combiners
#     - Options
#     - Quickstart
#   complexity: MEDIUM
#   tags:
#     - options
#     - count
#     - combine
#     - strings

import data_c

import argparse
import logging
import re
import json

import apache_beam as beam
import pandas as pd
from apache_beam.io import ReadFromText
from apache_beam.io import WriteToText
from google.cloud import storage
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.coders.coders import Coder
from beam_mysql.connector.io import WriteToMySQL

from apache_beam.io.gcp.internal.clients import bigquery

from dataclasses import dataclass

table_schema = 'operateur:STRING, annee:INTEGER, filiere:STRING, code_categorie_consommation:STRING, libelle_categorie_consommation:STRING, code_grand_secteur:STRING, libelle_grand_secteur:STRING, code_naf:INTEGER, libelle_secteur_naf2: STRING, conso:FLOAT, pdl:INTEGER, indqual:FLOAT, nombre_mailles_secretisees:INTEGER, code_region:INTEGER, libelle_region:STRING, naf_code:STRING, naf_INTEGER_vf:STRING, naf_INTEGER_v2_65c:STRING, naf_INTEGER_v2_40c: STRING'

@dataclass
class Conso:
    operateur: str
    annee: int
    filiere: str
    code_categorie_consommation: str
    libelle_categorie_consommation: str
    code_grand_secteur: str
    libelle_grand_secteur: str
    code_naf: int
    libelle_secteur_naf2: str
    conso: float
    pdl: int
    indqual: float
    nombre_mailles_secretisees: int
    code_region: int
    libelle_region: str




class CsvToJsonConvertion(beam.DoFn):

    def __init__(self, inputFilePath, outputFilePath):
        self.inputFilePath = inputFilePath
        self.outputFilePath = outputFilePath

    def start_bundle(self):
        from google.cloud import storage
        self.client = storage.Client()

    def process(self, something):
        import csv
        import pandas as pd
        import json
        print('DANS LE PROCESS CSV TO JSON')
        df = pd.read_csv(self.inputFilePath, delimiter=';')
        df = df.fillna(0)
        df.to_json(self.outputFilePath, orient='index')
        data = df.to_dict(orient='index')
        #output = []
        for index, row in df.iterrows():
            #print(data.get(index))
            yield (data.get(index))
            #output.append(data.get(index))
        #return output

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
    return asdict(ConsoAndNaf(**conso,**output))


def run(argv=None, save_main_session=True):
    """Main entry point; defines and runs the wordcount pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        dest='input',
        help='Input file to process.')
    parser.add_argument(
        '--output',
        dest='output',
        required=True,
        help='Output file to write results to.')
    parser.add_argument(
        '--naf',
        dest='naf',
        help='naf.')
    parser.add_argument("--host", dest="host", default="0.0.0.0")
    parser.add_argument("--port", dest="port", default=3306)
    parser.add_argument("--database", dest="database", default="test_db")
    parser.add_argument("--table", dest="table", default="tests")
    parser.add_argument("--user", dest="user", default="root")
    parser.add_argument("--password", dest="password", default="root")
    parser.add_argument("--batch_size", dest="batch_size", default=0)
    parser.add_argument("--tablebq", dest="tablebq", default="tests")
    known_args, pipeline_args = parser.parse_known_args(argv)

    # We use the save_main_session option because one or more DoFn's in this
    # workflow rely on global context (e.g., a module imported at module level).
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = save_main_session

    print('known_args.input : ',type(known_args.input))
    # The pipeline will be run on exiting the with block.

    write_to_mysql = WriteToMySQL(
        host=known_args.host,
        database=known_args.database,
        table=known_args.table,
        user=known_args.user,
        password=known_args.password,
        port=known_args.port,
        batch_size=int(known_args.batch_size),
    )

    schema =  {'fields': [
        {'name': 'operateur', 'type': 'STRING', 'mode': 'NULLABLE'},
        {'name': 'annee', 'type': 'STRING', 'mode': 'NULLABLE'},
        {'name': 'filiere', 'type': 'STRING', 'mode': 'NULLABLE'},
        {'name': 'code_categorie_consommation', 'type': 'STRING', 'mode': 'NULLABLE'},
        {'name': 'libelle_categorie_consommation', 'type': 'STRING', 'mode': 'NULLABLE'},
        {'name': 'code_grand_secteur', 'type': 'STRING', 'mode': 'NULLABLE'},
        {'name': 'libelle_grand_secteur', 'type': 'STRING', 'mode': 'NULLABLE'},
        {'name': 'code_naf', 'type': 'STRING', 'mode': 'NULLABLE'},
        {'name': 'libelle_secteur_naf2', 'type': 'STRING', 'mode': 'NULLABLE'},
        {'name': 'conso', 'type': 'STRING', 'mode': 'NULLABLE'},
        {'name': 'pdl', 'type': 'STRING', 'mode': 'NULLABLE'},
        {'name': 'indqual', 'type': 'STRING', 'mode': 'NULLABLE'},
        {'name': 'nombre_mailles_secretisees', 'type': 'STRING', 'mode': 'NULLABLE'},
        {'name': 'code_region', 'type': 'STRING', 'mode': 'NULLABLE'},
        {'name': 'libelle_region', 'type': 'STRING', 'mode': 'NULLABLE'},
        {'name': 'naf_code', 'type': 'STRING', 'mode': 'NULLABLE'},
        {'name': 'naf_int_vf', 'type': 'STRING', 'mode': 'NULLABLE'},
        {'name': 'naf_int_v2_65c', 'type': 'STRING', 'mode': 'NULLABLE'},
        {'name': 'naf_int_v2_40c', 'type': 'STRING', 'mode': 'NULLABLE'}]}

    with beam.Pipeline(options=pipeline_options) as pipeline:

        pconso = (
            pipeline
            | 'Start conso' >> beam.Create([None])
            | 'write bucket' >> beam.ParDo(CsvToJsonConvertion(known_args.input, known_args.output))
            | 'write data base' >> write_to_mysql
        )

        pnaf = (
            pipeline
            | 'Start Naf' >> beam.io.ReadFromText(known_args.naf)
            | "Read Naf" >> beam.Map(read_naf)
            | "get key naf" >> beam.Map(lambda x : (x["code"], x))
        )
        naf_dict = beam.pvalue.AsDict(pnaf)

        pjoin = (
            pconso
            | 'Jointure' >> beam.Map(jointure_naf_conso,naf_dict)
            | 'Write BigQuerry' >> beam.io.WriteToBigQuery(
                method=beam.io.WriteToBigQuery.Method.STREAMING_INSERTS,
                table=known_args.tablebq,
                schema=schema,
                insert_retry_strategy=beam.io.gcp.bigquery_tools.RetryStrategy.RETRY_NEVER
                )
            )



    #print('pconso : ',pconso)
    #print('pconso : ',pconso)
    print("APREPRPERPEPR")

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()