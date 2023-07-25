import psycopg2
import sys
import boto3
import os

ENDPOINT="nvadatabase.c96r5xa5f7rh.eu-west-3.rds.amazonaws.com"
PORT="3306"
USER="nicolasvallot"
REGION="eu-west-3"
DBNAME="nvadatabase"

#gets the credentials from .aws/credentials
session = boto3.Session(profile_name='RDSCreds')
client = session.client('rds')

token = client.generate_db_auth_token(DBHostname=ENDPOINT, Port=PORT, DBUsername=USER, Region=REGION)

try:
    conn = psycopg2.connect(host=ENDPOINT, port=PORT, database=DBNAME, user=USER, password=token, sslrootcert="SSLCERTIFICATE")
    cur = conn.cursor()
    cur.execute("""SELECT now()""")
    query_results = cur.fetchall()
    print(query_results)
except Exception as e:
    print("Database connection failed due to {}".format(e))