import sys
from os import path

current_folder = path.dirname(path.abspath(__file__))

import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "pyspark", "boto3", "-t", current_folder])

from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col


# naf_bucket = "s3a://bom-bucket1/Pipeline/input/naf.csv"
# conso_bucket = "s3a://bom-bucket1/Pipeline/input/conso.csv"
# output_bucket = "s3a://bom-bucket1/Pipeline/output/out.csv"

naf_bucket = path.join(current_folder, "naf.csv")
conso_bucket = path.join(current_folder, "conso.csv")
output_bucket = path.join(current_folder, "out.csv")

# naf_bucket = "C:\\Users\\mboudot\\Projects\\bom-aws-repo\\KDA\\naf.csv"
# conso_bucket = "C:\\Users\\mboudot\\Projects\\bom-aws-repo\\KDA\\conso.csv"
# output_bucket = "C:\\Users\\mboudot\\Projects\\bom-aws-repo\\KDA\\out"

import boto3
s3 = boto3.resource('s3')
bucket = s3.Bucket('nvabucket')

naf_object = bucket.Object('nafdata.csv')
naf_object.download_file(naf_bucket)
conso_object = bucket.Object('nico3.csv')
conso_object.download_file(conso_bucket)


print("Starting...")

# Create a Spark context
sc = SparkContext()
print("Spark context created")

# Initialize a Spark session
spark = (
    SparkSession
    .builder
    .appName("MyPipeline")
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.10.2,org.apache.hadoop:hadoop-client:2.10.2")
    .config("spark.jars.excludes", "com.google.guava:guava")
    .getOrCreate()
)
print("Spark session created")

# Step 1: Read the naf.csv file from S3
naf = spark.read.option("header", True).csv(naf_bucket)
print("Read the naf.csv file from ", naf_bucket)

# Step 2: Select and rename the necessary columns
naf_parsed = naf.select(col("Code").alias("code_naf"), col("ligne"))
print("Select and rename the necessary columns")

# Step 3: Cache the naf_parsed DataFrame for performance
naf_parsed.cache()
print("Cache the naf_parsed DataFrame for performance")

# Step 4: Read the conso.csv file from S3
conso = spark.read.option("header", True).option("delimiter", ";").csv(conso_bucket)
print("Read the conso.csv file from ", conso_bucket)

# Step 5: Select and rename the necessary columns
conso_parsed = conso.select(
    col("annee"),
    col("filiere"),
    col("code_categorie_consommation").alias("code_cat"),
    col("libelle_categorie_consommation").alias("libelle_cat"),
    col("code_grand_secteur").alias("code_secteur"),
    col("libelle_grand_secteur").alias("libelle_secteur"),
    col("conso"),
    col("pdl"),
    col("indqual"),
    col("nombre_mailles_secretisees").alias("nb_mailles"),
    col("code_region").alias("code_reg"),
    col("libelle_region").alias("libelle_reg"),
    col("code_naf").cast("integer").alias("code_naf_int") # Cast the column to integer for the join
)
print("Select and rename the necessary columns")

# Step 6: Do a left join between conso_parsed and naf_parsed
joined = conso_parsed.join(naf_parsed, conso_parsed.code_naf_int == naf_parsed.code_naf, "left")
print("Do a left join between conso_parsed and naf_parsed")

# Step 7: Write the output to s3://bom-bucket1/Pipeline/output/out/csv
joined.write.option("header", True).csv(output_bucket)
print("Write the output to ", output_bucket)

out_object = bucket.Object('out.csv')

import os
for file in os.listdir(output_bucket):
    if file.endswith(".csv"):
        out_object.upload_file(path.join(output_bucket, file))
        print("Uploaded ", file)

# Stop the Spark context
sc.stop()
