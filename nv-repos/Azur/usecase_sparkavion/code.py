from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml.feature import StringIndexer, VectorAssembler, VectorIndexer, MinMaxScaler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from azure.storage.blob import BlobServiceClient
from azure.storage.blob import BlobClient
import pandas as pd
import io
import os



print("Starting...")

spark = SparkSession.builder.appName("airline-model").getOrCreate()

# Enable Arrow-based columnar data transfers
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

# Define parameters
connectionString = "DefaultEndpointsProtocol=https;AccountName=nvastockage;AccountKey=vCMtCTE4Au52QUOtOd25VJSWeWoPzBQwDgSa2bpV9gbaL+h8k3yyFtCixV3zeUozZAeKr5NmwzmU+AStq6d7Lg==;EndpointSuffix=core.windows.net"

# Connexion au compte de stockage et recuperation du contenu du blob
blob_service_client = BlobServiceClient.from_connection_string(connectionString)
#df = pd.read_csv('https://nvastockage.blob.core.windows.net/nvabucket/flights.csv', delimiter=',')
container_client = blob_service_client.get_container_client("nvabucket")
blob_client = container_client.get_blob_client("flights.csv")
mystr = blob_client.download_blob().readall().decode('utf-8')
csv = spark.read.format('csv').option('header', 'true').load(mystr)
# Read the data from GCS
print("before read")
#csv = spark.read.format('csv').option('header', 'true').load('https://nvastockage.blob.core.windows.net/nvabucket/flights.csv')
#csv = spark_session.createDataFrame(df)
#csv.show(10)
print("Read the data from GCS")

# Select the columns we want to use
# data = csv.select("DayofMonth", "DayOfWeek", "Carrier", "OriginAirportID", "DestAirportID",
# "DepDelay", ((col("ArrDelay") > 15).cast("Int").alias("label")))
data = (csv
        .withColumn("DayofMonth", col("DayofMonth").cast(DoubleType()))
        .withColumn("DayOfWeek", col("DayOfWeek").cast(DoubleType()))
        .withColumn("OriginAirportID", col("OriginAirportID").cast(IntegerType()))
        .withColumn("DestAirportID", col("DestAirportID").cast(IntegerType()))
        .withColumn("DepDelay", col("DepDelay").cast("float"))
        .select("DayofMonth", "DayOfWeek", "Carrier", "OriginAirportID", "DestAirportID", "DepDelay",
                ((col("ArrDelay") > 15).cast("Int").alias("label"))))

print("Select the columns we want to use")

# Split the data into train and test
[train, test] = data.randomSplit([0.7, 0.3])
test = test.withColumnRenamed("label", "trueLabel")

print("Split the data into train and test")

# Create the pipeline
strIdx = StringIndexer(inputCol = "Carrier", outputCol = "CarrierIdx")
catVec = VectorAssembler(inputCols = ["CarrierIdx", "DayofMonth", "DayOfWeek", "OriginAirportID", "DestAirportID"],
                         outputCol="catFeatures")
catIdx = VectorIndexer(inputCol = catVec.getOutputCol(), outputCol = "idxCatFeatures")
numVec = VectorAssembler(inputCols = ["DepDelay"], outputCol="numFeatures")
minMax = MinMaxScaler(inputCol = numVec.getOutputCol(), outputCol="normFeatures")
featVec = VectorAssembler(inputCols=["idxCatFeatures", "normFeatures"], outputCol="features")
lr = LogisticRegression(labelCol="label",featuresCol="features",maxIter=10,regParam=0.3)
#dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")
pipeline = Pipeline(stages=[strIdx, catVec, catIdx, numVec, minMax, featVec, lr])

print("Create the pipeline")

# Train the model
pipelineModel = pipeline.fit(train)

print("Train the model")

# Make predictions
prediction = pipelineModel.transform(test)

print("Make predictions")

# Evaluate the model
evaluator = BinaryClassificationEvaluator(labelCol="trueLabel", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
aur = evaluator.evaluate(prediction)
print ("AUR = ", aur)

# Tune the model
paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.3, 0.1]).addGrid(lr.maxIter, [10, 5]).addGrid(lr.threshold,
                                                                                            [0.4, 0.3]).build()
cv = CrossValidator(estimator=pipeline, evaluator=BinaryClassificationEvaluator(), estimatorParamMaps=paramGrid,
                    numFolds=2)

model = cv.fit(train)

# Make predictions
newPrediction = model.transform(test)

# Evaluate the model
evaluator2 = BinaryClassificationEvaluator(labelCol="trueLabel", rawPredictionCol="prediction", metricName="areaUnderROC")
aur2 = evaluator2.evaluate(newPrediction)
print( "AUR2 = ", aur2)
blob = BlobClient.from_connection_string(conn_str=connectionString, container_name="nvabucket",
                                         blob_name="airplane.txt")
blob.upload_blob(aur2)

# Save the model
best_model = model.bestModel
print("Saved the model")


