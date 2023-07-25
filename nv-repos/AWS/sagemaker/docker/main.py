import sagemaker
import boto3

boto_session = boto3.Session(region_name="eu-west-3")
sagemaker_session = sagemaker.Session(boto_session=boto_session)

instance_count = 1
instance_type = 'ml.m5.large'

endpoint_name = 'nva-endpoint'

data = "1,2,3"

#ce code marche pas ici mais sur le notebook aws oui
'''
estimator = sagemaker.estimator.Estimator(
    image_uri='778331702232.dkr.ecr.eu-west-3.amazonaws.com/nv-train-sagemaker',
    role='arn:aws:iam::778331702232:role/nvsagemaker',
    instance_count=instance_count,
    instance_type=instance_type,
    output_path='s3://nvabucket/output',
    sagemaker_session=sagemaker_session,
    max_run=24 * 60 * 60
)

estimator.fit('s3://nvabucket/input/coinbaseUSD.csv', job_name='nva-job')
'''


# Create the model
model = sagemaker.Model(
    image_uri='778331702232.dkr.ecr.eu-west-3.amazonaws.com/nv-pred-sagemaker',
    model_data='s3://nvabucket/output/nv-job/output/model.tar.gz',
    role='arn:aws:iam::778331702232:role/nvsagemaker',
    sagemaker_session=sagemaker_session,
    name='nv-model'
)

# Deploy the model
predictor = model.deploy(
    initial_instance_count=instance_count,
    instance_type=instance_type,
    endpoint_name=endpoint_name
)


# Get the endpoint
predictor = sagemaker.predictor.Predictor(
    endpoint_name=endpoint_name,
    sagemaker_session=sagemaker_session,
    content_type="text/csv",
    accept="text/csv"
)

# Get predictions
predictions = predictor.predict(data, initial_args={"ContentType": "text/csv"})

#we can also print it in s3 bucket
print(predictions)


