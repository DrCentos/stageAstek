import argparse
import sagemaker
import boto3

import io
import numpy as np
import csv

# Create a sagemaker session
boto_session = boto3.Session(region_name="eu-west-3")
sagemaker_session = sagemaker.Session(boto_session=boto_session)


def main():
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest="action", help="Action to perform")

    ctSubpar = subparser.add_parser("createTrainingJob", aliases=["ct"], help='Create a training job')
    ctSubpar.add_argument("--name", type=str, required=True)
    ctSubpar.add_argument("--image", type=str, required=True)
    ctSubpar.add_argument("--role", type=str, required=True)
    ctSubpar.add_argument("--instanceType", type=str, required=False, default="ml.m5.large")
    ctSubpar.add_argument("--instanceCount", type=int, required=False, default=1)
    ctSubpar.add_argument("--outputPath", type=str, required=True)
    ctSubpar.add_argument("--inputPath", type=str, required=True)

    cmSubpar = subparser.add_parser("deployModel", aliases=["dm"], help='Create and deploy a model')
    cmSubpar.add_argument("--name", type=str, required=True)
    cmSubpar.add_argument("--endpointName", type=str, required=True)
    cmSubpar.add_argument("--image", type=str, required=True)
    cmSubpar.add_argument("--role", type=str, required=True)
    cmSubpar.add_argument("--modelPath", type=str, required=True)
    cmSubpar.add_argument("--instanceType", type=str, required=False, default="ml.m5.large")
    cmSubpar.add_argument("--instanceCount", type=int, required=False, default=1)

    predSubpar = subparser.add_parser("getPredictions", aliases=["gp"], help='Get predictions from an endpoint')
    predSubpar.add_argument("--endpointName", type=str, required=True)
    predSubpar.add_argument("--data", type=str, required=False)

    args = parser.parse_args()

    if args.action in ["createTrainingJob", "ct"]:
        estimator = createTrainingJob(args.name, args.image, args.role, args.instanceType, args.instanceCount, args.outputPath, args.inputPath)
        print("Model created stored here: ", estimator.model_data)

    elif args.action in ["deployModel", "dm"]:
        model = deployModel(args.name, args.endpointName, args.image, args.role, args.modelPath, args.instanceType, args.instanceCount)
        print("Model deployed to endpoint: ", model.endpoint)

    elif args.action in ["getPredictions", "gp"]:
        predictions = getPredictions(args.endpointName, args.data)
        print("Predictions: ", predictions)

def createTrainingJob(job_name, image, role, instance_type, instance_count, output_path, input_path):
    # Create the estimator
    estimator = sagemaker.estimator.Estimator(
        image_uri=image,
        role=role,
        instance_count=instance_count,
        instance_type=instance_type,
        output_path=output_path,
        sagemaker_session=sagemaker_session,
        max_run=10*60,
    )

    estimator.fit(input_path, job_name=job_name)

    return estimator


def deployModel(model_name, endpoint_name, image, role, model_path, instance_type, instance_count):
    # Create the model
    model = sagemaker.Model(
        image_uri=image,
        model_data=model_path,
        role=role,
        sagemaker_session=sagemaker_session,
        name=model_name
    )

    # Deploy the model
    predictor = model.deploy(
        initial_instance_count=instance_count,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
    )

    return predictor


def getPredictions(endpoint_name, data = "1,2,3"):
    # Get the endpoint
    predictor = sagemaker.predictor.Predictor(
        endpoint_name=endpoint_name,
        sagemaker_session=sagemaker_session,
        content_type="text/csv",
        accept="text/csv",
    )

    # Get predictions
    predictions = predictor.predict(data, initial_args={"ContentType": "text/csv"})

    return predictions


if __name__ == "__main__":
    main()