import mlflow.pyfunc
import pandas as pd
from dotenv import load_dotenv
import os
import boto3
from mlflow.tracking import MlflowClient

from pipelines.deploy import S3_BUCKET

load_dotenv()

# Load environment variables
s3_installs_model_path = os.getenv("S3_INSTALLS_MODEL_PATH")
s3_rating_model_path = os.getenv("S3_RATING_MODEL_PATH")
aws_region = os.getenv("AWS_REGION")

# Print environment variables to debug
print(f"S3_INSTALLS_MODEL_PATH: {s3_installs_model_path}")
print(f"S3_RATING_MODEL_PATH: {s3_rating_model_path}")
print(f"AWS_REGION: {aws_region}")

# Set the S3 endpoint URL explicitly
# Ensure the S3 endpoint is set to the correct region
mlflow.set_tracking_uri(f"https://s3.{aws_region}.amazonaws.com")

# Load models from S3 using the correct region
model_installs = mlflow.pyfunc.load_model(f"s3://{S3_BUCKET}/{s3_installs_model_path}")
model_rating = mlflow.pyfunc.load_model(f"s3://{S3_BUCKET}/{s3_rating_model_path}")


def predict_app_metrics(features):
    features_df = pd.DataFrame([features.dict()])
    installs_prediction = model_installs.predict(features_df)
    rating_prediction = model_rating.predict(features_df)
    return {
        "predicted_installs": installs_prediction[0],
        "predicted_rating": rating_prediction[0]
    }
