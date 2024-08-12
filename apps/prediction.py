import mlflow.pyfunc
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

# Load environment variables
s3_installs_model_path = os.getenv("S3_INSTALLS_MODEL_PATH")
s3_rating_model_path = os.getenv("S3_RATING_MODEL_PATH")
aws_region = os.getenv("AWS_REGION")
s3_bucket = os.getenv("S3_BUCKET")

# Print environment variables to debug
print(f"S3_INSTALLS_MODEL_PATH: {s3_installs_model_path}")
print(f"S3_RATING_MODEL_PATH: {s3_rating_model_path}")
print(f"AWS_REGION: {aws_region}")

# Ensure the S3 endpoint is set to the correct region
mlflow.set_tracking_uri(f"https://{aws_region}.s3.amazonaws.com")
mlflow.pyfunc.get_model_dependencies(model_uri=s3_installs_model_path)

# Load models from S3 using the correct region and path
model_installs = mlflow.pyfunc.load_model(s3_installs_model_path)
model_rating = mlflow.pyfunc.load_model(s3_rating_model_path)

def predict_app_metrics(features):
    features_df = pd.DataFrame([features.dict()])
    installs_prediction = model_installs.predict(features_df)
    rating_prediction = model_rating.predict(features_df)
    return {
        "predicted_installs": installs_prediction[0],
        "predicted_rating": rating_prediction[0]
    }
