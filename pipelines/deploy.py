import os
import boto3
from botocore.exceptions import NoCredentialsError
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# AWS configuration
S3_BUCKET = os.getenv("S3_BUCKET")
MODEL_DIR = "mlruns/"
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "me-central-1")

def upload_to_s3(local_path, bucket, s3_path):
    try:
        # Explicitly set the endpoint URL
        s3 = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=AWS_REGION,
            endpoint_url=f"https://s3.{AWS_REGION}.amazonaws.com"  # Explicitly setting the S3 endpoint
        )

        # Upload file to S3
        s3.upload_file(local_path, bucket, s3_path)
        print(f"Successfully uploaded {local_path} to {s3_path}")

    except FileNotFoundError:
        print(f"The file {local_path} was not found")
    except NoCredentialsError:
        print("Credentials not available")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def deploy_models():
    for root, dirs, files in os.walk(MODEL_DIR):
        for file in files:
            if file.endswith(".pkl") or file.endswith(".txt"):
                local_file_path = os.path.join(root, file)

                # Construct the S3 path, ensuring the 'mlruns/' prefix is maintained
                s3_file_path = os.path.join(MODEL_DIR, os.path.relpath(local_file_path, MODEL_DIR))

                # Replace backslashes with forward slashes for S3 compatibility
                s3_file_path = s3_file_path.replace("\\", "/")

                upload_to_s3(local_file_path, S3_BUCKET, s3_file_path)

if __name__ == "__main__":
    print("Starting model deployment to S3...")
    deploy_models()
    print("Deployment completed.")
