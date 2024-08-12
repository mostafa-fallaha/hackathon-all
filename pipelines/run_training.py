import os
import subprocess

def run_training():
    print("Current Working Directory:", os.getcwd())

    print("Pulling the latest data...")
    subprocess.run(["dvc", "pull", "--force"], check=True)

    print("Running the training script...")
    subprocess.run(["python", "ml/train.py"], check=True)

    print("Deploying the trained models...")
    subprocess.run(["python", "pipelines/deploy.py"], check=True)

if __name__ == "__main__":
    run_training()
