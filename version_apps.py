import subprocess
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run DVC and Git commands.")
parser.add_argument("commit_message", type=str, help="Commit message for Git")
args = parser.parse_args()

# Define variables
data_file = "apps/googleplaystore.csv"
dvc_file = "apps/googleplaystore.csv.dvc"
gitignore_file = "apps/.gitignore"
commit_message = args.commit_message

# Pull the latest CSV from remote storage
subprocess.run(["dvc", "pull", dvc_file], check=True)

# Run DVC and Git commands to add changes
subprocess.run(["dvc", "add", data_file], check=True)
subprocess.run(["git", "add", dvc_file, gitignore_file, 'run_versioning.ps1',
                'version_apps.py', 'README.md', 'apps/apps.ipynb'], check=True)
subprocess.run(["git", "commit", "-m", commit_message], check=True)
subprocess.run(["dvc", "push"], check=True)
subprocess.run(["git", "push"], check=True)

# Run the training script
subprocess.run(["python", "apps/train_models.py"], check=True)

# Optionally remove the CSV file to keep the directory clean
subprocess.run(["rm", data_file], check=True)
