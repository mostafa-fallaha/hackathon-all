import os
import pandas as pd

def fetch_data():
    # Command to pull the latest data from the remote storage
    os.system("dvc pull apps/googleplaystore.csv.dvc")
    df = pd.read_csv("apps/googleplaystore.csv")
    return df
