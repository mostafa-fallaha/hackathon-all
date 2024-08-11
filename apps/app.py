from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import mlflow.pyfunc
import pandas as pd
import os

app = FastAPI()

# Load the models from MLflow's model registry or artifact store
model_installs = mlflow.pyfunc.load_model("models:/LinearRegression_number_of_installs_Model/1")
model_rating = mlflow.pyfunc.load_model("models:/RandomForestRegressor_rating_Model/1")

# Pydantic model for data validation
class AppFeatures(BaseModel):
    Reviews: int = Field(..., example=50000, description="Number of reviews")
    Size: float = Field(..., example=25.0, description="Size of the app in MB")
    Price: float = Field(..., example=0.99, description="Price of the app in USD")
    last_updated_year: int = Field(..., example=2021, description="Year of the last update")
    last_updated_month: int = Field(..., example=6, description="Month of the last update")
    Category_encoded: int = Field(..., example=2, description="Encoded category of the app")
    Type_Free: int = Field(..., example=1, description="Indicator if the app is free")
    Type_Paid: int = Field(..., example=0, description="Indicator if the app is paid")

# Helper function to fetch data from remote storage using DVC
def fetch_data():
    # Command to pull the latest data from the remote storage
    os.system("dvc pull apps/googleplaystore.csv.dvc")
    df = pd.read_csv("apps/googleplaystore.csv")
    return df

@app.post("/predict/")
def predict(features: AppFeatures):
    try:
        features_df = pd.DataFrame([features.dict()])
        installs_prediction = model_installs.predict(features_df)
        rating_prediction = model_rating.predict(features_df)
        return {
            "predicted_installs": installs_prediction[0],
            "predicted_rating": rating_prediction[0]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/data/")
def get_data(skip: int = Query(0, description="Number of records to skip"),
             limit: int = Query(10, description="Number of records to return")):
    try:
        df = fetch_data()
        # Implement pagination
        data_chunk = df.iloc[skip:skip + limit].to_dict(orient="records")
        return {"data": data_chunk, "total": len(df), "skip": skip, "limit": limit}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/data/{row_id}")
def get_data_row(row_id: int):
    try:
        df = fetch_data()
        if row_id >= len(df):
            raise HTTPException(status_code=404, detail="Row not found")
        row_data = df.iloc[row_id].to_dict()  # Convert a specific row to a dictionary
        return row_data
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))