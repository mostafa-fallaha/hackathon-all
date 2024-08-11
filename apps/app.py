from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow.pyfunc
import pandas as pd

app = FastAPI()

# Load the models from MLflow's model registry or artifact store
model_installs = mlflow.pyfunc.load_model("models:/LinearRegression_number_of_installs_Model/1")
model_rating = mlflow.pyfunc.load_model("models:/RandomForestRegressor_rating_Model/1")

# Pydantic models for data validation

class InstallFeatures(BaseModel):
    Reviews: int = Field(..., example=50000, description="Number of reviews")
    Size: float = Field(..., example=25.0, description="Size of the app in MB")
    Price: float = Field(..., example=0.99, description="Price of the app in USD")
    last_updated_year: int = Field(..., example=2021, description="Year of the last update")
    last_updated_month: int = Field(..., example=6, description="Month of the last update")
    Category_encoded: int = Field(..., example=2, description="Encoded category of the app")
    Type_Free: int = Field(..., example=1, description="Indicator if the app is free")
    Type_Paid: int = Field(..., example=0, description="Indicator if the app is paid")

class RatingFeatures(BaseModel):
    Reviews: int = Field(..., example=200, description="Number of reviews")
    Size: float = Field(..., example=15.0, description="Size of the app in MB")
    Price: float = Field(..., example=0.0, description="Price of the app in USD")
    last_updated_year: int = Field(..., example=2022, description="Year of the last update")
    last_updated_month: int = Field(..., example=3, description="Month of the last update")
    Category_encoded: int = Field(..., example=1, description="Encoded category of the app")
    Type_Free: int = Field(..., example=1, description="Indicator if the app is free")
    Type_Paid: int = Field(..., example=0, description="Indicator if the app is paid")

@app.post("/predict/installs/")
def predict_installs(features: InstallFeatures):
    try:
        features_df = pd.DataFrame([features.dict()])
        prediction = model_installs.predict(features_df)
        return {"predicted_installs": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/rating/")
def predict_rating(features: RatingFeatures):
    try:
        features_df = pd.DataFrame([features.dict()])
        prediction = model_rating.predict(features_df)
        return {"predicted_rating": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
