from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd

app = FastAPI()

# Load the models
model_installs = joblib.load("models/random_forest_installs.pkl")
model_rating = joblib.load("models/random_forest_rating.pkl")

@app.post("/predict/installs/")
def predict_installs(features: dict):
    try:
        features_df = pd.DataFrame([features])
        prediction = model_installs.predict(features_df)
        return {"predicted_installs": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/rating/")
def predict_rating(features: dict):
    try:
        features_df = pd.DataFrame([features])
        prediction = model_rating.predict(features_df)
        return {"predicted_rating": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
