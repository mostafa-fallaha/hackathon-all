import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from data_preprocessing import load_and_preprocess_data
from model_training import train_and_evaluate_model
from predict import predict_new_entries

# Initialize FastAPI app
app = FastAPI()

# Load model
def load_model(filename):
    with open(filename, 'rb') as model_file:
        model = pickle.load(model_file)
    return model

# Load and preprocess data
X, y = load_and_preprocess_data('../../apps/123.csv')
model = train_and_evaluate_model(X, y)  # Train model if not already trained

# Define request body schema
class PredictionRequest(BaseModel):
    Android_Ver: str
    Size: float
    Price: int
    Category_encoded: int
    Type_Free: int
    Type_Paid: int
    Content_Ratings_Adults_only_18: int
    Content_Ratings_Everyone: int
    Content_Ratings_Everyone_10: int
    Content_Ratings_Mature_17: int
    Content_Ratings_Teen: int
    Content_Ratings_Unrated: int
    last_updated_year: int
    last_updated_month: int

# Define a root endpoint
@app.get("/")
def read_root():
    return {"message": "Hello World"}

# Prediction endpoint
@app.post("/predict/rating")
def predict(request: PredictionRequest):
    try:
        # Convert request data to DataFrame with one row
        new_entry = pd.DataFrame([{
            'Android Ver': request.Android_Ver,
            'Size': request.Size,
            'Price': request.Price,
            'Category_encoded': request.Category_encoded,
            'Type_Free': request.Type_Free,
            'Type_Paid': request.Type_Paid,
            'Content Ratings (Adults only 18+)': request.Content_Ratings_Adults_only_18,
            'Content Ratings (Everyone)': request.Content_Ratings_Everyone,
            'Content Ratings (Everyone 10+)': request.Content_Ratings_Everyone_10,
            'Content Ratings (Mature 17+)': request.Content_Ratings_Mature_17,
            'Content Ratings (Teen)': request.Content_Ratings_Teen,
            'Content Ratings (Unrated)': request.Content_Ratings_Unrated,
            'last_updated_year': request.last_updated_year,
            'last_updated_month': request.last_updated_month
        }])

        # Make predictions
        predictions = predict_new_entries(model, new_entry)
        return {"rating_predictions": predictions.tolist(),"installs_predictions": predictions.tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI app with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7500)
