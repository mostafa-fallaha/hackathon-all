from fastapi import FastAPI, HTTPException, Query
from apps.models import AppFeatures
from apps.prediction import predict_app_metrics
from apps.data_loader import fetch_data

app = FastAPI()

@app.post("/predict/")
def predict(features: AppFeatures):
    try:
        result = predict_app_metrics(features)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/data/")
def get_data(skip: int = Query(0), limit: int = Query(10)):
    try:
        df = fetch_data()
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
        row_data = df.iloc[row_id].to_dict()
        return row_data
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
