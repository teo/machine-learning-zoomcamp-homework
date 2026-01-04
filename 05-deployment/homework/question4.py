#!/usr/bin/env python3

# Serve a model with FastAPI to make predictions via an API endpoint.

import pickle
from fastapi import FastAPI

def load_model(model_path):
    """Load a machine learning model from a pickle file."""
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

app = FastAPI()
model = load_model('pipeline_v1.bin')
@app.post("/predict/")
def predict(data: dict):
    """Make predictions using the loaded model."""
    prediction = model.predict_proba([data])[:, 1][0]
    return {"prediction": prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# client in ipython3:
# import requests
# url = "http://127.0.0.1:8000/predict/"
# client = {
#     "lead_source": "organic_search",
#     "number_of_courses_viewed": 4,
#     "annual_income": 80304.0
# }
# requests.post(url, json=client).json()["prediction"]
