#!/usr/bin/env python3

# Question 3
# Load a model with pickle and use it to make predictions on new data.

import pickle

def load_model(model_path):
    """Load a machine learning model from a pickle file."""
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

if __name__ == "__main__":
    # Path to the saved model
    model_path = 'pipeline_v1.bin'
    
    # Load the model
    model = load_model(model_path)
    
    # Example new data (replace with actual data)
    new_data = {
        "lead_source": "paid_ads",
        "number_of_courses_viewed": 2,
        "annual_income": 79276.0
    }
    
    # Make predictions
    prediction = model.predict_proba([new_data])[:, 1][0]
    
    # Print the predictions
    print("Prediction:", prediction)