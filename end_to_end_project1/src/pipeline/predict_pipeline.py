import pandas as pd
import joblib
import sys
import os
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.components.data_ingestion import load_data
from src.components.data_transformation import transform_data
from src.components.model_trainer import get_best_model

# Load data
def train_and_save_model():
    # Load data
    df = load_data()

    # Transform data
    X_train, X_test, y_train, y_test, preprocessor = transform_data(df)

    # Train and get the best model
    best_model = get_best_model(X_train, y_train, X_test, y_test)

    # Save model and preprocessor
    joblib.dump(best_model, "best_model.pkl")
    joblib.dump(preprocessor, "preprocessor.pkl")

    print("Model and preprocessor saved successfully.")
    return best_model, preprocessor

def predict_new_data(new_data: pd.DataFrame):
    # Load preprocessor and model
    preprocessor = joblib.load("preprocessor.pkl")
    model = joblib.load("best_model.pkl")

    # Transform new data
    new_data_transformed = preprocessor.transform(new_data)

    # Make prediction
    prediction = model.predict(new_data_transformed)
    return prediction