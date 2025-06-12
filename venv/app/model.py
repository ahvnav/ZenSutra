# app/model.py
import joblib # type: ignore
import pandas as pd
import os # type: ignore
from sklearn.preprocessing import LabelEncoder, StandardScaler # type: ignore

# Define paths for the model and preprocessors
MODEL_PATH = 'churn_prediction_model.joblib'
PREPROCESSOR_PATH = 'preprocessor.joblib'

# Global variables to hold the loaded model and preprocessors
# These will be loaded once when the application starts
model = None
preprocessors = None

def load_artifacts():
    """
    Loads the trained model and preprocessors from disk.
    This function should be called once at application startup.
    """
    global model, preprocessors
    if model is None or preprocessors is None:
        try:
            model = joblib.load(MODEL_PATH) # type: ignore
            preprocessors = joblib.load(PREPROCESSOR_PATH) # type: ignore
            print(f"Successfully loaded model from {MODEL_PATH}")
            print(f"Successfully loaded preprocessors from {PREPROCESSOR_PATH}")
        except FileNotFoundError:
            print(f"Error: Model or preprocessor files not found. Make sure '{MODEL_PATH}' and '{PREPROCESSOR_PATH}' are in the root directory.")
            print("Please run `python train_model.py` first to generate them.")
            raise FileNotFoundError("ML artifacts not found.")
        except Exception as e:
            print(f"An error occurred while loading artifacts: {e}")
            raise e

def preprocess_single_input(data: dict, preprocessors: dict): # type: ignore
    """
    Preprocesses a single input data point using the loaded preprocessors.
    """
    df_input = pd.DataFrame([data]) # Convert dict to DataFrame

    # Apply label encoders
    gender_encoder = preprocessors['gender_encoder'] # type: ignore
    contract_encoder = preprocessors['contract_encoder'] # type: ignore

    # Ensure consistent handling of unknown categories if they appear (though not expected with Literal)
    # For robust production, you might add error handling for unseen categories
    df_input['gender'] = gender_encoder.transform(df_input['gender']) # type: ignore
    df_input['contract_type'] = contract_encoder.transform(df_input['contract_type']) # type: ignore

    # Apply scaler
    scaler = preprocessors['scaler'] # type: ignore
    numerical_cols = ['age', 'monthly_bill', 'data_usage_gb']
    df_input[numerical_cols] = scaler.transform(df_input[numerical_cols]) # type: ignore

    return df_input

def predict_churn(customer_features: dict): # type: ignore
    """
    Makes a churn prediction for a single customer.
    """
    if model is None or preprocessors is None:
        load_artifacts() # Ensure artifacts are loaded if not already

    # Preprocess the input features
    processed_features = preprocess_single_input(customer_features, preprocessors) # type: ignore

    # Make prediction
    prediction = model.predict(processed_features)[0] # type: ignore # [0] to get single value
    prediction_proba = model.predict_proba(processed_features)[0].tolist() # type: ignore # Convert to list for JSON

    # Convert prediction to human-readable string
    churn_status = "Churn" if prediction == 1 else "No Churn"

    return {
        "prediction": int(prediction), # Ensure it's a standard int for JSON
        "churn_status": churn_status,
        "probability_no_churn": prediction_proba[0],
        "probability_churn": prediction_proba[1]
    } # type: ignore

# Load artifacts when this module is imported (application startup)
try:
    load_artifacts()
except FileNotFoundError:
    print("Warning: Model artifacts not found on startup. They will be loaded on first prediction if `train_model.py` is run first.")