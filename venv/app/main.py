# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from app.schemas import CustomerFeatures # type: ignore # Import our input schema
from app.model import predict_churn # type: ignore # Import our prediction function

# Initialize the FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="A real-time API to predict customer churn based on various features.",
    version="1.0.0"
)

# --- Root Endpoint ---
@app.get("/", response_class=HTMLResponse, summary="Health Check")
async def read_root():
    """
    Provides a simple health check for the API.
    """
    return """
    <html>
        <head>
            <title>Churn Prediction API</title>
        </head>
        <body>
            <h1>Customer Churn Prediction API</h1>
            <p>Visit <a href="/docs">/docs</a> for the API documentation.</p>
        </body>
    </html>
    """

# --- Prediction Endpoint ---
@app.post("/predict_churn", summary="Predict Customer Churn",
          description="Predicts whether a customer is likely to churn (1) or not (0) based on input features.")
async def predict_churn_endpoint(features: CustomerFeatures): # type: ignore
    """
    Receives customer features and returns a churn prediction.

    - **age**: Customer's age (int)
    - **gender**: Customer's gender ('M' or 'F')
    - **monthly_bill**: Customer's average monthly bill (float)
    - **data_usage_gb**: Customer's average data usage in GB (float)
    - **contract_type**: Type of contract ('Monthly', 'Annual', 'Two Year')

    Returns:
    - **prediction**: 0 (No Churn) or 1 (Churn)
    - **churn_status**: "No Churn" or "Churn"
    - **probability_no_churn**: Probability of not churning
    - **probability_churn**: Probability of churning
    """
    try:
        # Convert Pydantic model to dictionary for the prediction function
        prediction_result = predict_churn(features.model_dump()) # type: ignore
        return prediction_result # type: ignore
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Model artifacts not found. Please ensure `train_model.py` has been run: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")