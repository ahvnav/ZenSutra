import pandas as pd
import os
import joblib # type: ignore # For saving/loading the model efficiently
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report # type: ignore
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- Configuration ---
# Define the path for the raw data CSV
RAW_DATA_PATH = 'customer_data.csv'
# Define the path where the trained model will be saved
MODEL_PATH = 'churn_prediction_model.joblib'
# Define the path for the preprocessor (scaler and encoder)
PREPROCESSOR_PATH = 'preprocessor.joblib'

# --- 1. Mock Data Generation (Same as before, ensures file exists) ---
def generate_mock_data(file_path): # type: ignore
    data = { # type: ignore
        'customer_id': range(1, 101), # Increased data points for better training
        'age': [25 + i % 30 for i in range(100)],
        'gender': ['M' if i % 2 == 0 else 'F' for i in range(100)],
        'monthly_bill': [50.0 + i * 0.5 for i in range(100)],
        'data_usage_gb': [10 + i * 0.3 for i in range(100)],
        'contract_type': ['Monthly' if i % 3 == 0 else ('Annual' if i % 3 == 1 else 'Two Year') for i in range(100)],
        'churn': [1 if i % 5 == 0 or i % 7 == 0 else 0 for i in range(100)] # More varied churn
    }
    df_mock = pd.DataFrame(data)
    if not os.path.exists(file_path): # type: ignore
        df_mock.to_csv(file_path, index=False) # type: ignore
        print(f"Created mock data file: {file_path}")

# --- 2. Data Loading ---
def load_data(file_path): # type: ignore
    try:
        df = pd.read_csv(file_path) # type: ignore
        print(f"Successfully loaded data from {file_path}")
        print(f"Dataset shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Generating mock data...")
        generate_mock_data(file_path) # type: ignore
        return pd.read_csv(file_path) # type: ignore # Try loading again after generating
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return None

# --- 3. Data Preprocessing ---
def preprocess_data(df): # type: ignore
    if df is None:
        return None, None, None

    print("\nStarting data preprocessing...")

    # Drop customer_id as it's not a feature for the model
    df = df.drop('customer_id', axis=1) # type: ignore

    # Separate features (X) and target (y)
    X = df.drop('churn', axis=1) # type: ignore
    y = df['churn'] # type: ignore

    # Handle categorical features using LabelEncoder
    # We need to save the encoder to transform new data later
    gender_encoder = LabelEncoder()
    X['gender'] = gender_encoder.fit_transform(X['gender']) # type: ignore

    contract_encoder = LabelEncoder()
    X['contract_type'] = contract_encoder.fit_transform(X['contract_type']) # type: ignore

    # Scale numerical features
    # We need to save the scaler to transform new data later
    scaler = StandardScaler()
    numerical_cols = ['age', 'monthly_bill', 'data_usage_gb']
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols]) # type: ignore

    print("Data preprocessing complete.")
    return X, y, {'gender_encoder': gender_encoder, 'contract_encoder': contract_encoder, 'scaler': scaler} # type: ignore

# --- 4. Model Training ---
def train_model(X, y): # type: ignore
    print("\nStarting model training...")
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # type: ignore

    # Initialize and train a Logistic Regression model
    model = LogisticRegression(random_state=42, solver='liblinear') # liblinear is good for small datasets
    model.fit(X_train, y_train) # type: ignore
    print("Model training complete.")
    return model, X_test, y_test # type: ignore

# --- 5. Model Evaluation ---
def evaluate_model(model, X_test, y_test): # type: ignore
    print("\nStarting model evaluation...")
    y_pred = model.predict(X_test) # type: ignore
    accuracy = accuracy_score(y_test, y_pred) # type: ignore
    report = classification_report(y_test, y_pred) # type: ignore

    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report) # type: ignore
    print("Model evaluation complete.")
    return accuracy, report # type: ignore

# --- Main Execution Flow ---
if __name__ == "__main__":
    # Ensure mock data is generated if not present
    generate_mock_data(RAW_DATA_PATH)

    # Load data
    df = load_data(RAW_DATA_PATH)

    if df is not None:
        # Preprocess data
        X, y, preprocessors = preprocess_data(df) # type: ignore

        if X is not None and y is not None:
            # Train model
            model, X_test, y_test = train_model(X, y) # type: ignore

            # Evaluate model
            accuracy, report = evaluate_model(model, X_test, y_test) # type: ignore

            # Save model and preprocessors
            try:
                joblib.dump(model, MODEL_PATH) # type: ignore
                joblib.dump(preprocessors, PREPROCESSOR_PATH) # type: ignore
                print(f"\nModel saved to: {MODEL_PATH}")
                print(f"Preprocessors saved to: {PREPROCESSOR_PATH}")
            except Exception as e:
                print(f"Error saving model or preprocessors: {e}")
        else:
            print("Skipping model training due to preprocessing errors.")
    else:
        print("Skipping model training due to data loading errors.")