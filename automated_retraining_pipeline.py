"""
Automated Retraining Pipeline for Continuous Learning

This script orchestrates data ingestion, model retraining, evaluation, and deployment using the MLFlowTracker.
"""

from src.tracking.mlflow_integration import MLFlowTracker
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- Configurable paths and settings ---
DATA_PATH = "data/new_data.csv"  # Update as needed
MODEL_OUTPUT_PATH = "models/latest_model.pkl"
EXPERIMENT_NAME = "Jarvis_AI_Continuous_Learning"

# --- Step 1: Data Ingestion ---
def ingest_data(path):
    df = pd.read_csv(path)
    # Add validation/cleaning as needed
    return df

def main():
    tracker = MLFlowTracker(experiment_name=EXPERIMENT_NAME)
    run_id = tracker.start_run(run_name="automated_retraining")

    # Ingest and split data
    df = ingest_data(DATA_PATH)
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Log parameters
    tracker.log_params({"model_type": "RandomForestClassifier", "n_features": X.shape[1]})

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    tracker.log_metrics({"accuracy": acc})

    # Log model
    tracker.log_model(model, model_name="rf_model", flavor="sklearn", registered_model_name="Jarvis_RF")

    # Save model artifact
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT_PATH)

    # Optionally: Add deployment logic here
    print(f"Retraining complete. Accuracy: {acc:.4f}")
    print(f"Model saved to {MODEL_OUTPUT_PATH}")

if __name__ == "__main__":
    main()
