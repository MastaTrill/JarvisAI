"""
Automated Retraining Pipeline for
Continuous Learning

This script orchestrates data ingestion,
model retraining, evaluation, and deployment
using the MLFlowTracker.
"""

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from src.tracking.mlflow_integration import MLFlowTracker

# --- Configurable paths and settings ---
DATA_PATH = "data/new_data.csv"  # Update as needed
MODEL_OUTPUT_PATH = "models/latest_model.pkl"
EXPERIMENT_NAME = "Jarvis_AI_Continuous_Learning"

# --- Step 1: Data Ingestion ---
def ingest_data(path):
    """
    Ingests data from the given CSV path.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    df = pd.read_csv(path)
    # Add validation/cleaning as needed
    return df

def main():
    """
    Main function to run the automated retraining pipeline.
    """
    tracker = MLFlowTracker(
        experiment_name=EXPERIMENT_NAME
    )
    tracker.start_run(
        run_name="automated_retraining"
    )

    # Ingest and split data
    df = ingest_data(DATA_PATH)
    features = df.drop(
        "target", axis=1
    )
    target = df["target"]
    features_train, features_test, target_train, target_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42
    )

    # Log parameters
    tracker.log_params({
        "model_type": "RandomForestClassifier",
        "n_features": features.shape[1]
    })

    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(features_train, target_train)

    # Evaluate
    predictions = model.predict(features_test)
    accuracy = float(accuracy_score(target_test, predictions))
    tracker.log_metrics({
        "accuracy": accuracy
    })

    # Log model
    tracker.log_model(
        model,
        model_name="rf_model",
        flavor="sklearn",
        registered_model_name="Jarvis_RF"
    )

    # Save model artifact
    os.makedirs(
        os.path.dirname(MODEL_OUTPUT_PATH),
        exist_ok=True
    )
    joblib.dump(model, MODEL_OUTPUT_PATH)

    # Optionally: Add deployment logic here
    print(
        f"Retraining complete. Accuracy: {accuracy:.4f}"
    )
    print(
        f"Model saved to {MODEL_OUTPUT_PATH}"
    )

if __name__ == "__main__":
    main()
