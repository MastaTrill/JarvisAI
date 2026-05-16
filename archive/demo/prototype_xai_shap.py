"""
Explainable AI (XAI) Prototype for Jarvis AI Platform

This script demonstrates model interpretability using SHAP for a scikit-learn model.
"""

import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

MODEL_PATH = "models/latest_model.pkl"
DATA_PATH = "data/new_data.csv"

# Load model and data
def load_model(path):
    return joblib.load(path)

def load_data(path):
    df = pd.read_csv(path)
    X = df.drop("target", axis=1)
    return X

def main():
    model = load_model(MODEL_PATH)
    X = load_data(DATA_PATH)

    # Create SHAP explainer
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    # Summary plot
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig("shap_summary_plot.png")
    print("SHAP summary plot saved as shap_summary_plot.png")

if __name__ == "__main__":
    main()
