"""
Multi-Modal Learning Prototype for Jarvis AI Platform

This script demonstrates a simple multi-modal pipeline combining tabular and image data.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skimage.feature import hog
import matplotlib.pyplot as plt
from PIL import Image
import os

# --- Configurable paths ---
TABULAR_DATA_PATH = "data/new_data.csv"
IMAGE_FOLDER = "data/images/"  # Folder with images named by row id or key


print("Loading tabular data from:", TABULAR_DATA_PATH)
df = pd.read_csv(TABULAR_DATA_PATH)
X_tabular = df.drop(["target", "image_id"], axis=1)
y = df["target"]
image_ids = df["image_id"]
print(f"Loaded {len(df)} rows of tabular data.")

# --- Step 2: Load and flatten images ---

def extract_image_features(image_path):
    try:
        img = Image.open(image_path).convert("L").resize((32, 32))
        arr = np.array(img)
        # Extract HOG features for more advanced image representation
        hog_features = hog(arr, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
        return hog_features
    except Exception as e:
        print(f"Warning: Could not load image {image_path}: {e}")
        # Return a zero array if image is missing or corrupt
        return np.zeros((144,), dtype=np.float32)  # HOG feature length for 32x32, 8x8, 2x2



print(f"Loading and extracting HOG features from images in: {IMAGE_FOLDER}")
image_features = []
missing_images = 0
for img_id in image_ids:
    img_path = os.path.join(IMAGE_FOLDER, f"{img_id}.png")
    if not os.path.exists(img_path):
        print(f"Warning: Image not found for id {img_id} at {img_path}")
        missing_images += 1
    image_features.append(extract_image_features(img_path))
X_image = np.array(image_features)
if missing_images > 0:
    print(f"{missing_images} images were missing and replaced with zeros.")



print("Concatenating tabular and image (HOG) features...")
X_combined = np.hstack([X_tabular.values, X_image])



print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)
print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

# Hyperparameter tuning with GridSearchCV
print("Tuning RandomForestClassifier hyperparameters with cross-validation...")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 4]
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1)
grid.fit(X_train, y_train)
print(f"Best parameters: {grid.best_params_}")
model = grid.best_estimator_

# Cross-validation score on training set
cv_scores = cross_val_score(model, X_train, y_train, cv=3)
print(f"Cross-validation accuracy (train set): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


# Final evaluation on test set
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test set accuracy: {acc:.4f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance visualization
print("\nPlotting feature importances...")
importances = model.feature_importances_
num_tabular = X_tabular.shape[1]
num_image = X_image.shape[1]
feature_names = list(X_tabular.columns) + [f"img_{i}" for i in range(num_image)]
indices = importances.argsort()[::-1][:10]  # Top 10 features
plt.figure(figsize=(10, 5))
plt.title("Top 10 Feature Importances (Tabular + HOG Image)")
plt.bar(range(len(indices)), importances[indices], align="center")
plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
plt.tight_layout()
plt.show()
