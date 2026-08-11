# CodeAlpha Internship Task 1
# Iris Flower Classification
# Author: Muhammad Abdullah

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load Dataset

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = pd.read_csv("Iris.csv")

# Display Dataset Information

print("\nFirst 5 Rows\n")
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nColumn Names")
print(df.columns)

print("\nDataset Information")
print(df.info())

print("\nStatistical Summary")
print(df.describe())

print("\nMissing Values")
print(df.isnull().sum())

print("\nFlower Species")
print(df["Species"].unique())

# Data Cleaning

print("\nRemoving Id Column...")

df.drop("Id", axis=1, inplace=True)

print("\nUpdated Columns")
print(df.columns)

# Data Visualization

print("\nCreating Visualizations...")

# Count Plot

plt.figure(figsize=(7,5))
sns.countplot(x="Species", data=df)
plt.title("Flower Species Count")
plt.savefig("species_count.png")
plt.show()

# Pair Plot

sns.pairplot(df, hue="Species")
plt.savefig("pairplot.png")
plt.show()

# Correlation Heatmap

plt.figure(figsize=(8,6))

numeric_df = df.drop("Species", axis=1)

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="Blues"
)

plt.title("Correlation Heatmap")
plt.savefig("heatmap.png")
plt.show()

# Prepare Data

X = df.drop("Species", axis=1)

y = df["Species"]

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples :", len(X_test))

# Build Machine Learning Model

print("\nTraining Random Forest Model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction

y_pred = model.predict(X_test)

# Model Evaluation

accuracy = accuracy_score(y_test, y_pred)

print("\n")
print("=" * 60)
print("MODEL ACCURACY")
print("=" * 60)

print(f"Accuracy : {accuracy*100:.2f}%")

print("\nClassification Report\n")

print(classification_report(y_test, y_pred))

# Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    cmap="Greens",
    fmt="d",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png")

plt.show()

# Feature Importance

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance\n")
print(importance)

plt.figure(figsize=(8,5))

sns.barplot(
    x="Importance",
    y="Feature",
    data=importance
)

plt.title("Feature Importance")

plt.savefig("feature_importance.png")

plt.show()

# Save Model

joblib.dump(model, "model.pkl")

print("\nModel saved successfully as model.pkl")

# Sample Prediction

sample = pd.DataFrame({
    "SepalLengthCm": [5.1],
    "SepalWidthCm": [3.5],
    "PetalLengthCm": [1.4],
    "PetalWidthCm": [0.2]
})

prediction = model.predict(sample)

print("\nSample Flower Prediction")
print(prediction)

print("\nProject Completed Successfully!")
