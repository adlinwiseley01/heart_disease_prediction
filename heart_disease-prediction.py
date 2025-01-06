# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv(r'C:\Users\adlin\OneDrive\Desktop\machine learning projects\heart_disease_uci.csv')

# Inspect the dataset structure
print("First 5 rows of the dataset:\n", data.head())
print("\nDataset Info:\n")
print(data.info())

# Check for missing values
print("\nMissing values per column:\n", data.isnull().sum())

# Ensure the correct target column is used
# Replace 'target' with the actual name of your target column
if 'target' not in data.columns:
    print("\nError: 'target' column not found. Please check the correct target column name.")
else:
    # Separate features (X) and target variable (y)
    X = data.drop('target', axis=1)
    y = data['target']

    # Standardize the feature values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Model building
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Model evaluation
    y_pred = model.predict(X_test)
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

    # Visualization: Feature importance
    feature_importances = model.feature_importances_
    features = X.columns  # Feature names

    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(features, feature_importances, color='skyblue')
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.title("Feature Importance")
    plt.gca().invert_yaxis()  # Invert y-axis for better readability
    plt.show()
