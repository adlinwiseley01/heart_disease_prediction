# heart_disease_prediction

# Heart Disease Prediction Using Machine Learning
Heart disease is one of the leading causes of mortality worldwide. Predicting the presence of heart disease based on medical data can help in early diagnosis and treatment. This project uses a **Random Forest Classifier** to predict whether a patient is at risk of heart disease.
---

## Technologies Used

- **Programming Language**: Python
- **Libraries**:
  - Pandas
  - NumPy
  - Matplotlib
  - Scikit-learn

---

## Dataset

The dataset used for this project is a CSV file named `heart_disease_uci.csv`. It contains the following features:

1. **age**: Age of the patient
2. **sex**: Gender (1 = Male, 0 = Female)
3. **cp**: Chest pain type (categorical)
4. **trestbps**: Resting blood pressure (mm Hg)
5. **chol**: Serum cholesterol (mg/dl)
6. **fbs**: Fasting blood sugar (> 120 mg/dl, 1 = True, 0 = False)
7. **restecg**: Resting electrocardiographic results
8. **thalach**: Maximum heart rate achieved
9. **exang**: Exercise-induced angina (1 = Yes, 0 = No)
10. **oldpeak**: ST depression induced by exercise relative to rest
11. **slope**: Slope of the peak exercise ST segment
12. **ca**: Number of major vessels (0–3) colored by fluoroscopy
13. **thal**: Thalassemia (categorical)
14. **target**: Presence of heart disease (1 = Yes, 0 = No)

---

## Features

- **Data Preprocessing**:
  - Standardized feature values using `StandardScaler`.
  - Split the dataset into training and testing sets (80/20 split).

- **Machine Learning Model**:
  - **Random Forest Classifier**: Used to predict the target variable (`target`).

- **Model Evaluation**:
  - Confusion Matrix
  - Classification Report
  - Accuracy Score
  - Feature Importance Visualization

---




