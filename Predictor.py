import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from fancyimpute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Load your dataset (replace 'data.csv' with your file path)
data = pd.read_csv('data.csv')

# Split features and target variable
X = data.drop('target', axis=1)
y = data['target']

# Impute missing values using MICE (IterativeImputer)
imputer = IterativeImputer()
X_imputed = imputer.fit_transform(X)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Apply SMOTE for oversampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)


# Train a logistic regression model
model_LR = LogisticRegression(random_state=42)
model_LR.fit(X_train, y_train)

# Make predictions
y_pred_LR = model_LR.predict(X_test)

# Calculate evaluation metrics
f1_LR = f1_score(y_test, y_pred_LR)
accuracy_LR = accuracy_score(y_test, y_pred_LR)

print("F1 Score_LR:", f1_LR)
print("Accuracy_LR:", accuracy_LR)



# Train a random forest model
model_RF = RandomForestClassifier(random_state=42)
model_RF.fit(X_train, y_train)

# Make predictions
y_pred_RF = model_RF.predict(X_test)

# Calculate evaluation metrics
f1_RF = f1_score(y_test, y_pred_RF)
accuracy_RF = accuracy_score(y_test, y_pred_RF)

print("F1 Score_RF:", f1_RF)
print("Accuracy_RF:", accuracy_RF)


# Train an XGBoost model
model_XG = XGBClassifier(random_state=42)
model_XG.fit(X_train, y_train)

# Make predictions
y_pred_XG = model_XG.predict(X_test)

# Calculate evaluation metrics
f1_XG = f1_score(y_test, y_pred_XG)
accuracy_XG = accuracy_score(y_test, y_pred_XG)

print("F1 Score_XG:", f1_XG)
print("Accuracy_XG:", accuracy_XG)
















