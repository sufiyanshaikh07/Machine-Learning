import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score

# Load dataset
file_path = r"E:\Internship CODveda\Data Set For Task-20250805T162743Z-1-001\Data Set For Task\Churn Prdiction Data\churn-bigml-80.csv"
df = pd.read_csv(file_path)

print("Dataset preview:")
print(df.head())

# Drop 'State' and 'Area code' columns
df = df.drop(columns=['State', 'Area code'])

# Map 'Churn' column: it has 'FALSE' and '1' mixed? Check and clean
df['Churn'] = df['Churn'].astype(str).str.strip().str.upper()
df['Churn'] = df['Churn'].map({'TRUE': 1, 'FALSE': 0, '1': 1, '0': 0})

# Map 'International plan' and 'Voice mail plan'
for col in ['International plan', 'Voice mail plan']:
    df[col] = df[col].astype(str).str.strip().str.lower()
    df[col] = df[col].map({'yes': 1, 'no': 0})

# Drop rows with NaNs after mapping
df = df.dropna()

print(f"Dataset shape after cleaning: {df.shape}")

# Separate features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Convert all features to numeric
X = X.apply(pd.to_numeric)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Define Random Forest with GridSearchCV for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(rf, param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print(f"Best parameters from GridSearchCV: {grid_search.best_params_}")

# Best estimator
best_rf = grid_search.best_estimator_

# Predict on test
y_pred = best_rf.predict(X_test)

# Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nTest set evaluation:")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance plot
importances = best_rf.feature_importances_
features = X.columns

indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title("Feature Importances - Random Forest")
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=45, ha='right')
plt.tight_layout()
plt.show()
