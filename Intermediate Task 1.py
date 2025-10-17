import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc, confusion_matrix

# Load dataset
file_path = r"E:\Internship CODveda\Data Set For Task-20250805T162743Z-1-001\Data Set For Task\Churn Prdiction Data\churn-bigml-20.csv"
df = pd.read_csv(file_path)

print("Dataset Preview:")
print(df.head())

# Drop 'State' and 'Area code' columns (high cardinality)
df = df.drop(columns=['State', 'Area code'])

# Map 'Churn' column (TRUE/FALSE)
df['Churn'] = df['Churn'].astype(str).str.strip().str.upper()
df['Churn'] = df['Churn'].map({'TRUE': 1, 'FALSE': 0})

# Map 'International plan' and 'Voice mail plan' columns (yes/no)
for col in ['International plan', 'Voice mail plan']:
    df[col] = df[col].astype(str).str.strip().str.lower()
    df[col] = df[col].map({'yes': 1, 'no': 0})

# Drop rows with any NaN values resulting from mapping or missing data
df = df.dropna()

print(f"\nDataset shape after dropping rows with missing values: {df.shape}")

# Separate features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Convert features to numeric (to be safe)
X = X.apply(pd.to_numeric)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split with stratify to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Train logistic regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Predict and get probabilities for ROC
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Evaluation metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\nModel Evaluation:")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {recall:.4f}")
print("Confusion Matrix:")
print(cm)

# Coefficients and odds ratios interpretation
feature_names = X.columns
coefficients = model.coef_[0]
odds_ratios = np.exp(coefficients)

print("\nFeature Coefficients and Odds Ratios:")
for feat, coef, odds in zip(feature_names, coefficients, odds_ratios):
    print(f"{feat:25} Coef: {coef:+.4f}  Odds Ratio: {odds:.4f}")

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()
