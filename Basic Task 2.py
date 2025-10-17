import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Path to your dataset
csv_path = r"E:\Internship CODveda\Data Set For Task-20250805T162743Z-1-001\Data Set For Task\4) house Prediction Data Set.csv"

# Column names for Boston Housing dataset
column_names = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
    "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"
]

# 1. Load dataset (space-separated, no header)
df = pd.read_csv(csv_path, delim_whitespace=True, header=None, names=column_names)

# 2. Handle missing values (fill numeric columns with mean)
df = df.fillna(df.mean(numeric_only=True))

# 3. One-Hot Encode the categorical variable 'CHAS'
df = pd.get_dummies(df, columns=['CHAS'], drop_first=True)

# 4. Features (X) and target (y)
X = df.drop(columns=["MEDV"])
y = df["MEDV"]

# 5. Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 7. Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 8. Make predictions
y_pred = model.predict(X_test)

# 9. Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 10. Print coefficients and metrics
print("\nModel Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.4f}")

print(f"\nIntercept (b0): {model.intercept_:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R²): {r2:.4f}")

# 11. Interpretation
print("\nInterpretation:")
print("Each coefficient shows the change in predicted house price (in $1000s) for a one standard deviation change in the feature, holding all else constant.")

# 12. Visualization - Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red', linewidth=2)
plt.xlabel("Actual Prices (MEDV)")
plt.ylabel("Predicted Prices (MEDV)")
plt.title("Actual vs Predicted House Prices")
plt.show()
