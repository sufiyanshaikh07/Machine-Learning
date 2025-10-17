import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Path to CSV
csv_path = r"E:\Internship CODveda\Data Set For Task-20250805T162743Z-1-001\Data Set For Task\4) house Prediction Data Set.csv"

# Correct Boston Housing column names
column_names = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
    "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"
]

# 1. Load dataset (space-separated, no header in file)
df = pd.read_csv(csv_path, delim_whitespace=True, header=None, names=column_names)

# 2. Handle missing values (fill numeric with mean)
df = df.fillna(df.mean(numeric_only=True))

# 3. One-Hot Encode the categorical variable 'CHAS'
df = pd.get_dummies(df, columns=['CHAS'], drop_first=True)  # drop_first avoids dummy variable trap

# 4. Define features and target
target = "MEDV"
X = df.drop(columns=[target])
y = df[target]

# 5. Standardize numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 7. Output shapes for verification
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
