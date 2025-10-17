import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

# Load dataset
file_path = r"E:\Internship CODveda\Data Set For Task-20250805T162743Z-1-001\Data Set For Task\1) iris.csv"
df = pd.read_csv(file_path)

print("Dataset Preview:")
print(df.head())

# Define features and target explicitly
feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
target_col = 'species'

X = df[feature_cols].values
y_raw = df[target_col]

# Encode species labels (e.g., setosa, versicolor, virginica) into integers
le = LabelEncoder()
y = le.fit_transform(y_raw)

# Standardize features for KNN
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split with stratify to keep class balance
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

k_values = [1, 3, 5, 7, 9]
accuracy_list = []
precision_list = []
recall_list = []

for k in k_values:
    print(f"\n--- KNN with k={k} ---")
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    accuracy_list.append(acc)
    precision_list.append(prec)
    recall_list.append(recall)

    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {recall:.4f}")

# Plot accuracy for different k values
plt.figure(figsize=(8, 6))
plt.bar([str(k) for k in k_values], accuracy_list, color='skyblue')
plt.title("KNN Classifier Accuracy for Different K Values on Iris Dataset")
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
