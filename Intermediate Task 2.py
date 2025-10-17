import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Load dataset
file_path = r"E:\Internship CODveda\Data Set For Task-20250805T162743Z-1-001\Data Set For Task\1) iris.csv"
df = pd.read_csv(file_path)

print("Dataset preview:")
print(df.head())

# Features and labels
X = df.drop('species', axis=1)
y = LabelEncoder().fit_transform(df['species'])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train decision tree classifier with pruning (limit max depth)
clf = DecisionTreeClassifier(random_state=42, max_depth=3, min_samples_leaf=5)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluate
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\nAccuracy: {acc:.4f}")
print(f"Weighted F1-score: {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=LabelEncoder().fit(df['species']).classes_))

# Visualize decision tree
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=X.columns, class_names=LabelEncoder().fit(df['species']).classes_,
          filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree Classifier - Iris Dataset")
plt.show()
