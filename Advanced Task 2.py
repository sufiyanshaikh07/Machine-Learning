import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.decomposition import PCA

# Load dataset
file_path = r"E:\Internship CODveda\Data Set For Task-20250805T162743Z-1-001\Data Set For Task\3) Sentiment dataset.csv"
df = pd.read_csv(file_path)

print("Dataset preview:")
print(df.head())

# Keep only relevant numeric features for classification
features = ['Retweets', 'Likes', 'Year', 'Month', 'Day', 'Hour']

# Drop rows with missing values in these columns or Sentiment
df = df.dropna(subset=features + ['Sentiment'])

# Encode Sentiment to binary: Assume Positive=1, Negative=0 (adjust if other labels)
df['Sentiment'] = df['Sentiment'].astype(str).str.strip().str.lower()
df = df[df['Sentiment'].isin(['positive', 'negative'])]  # filter only these two classes
df['Sentiment'] = df['Sentiment'].map({'positive': 1, 'negative': 0})

X = df[features]
y = df['Sentiment']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split (stratify to preserve class distribution)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Train SVM with linear kernel
svm_linear = SVC(kernel='linear', probability=True, random_state=42)
svm_linear.fit(X_train, y_train)
y_pred_linear = svm_linear.predict(X_test)
y_proba_linear = svm_linear.predict_proba(X_test)[:,1]

# Train SVM with RBF kernel
svm_rbf = SVC(kernel='rbf', probability=True, random_state=42)
svm_rbf.fit(X_train, y_train)
y_pred_rbf = svm_rbf.predict(X_test)
y_proba_rbf = svm_rbf.predict_proba(X_test)[:,1]

# Evaluation function
def evaluate_model(y_true, y_pred, y_proba, model_name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"AUC: {auc:.4f}")

evaluate_model(y_test, y_pred_linear, y_proba_linear, "SVM Linear Kernel")
evaluate_model(y_test, y_pred_rbf, y_proba_rbf, "SVM RBF Kernel")

# Visualize decision boundary using PCA (2D)
pca = PCA(n_components=2, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

def plot_decision_boundary(clf, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:,1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )
    Z = clf.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8,6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    scatter = plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.coolwarm, edgecolor='k', s=50)
    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.grid(True)
    plt.show()

plot_decision_boundary(svm_linear, X_test_pca, y_test.values, "SVM Linear Kernel Decision Boundary (PCA space)")
plot_decision_boundary(svm_rbf, X_test_pca, y_test.values, "SVM RBF Kernel Decision Boundary (PCA space)")
