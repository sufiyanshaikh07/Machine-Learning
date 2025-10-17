import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load dataset
file_path = r"E:\Internship CODveda\Data Set For Task-20250805T162743Z-1-001\Data Set For Task\3) Sentiment dataset.csv"
df = pd.read_csv(file_path)

print("Dataset preview:")
print(df.head())

# Select numeric columns for clustering
# From your columns: Retweets, Likes, Year, Month, Day, Hour
num_cols = ['Retweets', 'Likes', 'Year', 'Month', 'Day', 'Hour']

X = df[num_cols]

# Handle missing values if any by dropping rows
X = X.dropna()

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow method to find optimal number of clusters
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(k_range, inertia, 'bo-', markersize=8)
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters k')
plt.ylabel('Inertia (Sum of squared distances)')
plt.grid(True)
plt.show()

# Based on elbow plot choose k (say k=3 for example)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to dataframe
df['Cluster'] = clusters

# Reduce dimensions for 2D visualization using PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 7))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=clusters, palette='Set2', s=80)
plt.title(f'K-Means Clustering with k={optimal_k}')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# Interpret clusters by showing mean values per cluster
print("\nCluster centroids (in scaled features):")
print(kmeans.cluster_centers_)

print("\nCluster summary (original scale):")
print(df.groupby('Cluster')[num_cols].mean())
