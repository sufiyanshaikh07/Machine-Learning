import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
file_path = r"E:\Internship CODveda\Data Set For Task-20250805T162743Z-1-001\Data Set For Task\3) Sentiment dataset.csv"
df = pd.read_csv(file_path)

print("Dataset preview:")
print(df.head())

# Select relevant numeric features
features = ['Retweets', 'Likes', 'Year', 'Month', 'Day', 'Hour']

# Filter rows with complete data and valid Sentiment labels
df = df.dropna(subset=features + ['Sentiment'])

# Encode Sentiment to binary (Positive=1, Negative=0)
df['Sentiment'] = df['Sentiment'].astype(str).str.strip().str.lower()
df = df[df['Sentiment'].isin(['positive', 'negative'])]
df['Sentiment'] = df['Sentiment'].map({'positive': 1, 'negative': 0})

X = df[features].values
y = df['Sentiment'].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Build neural network model with updated InputLayer argument
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # binary classification output
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model with validation split
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    verbose=2
)

# Evaluate on test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")

# Plot training & validation loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
