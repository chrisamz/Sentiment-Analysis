# feature_engineering.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

# Define file paths
processed_train_data_path = 'data/processed/train_data.csv'
processed_test_data_path = 'data/processed/test_data.csv'
train_features_path = 'data/processed/train_features.pkl'
test_features_path = 'data/processed/test_features.pkl'
vectorizer_path = 'models/tfidf_vectorizer.pkl'

# Create directories if they don't exist
os.makedirs(os.path.dirname(train_features_path), exist_ok=True)
os.makedirs(os.path.dirname(test_features_path), exist_ok=True)
os.makedirs(os.path.dirname(vectorizer_path), exist_ok=True)

# Load processed training and testing data
print("Loading processed data...")
train_data = pd.read_csv(processed_train_data_path)
test_data = pd.read_csv(processed_test_data_path)

# Text data
X_train_text = train_data['text']
y_train = train_data['sentiment']
X_test_text = test_data['text']
y_test = test_data['sentiment']

# Initialize TF-IDF Vectorizer
print("Initializing TF-IDF Vectorizer...")
tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))

# Fit and transform the training data
print("Fitting and transforming training data...")
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)

# Transform the testing data
print("Transforming testing data...")
X_test_tfidf = tfidf_vectorizer.transform(X_test_text)

# Save the features and labels
print("Saving features and labels...")
joblib.dump((X_train_tfidf, y_train), train_features_path)
joblib.dump((X_test_tfidf, y_test), test_features_path)

# Save the TF-IDF vectorizer
print("Saving TF-IDF Vectorizer...")
joblib.dump(tfidf_vectorizer, vectorizer_path)

print("Feature engineering completed!")
