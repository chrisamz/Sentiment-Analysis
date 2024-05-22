# data_preprocessing.py

import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# Define file paths
raw_data_path = 'data/raw/reviews.csv'
processed_train_data_path = 'data/processed/train_data.csv'
processed_test_data_path = 'data/processed/test_data.csv'

# Create directories if they don't exist
os.makedirs(os.path.dirname(processed_train_data_path), exist_ok=True)

# Function to clean text data
def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Convert to lower case
    text = text.lower()
    # Remove extra whitespace
    text = text.strip()
    return text

# Load raw data
print("Loading raw data...")
data = pd.read_csv(raw_data_path)

# Display the first few rows of the dataset
print("Raw Data:")
print(data.head())

# Data Cleaning
print("Cleaning data...")

# Apply text cleaning
data['text'] = data['text'].apply(clean_text)

# Encode labels if necessary (assuming binary sentiment: positive/negative)
print("Encoding labels...")
label_encoder = LabelEncoder()
data['sentiment'] = label_encoder.fit_transform(data['sentiment'])

# Split the data into training and testing sets
print("Splitting data into training and testing sets...")
X = data['text']
y = data['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save processed data
print("Saving processed data...")
train_data = pd.DataFrame({'text': X_train, 'sentiment': y_train})
test_data = pd.DataFrame({'text': X_test, 'sentiment': y_test})

train_data.to_csv(processed_train_data_path, index=False)
test_data.to_csv(processed_test_data_path, index=False)

print("Data preprocessing completed!")
