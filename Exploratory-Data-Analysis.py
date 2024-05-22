# exploratory_data_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os

# Define file paths
processed_train_data_path = 'data/processed/train_data.csv'
figures_path = 'figures'

# Create directories if they don't exist
os.makedirs(figures_path, exist_ok=True)

# Load processed training data
print("Loading processed data...")
train_data = pd.read_csv(processed_train_data_path)

# Display the first few rows of the dataset
print("Train Data:")
print(train_data.head())

# Basic statistics
print("Basic Statistics:")
print(train_data.describe())

# Sentiment distribution
print("Sentiment Distribution:")
print(train_data['sentiment'].value_counts(normalize=True))

# Plot sentiment distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='sentiment', data=train_data)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.savefig(os.path.join(figures_path, 'sentiment_distribution.png'))
plt.show()

# Word cloud for positive sentiment
positive_texts = ' '.join(train_data[train_data['sentiment'] == 1]['text'])
wordcloud = WordCloud(width=800, height=400, max_font_size=100, max_words=100, background_color='white').generate(positive_texts)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Word Cloud for Positive Sentiment')
plt.axis('off')
plt.savefig(os.path.join(figures_path, 'wordcloud_positive.png'))
plt.show()

# Word cloud for negative sentiment
negative_texts = ' '.join(train_data[train_data['sentiment'] == 0]['text'])
wordcloud = WordCloud(width=800, height=400, max_font_size=100, max_words=100, background_color='black').generate(negative_texts)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Word Cloud for Negative Sentiment')
plt.axis('off')
plt.savefig(os.path.join(figures_path, 'wordcloud_negative.png'))
plt.show()

# Text length distribution
train_data['text_length'] = train_data['text'].apply(len)

plt.figure(figsize=(12, 6))
sns.histplot(train_data[train_data['sentiment'] == 1]['text_length'], kde=True, color='blue', label='Positive')
sns.histplot(train_data[train_data['sentiment'] == 0]['text_length'], kde=True, color='red', label='Negative')
plt.title('Text Length Distribution by Sentiment')
plt.xlabel('Text Length')
plt.ylabel('Frequency')
plt.legend()
plt.savefig(os.path.join(figures_path, 'text_length_distribution.png'))
plt.show()

print("Exploratory Data Analysis completed!")
