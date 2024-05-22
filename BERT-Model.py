# bert_model.py

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import TextClassificationPipeline
from sklearn.model_selection import train_test_split
import joblib
import os

# Define file paths
train_data_path = 'data/processed/train_data.csv'
test_data_path = 'data/processed/test_data.csv'
bert_model_path = 'models/bert_model'
results_path = 'results/bert_report.txt'

# Create directories if they don't exist
os.makedirs(bert_model_path, exist_ok=True)
os.makedirs(os.path.dirname(results_path), exist_ok=True)

# Load processed training and testing data
print("Loading processed data...")
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_data['text'], train_data['sentiment'], test_size=0.2, random_state=42)

# Initialize the BERT tokenizer
print("Initializing BERT tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the data
print("Tokenizing data...")
train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(list(X_val), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(list(test_data['text']), truncation=True, padding=True, max_length=128)

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SentimentDataset(train_encodings, list(y_train))
val_dataset = SentimentDataset(val_encodings, list(y_val))
test_dataset = SentimentDataset(test_encodings, list(test_data['sentiment']))

# Load pre-trained BERT model
print("Loading pre-trained BERT model...")
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir=bert_model_path,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch"
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train the model
print("Training the model...")
trainer.train()

# Evaluate the model on the test set
print("Evaluating the model...")
test_pred = trainer.predict(test_dataset)
y_pred = np.argmax(test_pred.predictions, axis=1)
y_prob = test_pred.predictions[:, 1]

accuracy = accuracy_score(test_data['sentiment'], y_pred)
precision = precision_score(test_data['sentiment'], y_pred)
recall = recall_score(test_data['sentiment'], y_pred)
f1 = f1_score(test_data['sentiment'], y_pred)
roc_auc = roc_auc_score(test_data['sentiment'], y_prob)
report = classification_report(test_data['sentiment'], y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print("Classification Report:")
print(report)

# Save the evaluation report
with open(results_path, 'w') as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"ROC-AUC: {roc_auc:.4f}\n")
    f.write("Classification Report:\n")
    f.write(report)

# Save the model
print("Saving the model...")
model.save_pretrained(bert_model_path)
tokenizer.save_pretrained(bert_model_path)

print("BERT model training and evaluation completed!")
