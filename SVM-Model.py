# svm_model.py

import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import os

# Define file paths
train_features_path = 'data/processed/train_features.pkl'
test_features_path = 'data/processed/test_features.pkl'
model_path = 'models/svm_model.pkl'
results_path = 'results/svm_report.txt'

# Create directories if they don't exist
os.makedirs(os.path.dirname(model_path), exist_ok=True)
os.makedirs(os.path.dirname(results_path), exist_ok=True)

# Load features and labels
print("Loading features and labels...")
X_train_tfidf, y_train = joblib.load(train_features_path)
X_test_tfidf, y_test = joblib.load(test_features_path)

# Build SVM model
print("Building SVM model...")
model = SVC(random_state=42, probability=True)
model.fit(X_train_tfidf, y_train)

# Predict on the test set
print("Predicting on the test set...")
y_pred = model.predict(X_test_tfidf)
y_prob = model.predict_proba(X_test_tfidf)[:, 1]

# Evaluate the model
print("Evaluating the model...")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
report = classification_report(y_test, y_pred)

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
joblib.dump(model, model_path)

print("SVM model training and evaluation completed!")
