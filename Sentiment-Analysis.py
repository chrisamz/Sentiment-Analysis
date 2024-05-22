# sentiment_analysis.py

import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import class_weight
import os

# Define file paths
train_features_path = 'data/processed/train_features.pkl'
test_features_path = 'data/processed/test_features.pkl'
logistic_regression_model_path = 'models/logistic_regression_model.pkl'
svm_model_path = 'models/svm_model.pkl'
random_forest_model_path = 'models/random_forest_model.pkl'
lstm_model_path = 'models/lstm_model.h5'
cnn_model_path = 'models/cnn_model.h5'
bert_model_path = 'models/bert_model'
results_path = 'results/sentiment_analysis_report.txt'

# Create directories if they don't exist
os.makedirs(os.path.dirname(results_path), exist_ok=True)

# Load features and labels
print("Loading features and labels...")
X_train_tfidf, y_train = joblib.load(train_features_path)
X_test_tfidf, y_test = joblib.load(test_features_path)

# Function to evaluate and save model
def evaluate_and_save_model(model, model_name, model_path):
    print(f"Evaluating {model_name}...")
    y_pred = model.predict(X_test_tfidf)
    y_prob = model.predict_proba(X_test_tfidf)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred)
    
    print(f"{model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("Classification Report:")
    print(report)
    
    with open(results_path, 'a') as f:
        f.write(f"{model_name} Performance:\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"ROC-AUC: {roc_auc:.4f}\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n")
    
    print(f"Saving {model_name}...")
    joblib.dump(model, model_path)

# Logistic Regression
print("Training Logistic Regression model...")
logistic_regression_model = LogisticRegression(random_state=42, max_iter=1000)
logistic_regression_model.fit(X_train_tfidf, y_train)
evaluate_and_save_model(logistic_regression_model, 'Logistic Regression', logistic_regression_model_path)

# Support Vector Machine
print("Training SVM model...")
svm_model = SVC(random_state=42, probability=True)
svm_model.fit(X_train_tfidf, y_train)
evaluate_and_save_model(svm_model, 'SVM', svm_model_path)

# Random Forest
print("Training Random Forest model...")
random_forest_model = RandomForestClassifier(random_state=42, n_estimators=100)
random_forest_model.fit(X_train_tfidf, y_train)
evaluate_and_save_model(random_forest_model, 'Random Forest', random_forest_model_path)

# Neural Network (LSTM)
print("Training LSTM model...")
lstm_model = Sequential()
lstm_model.add(Dense(64, input_dim=X_train_tfidf.shape[1], activation='relu'))
lstm_model.add(Dropout(0.5))
lstm_model.add(Dense(32, activation='relu'))
lstm_model.add(Dropout(0.5))
lstm_model.add(Dense(1, activation='sigmoid'))

lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

lstm_model.fit(X_train_tfidf, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)

print("Evaluating LSTM model...")
y_prob_lstm = lstm_model.predict(X_test_tfidf).flatten()
y_pred_lstm = (y_prob_lstm > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred_lstm)
precision = precision_score(y_test, y_pred_lstm)
recall = recall_score(y_test, y_pred_lstm)
f1 = f1_score(y_test, y_pred_lstm)
roc_auc = roc_auc_score(y_test, y_prob_lstm)
report = classification_report(y_test, y_pred_lstm)

print(f"LSTM Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print("Classification Report:")
print(report)

with open(results_path, 'a') as f:
    f.write(f"LSTM Performance:\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"ROC-AUC: {roc_auc:.4f}\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\n")

print("Saving LSTM model...")
lstm_model.save(lstm_model_path)

print("Sentiment analysis model training and evaluation completed!")
