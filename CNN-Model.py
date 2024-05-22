# cnn_model.py

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import os

# Define file paths
train_data_path = 'data/processed/train_data.csv'
test_data_path = 'data/processed/test_data.csv'
cnn_model_path = 'models/cnn_model.h5'
tokenizer_path = 'models/tokenizer.pkl'
results_path = 'results/cnn_report.txt'

# Create directories if they don't exist
os.makedirs(os.path.dirname(cnn_model_path), exist_ok=True)
os.makedirs(os.path.dirname(results_path), exist_ok=True)

# Load processed training and testing data
print("Loading processed data...")
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

# Tokenization and padding
max_features = 10000  # Number of words to consider as features
maxlen = 100  # Cut texts after this number of words

print("Tokenizing text data...")
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(train_data['text'].values)

X_train_seq = tokenizer.texts_to_sequences(train_data['text'].values)
X_test_seq = tokenizer.texts_to_sequences(test_data['text'].values)

X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen)
X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen)

y_train = train_data['sentiment']
y_test = test_data['sentiment']

# Build CNN model
print("Building CNN model...")
model = Sequential()
model.add(Embedding(input_dim=max_features, output_dim=128, input_length=maxlen))
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
print("Training the model...")
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(X_train_pad, y_train, epochs=10, batch_size=64, validation_split=0.2, callbacks=[early_stopping], verbose=1)

# Predict on the test set
print("Predicting on the test set...")
y_prob = model.predict(X_test_pad).flatten()
y_pred = (y_prob > 0.5).astype(int)

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

# Save the model and tokenizer
print("Saving the model and tokenizer...")
model.save(cnn_model_path)
joblib.dump(tokenizer, tokenizer_path)

print("CNN model training and evaluation completed!")
