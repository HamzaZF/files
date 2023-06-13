import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout

# Specify the file paths
file_path_labels = "/content/drive/MyDrive/TP_API/labels.txt"
file_path_api_calls = "/content/drive/MyDrive/TP_API/all_analysis_data.txt"

# Read the text files using pandas
labels = pd.read_csv(file_path_labels, delimiter='\t', header=None, skip_blank_lines=True)
calls = pd.read_csv(file_path_api_calls, delimiter='\t', header=None, skip_blank_lines=True)

calls_np = calls.values
labels_np = labels.values

types = np.array(["Spyware", "Downloader", "Trojan", "Worms", "Adware", "Dropper", "Virus", "Backdoor"])

counter = 0
groups = [[] for _ in range(8)]

for call in calls_np:
    index = np.where(types == labels_np[counter])[0][0]
    groups[index].append(call)
    counter += 1

min_size = min(len(group) for group in groups)

all_calls = []
all_labels = []

for i, group in enumerate(groups):
    calls_splitted = []
    for e in group:
        calls_splitted.append(str(e).split())
    all_calls.extend(calls_splitted[:min_size])
    all_labels.extend([i] * min_size)

calls_concatenated = [' '.join(call) for call in all_calls]

vectorizer = CountVectorizer()
calls_encoded = vectorizer.fit_transform(calls_concatenated)

print("Number of features before feature selection:", calls_encoded.shape[1])

feature_selector = SelectKBest(chi2, k='all')
calls_encoded_selected = feature_selector.fit_transform(calls_encoded, all_labels)

print("Number of features after feature selection:", calls_encoded_selected.shape[1])

X_train, X_test, y_train, y_test = train_test_split(calls_encoded_selected, all_labels, test_size=0.2, random_state=42)

# Convert X_train to a dense numpy array
X_train = X_train.toarray()

# Convert y_train to a numpy array
y_train = np.array(y_train)

#vocab_size = len(vectorizer.vocabulary_) + 1
vocab_size = np.max(X_train) + 1
embedding_dim = 100
max_sequence_length = calls_encoded_selected.shape[1]

# Define the deep learning model (RNN)
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length))
model.add(SimpleRNN(units=128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Training the deep learning model (RNN)...")
start_time = time.time()

model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test), verbose=1)

end_time = time.time()
execution_time = end_time - start_time

print("\nDeep learning model (RNN) trained.")
print("Execution Time: {:.2f} seconds\n".format(execution_time))

# Make predictions on the test set
y_pred = model.predict_classes(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)

# Print the results
print("Deep Learning Model (RNN) Results:")
print("Accuracy: {:.4f}".format(accuracy))