import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2

# Specify the file paths
file_path_labels = "./labels.txt"
file_path_api_calls = "./all_analysis_data.txt"

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

feature_selector = SelectKBest(chi2, k=1000)
calls_encoded_selected = feature_selector.fit_transform(calls_encoded, all_labels)

X_train, X_test, y_train, y_test = train_test_split(calls_encoded_selected, all_labels, test_size=0.2, random_state=42)

# Define the main model algorithms
models = [
    ("Random Forest", RandomForestClassifier()),
    ("Multi-Layer Perceptron", MLPClassifier()),
    ("K-Nearest Neighbors", KNeighborsClassifier()),
    ("Support Vector Machine", SVC())
]

# Iterate over the models and perform benchmarking
for model_name, model in models:
    start_time = time.time()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)

    end_time = time.time()
    execution_time = end_time - start_time

    # Print the results
    print("Model: {}".format(model_name))
    print("Accuracy: {:.4f}".format(accuracy))
    print("Execution Time: {:.2f} seconds\n".format(execution_time))
