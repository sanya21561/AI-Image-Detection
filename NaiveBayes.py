import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

train_data_path = '/Users/brindamuralie/Desktop/kaggle/input/train'
test_data_path = '/Users/brindamuralie/Desktop/kaggle/input/test'

# Create empty lists to store image data and labels
data = []
labels = []

# Loop through the REAL and FAKE subfolders in the train data directory
for folder in os.listdir(train_data_path):
    if folder != '.DS_Store':
        folder_path = os.path.join(train_data_path, folder)
        label = folder  # Label is either 'REAL' or 'FAKE'

        # Loop through the images in each subfolder
        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (32, 32))  # Resize to 32x32 pixels
            data.append(image)
            labels.append(label)

# Convert data and labels to NumPy arrays
data = np.array(data)
labels = np.array(labels)

# Encode the labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Split the data into training and validation sets (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

# Train a Multinomial Naive Bayes classifier
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_train.reshape(-1, 32 * 32 * 3), y_train)  # Reshape for NB classifier

# Predict on the validation data
y_pred = naive_bayes_classifier.predict(X_val.reshape(-1, 32 * 32 * 3))

# Calculate accuracy on validation data
accuracy = accuracy_score(y_val, y_pred)
print("train")
print(f"Validation Accuracy: {accuracy}")

# Load and preprocess test data
test_data = []
test_labels = []

for folder in os.listdir(test_data_path):
    folder_path = os.path.join(test_data_path, folder)
    label = folder  # Label is either 'REAL' or 'FAKE'
    if folder != '.DS_Store':
        # Loop through the images in each subfolder
        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (32, 32))  # Resize to 32x32 pixels
            test_data.append(image)
            test_labels.append(label)

# Convert test data and labels to NumPy arrays
test_data = np.array(test_data)
test_labels = label_encoder.transform(test_labels)  # Encode test labels

# Predict on the test data
test_pred = naive_bayes_classifier.predict(test_data.reshape(-1, 32 * 32 * 3))

# Calculate accuracy on the test data
test_accuracy = accuracy_score(test_labels, test_pred)
print("Test")
print(f"Test Accuracy: {test_accuracy}")
