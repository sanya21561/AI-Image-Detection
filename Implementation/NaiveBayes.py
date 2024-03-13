import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

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
# Calculate accuracy on validation data
accuracy = accuracy_score(y_val, y_pred)
print("Validation")
print(f"Accuracy: {accuracy}")

# Calculate precision, recall, and F1-Score on validation data
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")

# Calculate the confusion matrix on validation data
conf_matrix = confusion_matrix(y_val, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Calculate specificity and false positive rate on validation data
tn, fp, fn, tp = conf_matrix.ravel()
specificity = tn / (tn + fp)
fpr = fp / (tn + fp)
print(f"Specificity: {specificity}")
print(f"False Positive Rate: {fpr}")

# Calculate ROC curve and AUC-ROC on validation data
y_prob = naive_bayes_classifier.predict_proba(X_val.reshape(-1, 32 * 32 * 3))
fpr, tpr, thresholds = roc_curve(y_val, y_prob[:, 1])
roc_auc = auc(fpr, tpr)
print("ROC Curve and AUC-ROC:")
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Calculate Precision-Recall curve and AUC-PR on validation data
precision, recall, _ = precision_recall_curve(y_val, y_prob[:, 1])
pr_auc = average_precision_score(y_val, y_prob[:, 1])
print("Precision-Recall Curve and AUC-PR:")
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label=f'AUC = {pr_auc:.2f}')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.show()




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



# Calculate precision, recall, and F1-Score on test data
precision = precision_score(test_labels, test_pred)
recall = recall_score(test_labels, test_pred)
f1 = f1_score(test_labels, test_pred)
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")

# Calculate the confusion matrix on validation data
conf_matrix = confusion_matrix(test_labels, test_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Calculate specificity and false positive rate on validation data
tn, fp, fn, tp = conf_matrix.ravel()
specificity = tn / (tn + fp)
fpr = fp / (tn + fp)
print(f"Specificity: {specificity}")
print(f"False Positive Rate: {fpr}")

# Calculate ROC curve and AUC-ROC on validation data
test_prob = naive_bayes_classifier.predict_proba(X_val.reshape(-1, 32 * 32 * 3))
fpr, tpr, thresholds = roc_curve(test_labels, test_prob[:, 1])
roc_auc = auc(fpr, tpr)
print("ROC Curve and AUC-ROC:")
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Calculate Precision-Recall curve and AUC-PR on validation data
precision, recall, _ = precision_recall_curve(test_labels, test_prob[:, 1])
pr_auc = average_precision_score(test_labels, test_prob[:, 1])
print("Precision-Recall Curve and AUC-PR:")
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label=f'AUC = {pr_auc:.2f}')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.show()
