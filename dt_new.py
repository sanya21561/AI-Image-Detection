import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
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
            image = cv2.resize(image, (32, 32))  # we are given 32*32 images but created check for consistency
            data.append(image)
            labels.append(label)



data = np.array(data)
labels = np.array(labels)

# Encode the labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Split the data into training and validation sets (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)




# Create a Decision Tree classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# Train the classifier on the training data
dt_classifier.fit(X_train.reshape(-1, 32*32*3), y_train)
print("Model Trained")

# Make predictions on the validation data
val_predictions = dt_classifier.predict(X_val.reshape(-1, 32*32*3))

# Calculate accuracy on the validation set
validation_accuracy = accuracy_score(y_val, val_predictions)
print("Validation Accuracy:", validation_accuracy)

# Calculate precision, recall, and F1-Score on the validation data
precision = precision_score(y_val, val_predictions)
recall = recall_score(y_val, val_predictions)
f1 = f1_score(y_val, val_predictions)
print("Validation Precision:", precision)
print("Validation Recall:", recall)
print("Validation F1-Score:", f1)

# Calculate the confusion matrix on the validation data
conf_matrix = confusion_matrix(y_val, val_predictions)
print("Confusion Matrix:")
print(conf_matrix)

# Calculate specificity and false positive rate on the validation data
tn, fp, fn, tp = conf_matrix.ravel()
specificity = tn / (tn + fp)
fpr = fp / (tn + fp)
print("Validation Specificity:", specificity)
print("Validation False Positive Rate:", fpr)

# Calculate ROC curve and AUC-ROC on the validation data
y_prob = dt_classifier.predict_proba(X_val.reshape(-1, 32*32*3))
fpr, tpr, thresholds = roc_curve(y_val, y_prob[:, 1])
roc_auc = auc(fpr, tpr)
print("Validation ROC Curve and AUC-ROC:")
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

# Calculate Precision-Recall curve and AUC-PR on the validation data
precision, recall, _ = precision_recall_curve(y_val, y_prob[:, 1])
pr_auc = average_precision_score(y_val, y_prob[:, 1])
print("Validation Precision-Recall Curve and AUC-PR:")
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label=f'AUC = {pr_auc:.2f}')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.show()




test_data = []
test_labels = []

for folder in os.listdir(test_data_path):
    if folder != '.DS_Store':
        folder_path = os.path.join(test_data_path, folder)
        label = folder  # Label is either 'REAL' or 'FAKE'

        # Loop through the images in each subfolder
        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (32, 32))  # we are given 32*32 images but created check for consistency
            test_data.append(image)
            test_labels.append(label)



test_data = np.array(test_data)

# Encode the test labels using the same label_encoder
test_labels_encoded = label_encoder.transform(test_labels)

# Make predictions on the test data
test_predictions = dt_classifier.predict(test_data.reshape(-1, 32*32*3))

# Calculate accuracy on the test data
test_accuracy = accuracy_score(test_labels_encoded, test_predictions)
print("Test Accuracy:", test_accuracy)

# Calculate precision, recall, and F1-Score on the test data
test_precision = precision_score(test_labels_encoded, test_predictions, pos_label=label_encoder.transform(['REAL'])[0])
test_recall = recall_score(test_labels_encoded, test_predictions, pos_label=label_encoder.transform(['REAL'])[0])
test_f1 = f1_score(test_labels_encoded, test_predictions, pos_label=label_encoder.transform(['REAL'])[0])
print("Test Precision:", test_precision)
print("Test Recall:", test_recall)
print("Test F1-Score:", test_f1)

# Calculate the confusion matrix on the test data
test_conf_matrix = confusion_matrix(test_labels_encoded, test_predictions)
print("Test Confusion Matrix:")
print(test_conf_matrix)

# Calculate specificity and false positive rate on the test data
test_tn, test_fp, test_fn, test_tp = test_conf_matrix.ravel()
test_specificity = test_tn / (test_tn + test_fp)
test_fpr = test_fp / (test_tn + test_fp)
print("Test Specificity:", test_specificity)
print("Test False Positive Rate:", test_fpr)



# Calculate ROC curve and AUC-ROC on the test data
test_y_prob = dt_classifier.predict_proba(test_data.reshape(-1, 32*32*3))
test_fpr, test_tpr, test_thresholds = roc_curve(test_labels_encoded, test_y_prob[:, 1])


test_roc_auc = auc(test_fpr, test_tpr)
print("Test ROC Curve and AUC-ROC:")
plt.figure(figsize=(8, 6))
plt.plot(test_fpr, test_tpr, color='darkorange', lw=2, label=f'AUC = {test_roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (Test)')
plt.legend(loc='lower right')
plt.show()

# Calculate Precision-Recall curve and AUC-PR on the test data
test_precision, test_recall, _ = precision_recall_curve(test_labels_encoded, test_y_prob[:, 1], pos_label=1)

test_pr_auc = average_precision_score(test_labels, test_y_prob[:, 1])
print("Test Precision-Recall Curve and AUC-PR:")
plt.figure(figsize=(8, 6))
plt.plot(test_recall, test_precision, color='blue', lw=2, label=f'AUC = {test_pr_auc:.2f}')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Test)')
plt.legend(loc='lower left')
plt.show()
