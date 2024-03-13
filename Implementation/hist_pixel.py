import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Define the paths to your training and test data
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

# Create histograms to visualize pixel value distributions for 'REAL' and 'FAKE' classes (Training Data)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(data[labels == 0].ravel(), bins=256, color='blue', alpha=0.7, label='REAL', density=True)
plt.hist(data[labels == 1].ravel(), bins=256, color='red', alpha=0.7, label='FAKE', density=True)
plt.xlim(0, 255)
plt.xlabel('Pixel Value')
plt.ylabel('Normalized Frequency')
plt.title('Pixel Value Distribution (REAL vs. FAKE - Training Data)')
plt.legend()

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
# test_data = np.array(test_data)
# test_labels = label_encoder.transform(test_labels)  # Encode test labels

# # Create histograms to visualize pixel value distributions for 'REAL' and 'FAKE' classes (Test Data)
# plt.subplot(1, 2, 2)
# plt.hist(test_data[test_labels == 0].ravel(), bins=256, color='blue', alpha=0.7, label='REAL', density=True)
# plt.hist(test_data[test_labels == 1].ravel(), bins=256, color='red', alpha=0.7, label='FAKE', density=True)
# plt.xlim(0, 255)
# plt.xlabel('Pixel Value')
# plt.ylabel('Normalized Frequency')
# plt.title('Pixel Value Distribution (REAL vs. FAKE - Test Data)')
# plt.legend()

plt.tight_layout()
plt.show()
