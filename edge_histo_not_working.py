import os
import cv2
import numpy as np
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
            if label == "REAL":
                labels.append(0)
            else:
                labels.append(1)

# Convert data and labels to NumPy arrays
data = np.array(data)
labels = np.array(labels)

# Encode the labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Function to calculate edge histogram for an image
def calculate_edge_histogram(image, num_bins=256, range_min=0, range_max=256, threshold1=30, threshold2=100):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    # edges = cv2.Canny(gray_image, threshold1=threshold1, threshold2=threshold2)
    edges = cv2.Canny(gray_image, threshold1=threshold1, threshold2=threshold2)

    # Calculate the histogram of edges
    hist, _ = np.histogram(edges, bins=num_bins, range=(range_min, range_max))
    
    return hist

# Calculate edge histograms for the images in the dataset
# Calculate edge histograms for the images in the dataset
# edge_histograms = [calculate_edge_histogram(image) for image in data]
# Calculate edge histograms for the images in the dataset with custom parameters
edge_histograms = [calculate_edge_histogram(image, num_bins=256, range_min=0, range_max=256, threshold1=30, threshold2=100) for image in data]


# Normalize the histograms
normalized_edge_histograms = [hist / np.sum(hist) if np.sum(hist) != 0 else hist for hist in edge_histograms]

# Plot the histograms for 'REAL' and 'FAKE' classes with automatic colors
plt.figure(figsize=(8, 5))
plt.hist(np.array(normalized_edge_histograms)[np.where(labels == 0)], bins=256, range=(0, 1), alpha=0.5, label='REAL', density=True)
plt.hist(np.array(normalized_edge_histograms)[np.where(labels == 1)], bins=256, range=(0, 1), alpha=0.5, label='FAKE', density=True)
plt.title('Edge Histogram (Training Data)')
plt.xlabel('Edge Strength (Normalized)')
plt.ylabel('Frequency')
plt.legend(loc='upper right')

plt.show()
