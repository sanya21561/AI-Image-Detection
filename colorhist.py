import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# ...

# Create empty lists to store image data and labels
data = []
labels = []
color_histograms = []

train_data_path = '/Users/brindamuralie/Desktop/kaggle/input/train'
test_data_path = '/Users/brindamuralie/Desktop/kaggle/input/test'

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

            histogram = cv2.calcHist([image], [0, 1, 2], None, [32, 32, 32], [0, 32, 0, 32, 0, 32])
            color_histograms.append(histogram)

print(len(data))
# Convert data and labels to NumPy arrays
data = np.array(data)
labels = np.array(labels)

# Create a LabelEncoder and encode the labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

real_color_histograms = [hist for i, hist in enumerate(color_histograms) if labels[i] == 0]
fake_color_histograms = [hist for i, hist in enumerate(color_histograms) if labels[i] == 1]

# Plot histograms for the 'REAL' class
plt.figure(figsize=(8, 5))
for channel, color in zip(range(3), ['blue', 'green', 'red']):
    channel_data = [hist[channel] for hist in real_color_histograms]
    channel_data = np.array(channel_data).flatten()  # Flatten the data
    plt.hist(channel_data, bins=32, range=[0, 32], color=color, alpha=0.7, label=f'{color.capitalize()} Channel')
plt.title('Color Histogram (REAL Class)')
plt.xlabel('Color Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Plot histograms for the 'FAKE' class
plt.figure(figsize=(8, 5))
for channel, color in zip(range(3), ['blue', 'green', 'red']):
    channel_data = [hist[channel] for hist in fake_color_histograms]
    channel_data = np.array(channel_data).flatten()  # Flatten the data
    plt.hist(channel_data, bins=32, range=[0, 32], color=color, alpha=0.7, label=f'{color.capitalize()} Channel')
plt.title('Color Histogram (FAKE Class)')
plt.xlabel('Color Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()
