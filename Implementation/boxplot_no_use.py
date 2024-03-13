# import os
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import LabelEncoder


# # ...

# # Create empty lists to store image data and labels
# data = []
# labels = []

# train_data_path = '/Users/brindamuralie/Desktop/kaggle/input/train'
# test_data_path = '/Users/brindamuralie/Desktop/kaggle/input/test'

# # Loop through the REAL and FAKE subfolders in the train data directory
# for folder in os.listdir(train_data_path):
#     if folder != '.DS_Store':
#         folder_path = os.path.join(train_data_path, folder)
#         label = folder  # Label is either 'REAL' or 'FAKE'

#         # Loop through the images in each subfolder
#         for filename in os.listdir(folder_path):
#             image_path = os.path.join(folder_path, filename)
#             image = cv2.imread(image_path)
#             image = cv2.resize(image, (32, 32))  # Resize to 32x32 pixels
#             data.append(image)
#             labels.append(label)

# # Convert data and labels to NumPy arrays
# data = np.array(data)
# labels = np.array(labels)


# label_encoder = LabelEncoder()
# labels = label_encoder.fit_transform(labels)

# # ...

# # Create box plots to visualize pixel value distributions for 'REAL' and 'FAKE' classes
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.boxplot([data[labels == 'REAL'].ravel(), data[labels == 'FAKE'].ravel()])
# plt.xticks([1, 2], ['REAL', 'FAKE'])
# plt.title('Pixel Value Distribution (Training Data)')

# # Load and preprocess test data
# test_data = []
# test_labels = []

# for folder in os.listdir(test_data_path):
#     folder_path = os.path.join(test_data_path, folder)
#     label = folder  # Label is either 'REAL' or 'FAKE'
#     if folder != '.DS_Store':
#         # Loop through the images in each subfolder
#         for filename in os.listdir(folder_path):
#             image_path = os.path.join(folder_path, filename)
#             image = cv2.imread(image_path)
#             image = cv2.resize(image, (32, 32))  # Resize to 32x32 pixels
#             test_data.append(image)
#             test_labels.append(label)

# # Convert test data and labels to NumPy arrays
# test_data = np.array(test_data)
# test_labels = label_encoder.transform(test_labels)  # Encode test labels

# # ...

# # Create box plots to visualize pixel value distributions for 'REAL' and 'FAKE' classes (Test Data)
# plt.subplot(1, 2, 2)
# plt.boxplot([test_data[test_labels == 0].ravel(), test_data[test_labels == 1].ravel()])
# plt.xticks([1, 2], ['REAL', 'FAKE'])
# plt.title('Pixel Value Distribution (Test Data)')

# plt.tight_layout()
# plt.show()



import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# ...

# Create empty lists to store image data and labels
data = []
labels = []

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

# Convert data and labels to NumPy arrays
data = np.array(data)
labels = np.array(labels)

# Create a LabelEncoder and encode the labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# ...

# Create a box plot to visualize pixel value distributions for 'REAL' and 'FAKE' classes (Training Data)
plt.figure(figsize=(8, 5))
plt.boxplot([data[labels == 0].ravel(), data[labels == 1].ravel()])
plt.xticks([1, 2], ['REAL', 'FAKE'])
plt.title('Pixel Value Distribution (Training Data)')
plt.ylabel('Pixel Value')

plt.tight_layout()
plt.show()
