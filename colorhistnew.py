import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# ...

# Create empty lists to store image data and labels
data = []
labels = []
color_moments = []

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

            # Calculate color moments for each channel (BGR)
            b, g, r = cv2.split(image)
            moments = {
                'mean_b': np.mean(b),
                'mean_g': np.mean(g),
                'mean_r': np.mean(r),
                'std_dev_b': np.std(b),
                'std_dev_g': np.std(g),
                'std_dev_r': np.std(r),
                'skewness_b': np.mean((b - np.mean(b)) ** 3) / (np.std(b) ** 3),
                'skewness_g': np.mean((g - np.mean(g)) ** 3) / (np.std(g) ** 3),
                'skewness_r': np.mean((r - np.mean(r)) ** 3) / (np.std(r) ** 3),
                'kurtosis_b': np.mean((b - np.mean(b)) ** 4) / (np.std(b) ** 4),
                'kurtosis_g': np.mean((g - np.mean(g)) ** 4) / (np.std(g) ** 4),
                'kurtosis_r': np.mean((r - np.mean(r)) ** 4) / (np.std(r) ** 4)
            }
            color_moments.append(moments)

print(len(data))
# Convert data and labels to NumPy arrays
data = np.array(data)
labels = np.array(labels)

# Create a LabelEncoder and encode the labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

real_color_moments = [moments for i, moments in enumerate(color_moments) if labels[i] == 0]
fake_color_moments = [moments for i, moments in enumerate(color_moments) if labels[i] == 1]

# Plot histograms for the 'REAL' class
plt.figure(figsize=(8, 5))
for feature in moments.keys():
    real_data = [moment[feature] for moment in real_color_moments]
    plt.hist(real_data, bins=32, alpha=0.7, label=feature)
plt.title('Color Moments (REAL Class)')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Plot histograms for the 'FAKE' class
plt.figure(figsize=(8, 5))
for feature in moments.keys():
    fake_data = [moment[feature] for moment in fake_color_moments]
    plt.hist(fake_data, bins=32, alpha=0.7, label=feature)
plt.title('Color Moments (FAKE Class)')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()



# import os
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import LabelEncoder

# # ...

# # Create empty lists to store image data and labels
# data = []
# labels = []
# color_moments = []

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

#             # Calculate color moments for each channel (BGR)
#             b, g, r = cv2.split(image)
#             moments = {
#                 'mean_b': np.mean(b),
#                 'mean_g': np.mean(g),
#                 'mean_r': np.mean(r),
#                 'std_dev_b': np.std(b),
#                 'std_dev_g': np.std(g),
#                 'std_dev_r': np.std(r),
#                 'skewness_b': np.mean((b - np.mean(b)) ** 3) / (np.std(b) ** 3),
#                 'skewness_g': np.mean((g - np.mean(g)) ** 3) / (np.std(g) ** 3),
#                 'skewness_r': np.mean((r - np.mean(r)) ** 3) / (np.std(r) ** 3),
#                 'kurtosis_b': np.mean((b - np.mean(b)) ** 4) / (np.std(b) ** 4),
#                 'kurtosis_g': np.mean((g - np.mean(g)) ** 4) / (np.std(g) ** 4),
#                 'kurtosis_r': np.mean((r - np.mean(r)) ** 4) / (np.std(r) ** 4)
#             }
#             color_moments.append((moments, label))

# print(len(data))
# # Convert data and labels to NumPy arrays
# data = np.array(data)

# # Create a LabelEncoder and encode the labels
# label_encoder = LabelEncoder()
# encoded_labels = label_encoder.fit_transform(labels)

# # Separate color moments for REAL and FAKE classes
# real_color_moments = [moments for moments, label in color_moments if label == 'REAL']
# fake_color_moments = [moments for moments, label in color_moments if label == 'FAKE']

# # Color mapping for plotting
# color_mapping = {'mean_b': 'blue', 'mean_g': 'green', 'mean_r': 'red',
#                  'std_dev_b': 'cyan', 'std_dev_g': 'magenta', 'std_dev_r': 'yellow',
#                  'skewness_b': 'purple', 'skewness_g': 'orange', 'skewness_r': 'pink',
#                  'kurtosis_b': 'brown', 'kurtosis_g': 'gray', 'kurtosis_r': 'olive'}

# # Plot color moments for both classes
# for moment_name, color in color_mapping.items():
#     real_data = [moment[moment_name] for moment in real_color_moments]
#     fake_data = [moment[moment_name] for moment in fake_color_moments]

#     plt.figure(figsize=(8, 5))
#     plt.hist(real_data, bins=32, alpha=0.5, color=color, label='REAL', density=True)
#     plt.hist(fake_data, bins=32, alpha=0.5, color='black', label='FAKE', density=True)
#     plt.title(f'Color Moment: {moment_name}')
#     plt.xlabel('Value')
#     plt.ylabel('Normalized Frequency')
#     plt.legend()
#     plt.show()
