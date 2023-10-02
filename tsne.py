import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

real = "data/test/REAL"
fake = "data/test/FAKE"
def load_and_preprocess_images(directory):
    images = []
    for filename in os.listdir(directory):
        image_path = os.path.join(directory, filename)
        img = cv2.imread(image_path)
        img = cv2.resize(img, (32, 32)) #images are already 32*32 but still paranoia
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #rgb hi toh tha 
        img = img / 255.0
        images.append(img)
    return images

real_images = load_and_preprocess_images(real)

fake_images = load_and_preprocess_images(fake)

dataset = real_images + fake_images
labels = [0] * len(real_images) + [1] * len(fake_images)

dataset = np.array(dataset)
labels = np.array(labels)

dataset_flattened = dataset.reshape(len(dataset), -1)

tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(dataset_flattened)

plt.figure(figsize=(10, 8))
plt.scatter(tsne_result[labels == 0, 0], tsne_result[labels == 0, 1], label='Real Images', alpha=0.7)
plt.scatter(tsne_result[labels == 1, 0], tsne_result[labels == 1, 1], label='Fake Images', alpha=0.7)
plt.title('t-SNE Visualization')
plt.legend()
plt.show()

