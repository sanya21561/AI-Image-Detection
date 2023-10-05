import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def compute_color_histogram(image, bins=(8, 8, 8)):
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def compute_edge_histogram(image):
    edges = cv2.Canny(image,100,200)
    hist = np.histogram(edges,bins=[0,128,256])
    return hist[0]

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def plot_histograms(images, title):
    color_histograms = []
    edge_histograms = []
    for img in images:
        color_histograms.append(compute_color_histogram(img))
        edge_histograms.append(compute_edge_histogram(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.title(f'Color Histograms for {title}')
    for histogram in color_histograms:
        plt.hist(histogram,bins=range(257),color='blue',alpha=0.7)

    plt.subplot(1,2,2)
    plt.title(f'Edge Histograms for {title}')
    for histogram in edge_histograms:
        plt.hist(histogram,bins=range(257),color='red',alpha=0.7)

    plt.show()

# Load images
class1_images = load_images_from_folder('data/test/FAKE')
class2_images = load_images_from_folder('data/test/REAL')

# Plot histograms
plot_histograms(class1_images,'REAL')
plot_histograms(class2_images,'FAKE')
