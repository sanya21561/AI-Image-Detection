import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to compute and visualize image statistics
def visualize_image_stats(image, title):
    # Convert the image to grayscale for edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute edges using the Canny edge detector
    edges = cv2.Canny(gray, 100, 200)

    # Compute color histogram
    hist = cv2.calcHist([image], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])

    # Flatten the histogram for plotting
    hist = hist.flatten()

    # Plot the original image, edges, and color histogram
    plt.figure(figsize=(12, 4))

    plt.subplot(131)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")

    plt.subplot(132)
    plt.imshow(edges, cmap='gray')
    plt.title("Edges")

    plt.subplot(133)
    plt.bar(range(len(hist)), hist)
    plt.title("Color Histogram")

    plt.suptitle(title)
    plt.show()



# Load and visualize two sample images from each class
class1_image1 = cv2.imread('data/test/FAKE/0.jpg')
class1_image2 = cv2.imread('data/test/FAKE/1.jpg')
class2_image1 = cv2.imread('data/test/REAL/0000.jpg')
class2_image2 = cv2.imread('data/test/REAL/0001.jpg')

# Visualize images and their statistics
visualize_image_stats(class1_image1, "real Image 1")
visualize_image_stats(class1_image2, "real Image 2")
visualize_image_stats(class2_image1, "fake Image 1")
visualize_image_stats(class2_image2, "fake Image 2")
