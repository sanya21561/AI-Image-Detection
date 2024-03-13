import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def display_comparison_images(category1, category2, dataset_path, num_images=5):
    category1_path = os.path.join(dataset_path, category1)
    category2_path = os.path.join(dataset_path, category2)

    image_files_category1 = os.listdir(category1_path)
    image_files_category2 = os.listdir(category2_path)

    selected_images_category1 = random.sample(image_files_category1, num_images)
    selected_images_category2 = random.sample(image_files_category2, num_images)

    plt.figure(figsize=(15, 6))

    for i in range(num_images):
        # Display image from category 1
        plt.subplot(2, num_images, i + 1)
        image_path_category1 = os.path.join(category1_path, selected_images_category1[i])
        image_category1 = Image.open(image_path_category1)
        plt.imshow(image_category1)
        plt.title(f"{category1} - {selected_images_category1[i]}")
        plt.axis('off')

        # Display image from category 2
        plt.subplot(2, num_images, num_images + i + 1)
        image_path_category2 = os.path.join(category2_path, selected_images_category2[i])
        image_category2 = Image.open(image_path_category2)
        plt.imshow(image_category2)
        plt.title(f"{category2} - {selected_images_category2[i]}")
        plt.axis('off')

    plt.show()

def plot_pixel_intensity_histogram(category1, category2, dataset_path):
    category1_path = os.path.join(dataset_path, category1)
    category2_path = os.path.join(dataset_path, category2)

    pixel_values_category1 = []
    pixel_values_category2 = []

    for image_file in os.listdir(category1_path):
        image_path = os.path.join(category1_path, image_file)
        image = np.array(Image.open(image_path))
        pixel_values_category1.extend(image.flatten())

    for image_file in os.listdir(category2_path):
        image_path = os.path.join(category2_path, image_file)
        image = np.array(Image.open(image_path))
        pixel_values_category2.extend(image.flatten())

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(pixel_values_category1, bins=50, color='blue', alpha=0.7)
    plt.title(f'Pixel Intensity Histogram - {category1}')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(pixel_values_category2, bins=50, color='orange', alpha=0.7)
    plt.title(f'Pixel Intensity Histogram - {category2}')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')

    plt.show()

# Set your dataset path
dataset_path = 'data/train'

# Display comparison images
# display_comparison_images('FAKE', 'REAL', dataset_path)
# Plot pixel intensity histograms
# plot_pixel_intensity_histogram('FAKE', 'REAL', dataset_path)

def display_image_grid(category1, category2, dataset_path, num_rows=2, num_cols=5):
    category1_path = os.path.join(dataset_path, category1)
    category2_path = os.path.join(dataset_path, category2)

    selected_images_category1 = random.sample(os.listdir(category1_path), num_rows * num_cols)
    selected_images_category2 = random.sample(os.listdir(category2_path), num_rows * num_cols)

    plt.figure(figsize=(15, 6))

    for i in range(num_rows * num_cols):
        # Display image from category 1
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        image_path_category1 = os.path.join(category1_path, selected_images_category1[i])
        image_category1 = Image.open(image_path_category1)
        plt.imshow(image_category1)
        plt.title(f"{category1} - {selected_images_category1[i]}")
        plt.axis('off')

        # Display image from category 2
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        image_path_category2 = os.path.join(category2_path, selected_images_category2[i])
        image_category2 = Image.open(image_path_category2)
        plt.imshow(image_category2)
        plt.title(f"{category2} - {selected_images_category2[i]}")
        plt.axis('off')

    plt.show()

# display_image_grid('FAKE', 'REAL', dataset_path)

import cv2

def calculate_blur(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calculate the variance of Laplacian
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def plot_blur_levels(category1, category2, dataset_path):
    category1_path = os.path.join(dataset_path, category1)
    category2_path = os.path.join(dataset_path, category2)

    blur_levels_category1 = []
    blur_levels_category2 = []

    for image_file in os.listdir(category1_path):
        image_path = os.path.join(category1_path, image_file)
        image = cv2.imread(image_path)
        blur_levels_category1.append(calculate_blur(image))

    for image_file in os.listdir(category2_path):
        image_path = os.path.join(category2_path, image_file)
        image = cv2.imread(image_path)
        blur_levels_category2.append(calculate_blur(image))

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(blur_levels_category1, bins=50, color='blue', alpha=0.7)
    plt.title(f'Blur Level Histogram - {category1}')
    plt.xlabel('Blur Level')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(blur_levels_category2, bins=50, color='orange', alpha=0.7)
    plt.title(f'Blur Level Histogram - {category2}')
    plt.xlabel('Blur Level')
    plt.ylabel('Frequency')

    plt.show()

# Plot blur level histograms
# plot_blur_levels('FAKE', 'REAL', dataset_path)

def plot_background_defects(category1, category2, dataset_path):
    category1_path = os.path.join(dataset_path, category1)
    category2_path = os.path.join(dataset_path, category2)

    # Example: Use edge detection (Canny) to identify background defects
    def detect_edges(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return edges

    plt.figure(figsize=(15, 6))

    for i, (category_path, label) in enumerate([(category1_path, category1), (category2_path, category2)], 1):
        plt.subplot(1, 2, i)
        for image_file in os.listdir(category_path):
            image_path = os.path.join(category_path, image_file)
            image = cv2.imread(image_path)
            edges = detect_edges(image)
            plt.imshow(edges, cmap='gray')
            plt.title(f'Background Defects - {label}')
            plt.axis('off')
            break  # Display only one example for brevity

    plt.show()

# Plot background defects using edge detection
plot_background_defects('FAKE', 'REAL', dataset_path)
import cv2

def calculate_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def count_images_above_threshold(category_path, threshold):
    count = 0
    for image_file in os.listdir(category_path):
        image_path = os.path.join(category_path, image_file)
        image = cv2.imread(image_path)
        blur_level = calculate_blur(image)
        if blur_level > threshold:
            count += 1
    return count

def count_images_with_background_defects(category, dataset_path, threshold):
    category_path = os.path.join(dataset_path, category)
    count = count_images_above_threshold(category_path, threshold)
    return count

dataset_path = 'data/train'

# Set the blur level threshold (adjust as needed)
blur_threshold = 5000

# Count images with background defects in both categories
count_fake = count_images_with_background_defects('FAKE', dataset_path, blur_threshold)
count_real = count_images_with_background_defects('REAL', dataset_path, blur_threshold)

print(f"Number of images with background defects in FAKE category: {count_fake}")
print(f"Number of images with background defects in REAL category: {count_real}")

from skimage import exposure
from skimage import filters
from skimage.morphology import disk
from skimage import segmentation
from skimage import color

def preprocess_image(image):
    # Apply Gaussian blur for noise suppression
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply adaptive histogram equalization for illumination correction
    equalized = exposure.equalize_adapthist(blurred, clip_limit=0.03)

    return equalized

def adaptive_defect_segmentation(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply thresholding using Otsu's method
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply morphological operations for noise reduction
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel=np.ones((5, 5), np.uint8))

    # Use SLIC (Simple Linear Iterative Clustering) for adaptive segmentation
    segments = segmentation.slic(image, n_segments=100, compactness=10, sigma=1)

    # Mask the original image with the binary defect mask
    masked_image = color.label2rgb(segments, image, colors=[(1, 0, 0)])

    return masked_image

def plot_preprocessed_and_segmented_images(category1, category2, dataset_path):
    category1_path = os.path.join(dataset_path, category1)
    category2_path = os.path.join(dataset_path, category2)

    plt.figure(figsize=(15, 6))

    for i, (category_path, label) in enumerate([(category1_path, category1), (category2_path, category2)], 1):
        plt.subplot(2, 3, 3 * i - 2)
        for image_file in os.listdir(category_path):
            image_path = os.path.join(category_path, image_file)
            image = cv2.imread(image_path)

            # Display the original image
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title(f'Original - {label}')
            plt.axis('off')
            break  

        plt.subplot(2, 3, 3 * i - 1)
        for image_file in os.listdir(category_path):
            image_path = os.path.join(category_path, image_file)
            image = cv2.imread(image_path)

            preprocessed_image = preprocess_image(image)

            plt.imshow(preprocessed_image, cmap='gray')
            plt.title(f'Preprocessed - {label}')
            plt.axis('off')
            break  

        plt.subplot(2, 3, 3 * i)
        for image_file in os.listdir(category_path):
            image_path = os.path.join(category_path, image_file)
            image = cv2.imread(image_path)

            # Apply adaptive defect segmentation
            segmented_image = adaptive_defect_segmentation(image)

            # Display the segmented image
            plt.imshow(segmented_image)
            plt.title(f'Segmented - {label}')
            plt.axis('off')
            break  # Display only one example for brevity

    plt.show()



dataset_path = 'data/train'

# Display preprocessed and segmented images
plot_preprocessed_and_segmented_images('FAKE', 'REAL', dataset_path)
