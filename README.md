# AI-Image-Detection

****Motivation****


- Differentiating between real and AI-generated images has become a complex task for humans due to the remarkable progress in AI image generation.
- Misleading images that convincingly imitate reality can cause significant harm like spreading fake news, damaging reputations, and manipulating public opinion through media.

**Dataset Description**
- Dataset includes 120,000 images, evenly split between real and synthetic (fake) images, categorized into ten distinct classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck, with 6,000 images per class)
- The dataset is divided into 100,000 training images (50,000 each for real and fake images) and 20,000 testing images (10,000 each for real and fake images), all in RGB format and resized to 32x32 pixels.
- The real images used in the dataset were extracted from the CIFAR-10 dataset introduced by Krizhevsky and Hinton. The synthetic images, on the other hand, were generated using Hugging Face's Stable Diffusion Model Version 1.4. 

**Dataset Visualisation**

- TSNE plot (Figure 1) unsuitable due to scattered real and fake images.  
- Pixel intensity histogram (Figure 2) does not display much differences between real and fake images.
<img width="883" alt="image" src="https://github.com/sanya21561/AI-Image-Detection/assets/108250053/a0f70b8c-1fab-45be-81b6-93c3f1549b1e">
 
- The background defects (Figure 4) highlights variations in background quality, with real images having fewer defects and fake images showing a higher frequency, suggesting potential generation artifacts. These features aid in distinguishing real and fake images based on blur and background characteristics.
<img width="624" alt="image" src="https://github.com/sanya21561/AI-Image-Detection/assets/108250053/40e2a50e-b9b1-4696-938f-dbb207b6846c">

- The blur level histogram (Figure 3) reveals that real images generally have lower blur levels, preserving finer details, while fake images concentrate at higher blur levels, suggesting potential differences in quality or generation methods.
<img width="715" alt="image" src="https://github.com/sanya21561/AI-Image-Detection/assets/108250053/9201b0aa-2803-4aa8-86df-041da2cb3bc7">



**Dataset Preprocessing**
- Images loaded using CV2's 'imread' function and converted to numeric data with Numpy.
- Uniform 32x32 pixel dimensions ensured using OpenCV.
- Class labels transformed to numeric values using scikit-learn's LabelEncoder.
- No outliers found via data visualization and box plot (Figure 6).
- PCA deemed unnecessary as the dataset already had optimal dimensionality which was verified by experiments.


**Methodology**

- Employed Logistic Regression, Naïve Bayes classifier, Decision tree Classifier, Random Forest Classifier, Support Vector Machine, Multilayer Perceptron and Convolution Neural Network.
- We have used 7 models to train the dataset and evaluated accuracy scores for each model to compare them. 
- Used Scikit learn, matplotlib, TensorFlow, Numpy and pandas library to implement this.
- Also plotted Precision recall curve and calculated various values like AUC, test accuracy, validation accuracy, cross entropy loss to study each model.
- We performed hyperparameter tuning to achieve the best accuracy in every model.
- We also tried ensemble methods, with multiple models including a combination of Logistic Regression and SVM, KNN and CNN.
- We found out the best accuracy method being CNN alone.

<img width="1072" alt="image" src="https://github.com/sanya21561/AI-Image-Detection/assets/108250053/a85372ad-50d6-45e6-bd8c-f084a65b126f">   
<img width="1072" alt="image" src="https://github.com/sanya21561/AI-Image-Detection/assets/108250053/ccce8efd-7a31-4c1b-ba5d-ddedb6a001d8">   
<img width="1072" alt="image" src="https://github.com/sanya21561/AI-Image-Detection/assets/108250053/1d428a55-b3c7-44a2-bb1e-5487a30f304f">  
<img width="802" alt="image" src="https://github.com/sanya21561/AI-Image-Detection/assets/108250053/7c436d49-b624-499c-9147-458b686e0e21"> 
<img width="598" alt="image" src="https://github.com/sanya21561/AI-Image-Detection/assets/108250053/96981a97-503f-433d-b25e-752559562bd9">  


**Results and Analysis**

- We evaluated seven machine learning models, and found varying levels of performance.
- CNN achieved the highest test accuracy, reaching 92.5%, making it the top-performing model in our study. It is due to its ability to automatically learn hierarchical features and spatial hierarchies, enabling it to capture intricate patterns and relationships within images.
- Naïve Bayes exhibited the poorest performance with a test accuracy of 59.275%, due to its simplistic assumption of feature independence.
- Random Forests, MLP and SVM also showed good accuracies ranging from 80% to 90%.
  <img width="701" alt="image" src="https://github.com/sanya21561/AI-Image-Detection/assets/108250053/e92f7edf-1e83-4b13-85d1-50b8a07c080d">

**Conclusions**

- In conclusion, this project addresses the critical challenge of distinguishing real images from AI-generated ones, which have the potential to spread misinformation and manipulate public opinion. 
- By utilizing the CIFAKE2 dataset and employing various machine learning models, we have achieved promising results.
- The project has various limitations like hardware requirements for larger image datasets, and lesser research papers due to novelty of topic. 










