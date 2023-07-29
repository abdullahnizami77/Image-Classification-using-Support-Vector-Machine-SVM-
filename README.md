

# Image Classification using Support Vector Machine (SVM)

This repository contains Python code for image classification using the Support Vector Machine (SVM) algorithm. The project focuses on differentiating between "Infected" and "Healthy" images.

## Dataset

The dataset consists of two classes: "Infected" and "Healthy." The images are loaded and preprocessed using the `skimage` library.

## Data Augmentation

Data augmentation is performed using the `ImageDataGenerator` class from TensorFlow to create augmented images for better model training.

## SVM Model Training

The Support Vector Machine (SVM) model is trained on the augmented dataset with radial basis function (RBF) kernel.

## Evaluation

The model is evaluated on a test set, and the performance metrics, including Accuracy, Precision, Recall, and F1 Score, are calculated.

## Requirements

- Python 3.x
- pandas
- scikit-learn
- matplotlib
- skimage
- TensorFlow
- cv2 (OpenCV)

## Usage

1. Clone or download this repository to your local machine.

2. Install the required dependencies listed in the `requirements.txt` file.

3. Run the `image_classification.py` script to perform SVM-based image classification.

## Results

The SVM model achieves the following performance on the test data:
- Accuracy: 97.21%
- Precision: 97.62%
- Recall: 96.47%
- F1 Score: 97.04%

## Visualization

A 3D scatter plot is generated using Plotly Express to visualize the SVM model's performance metrics.



Author: [Your Name]
Contact: [Your Email Address]

---

Replace `[Your Name]` and `[Your Email Address]` with your actual name and email address. Additionally, you can provide more details about the project, its purpose, and potential improvements in the README file.
