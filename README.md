# 🐱🐶 Cats vs. Dogs Image Classification using Support Vector Machine (SVM)

[![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-white?logo=opencv&logoColor=red)](https://opencv.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg?style=flat&logo=Jupyter)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains a complete, end-to-end Machine Learning pipeline to classify images of **Cats** and **Dogs** using a **Support Vector Machine (SVM)** classifier. By combining classical computer vision shape descriptors (**Histogram of Oriented Gradients (HOG)**) with hyperparameter-tuned Support Vector Classifiers (SVC), this project demonstrates high-accuracy classification without the complexity or hardware overhead of Deep Learning.

---

## 📁 Repository Structure

```text
PRODIGY_ML_03/
│
├── data/
│   ├── train/
│   │   ├── cats/             # 1,000 Cat training images (cat_0.jpg ... cat_999.jpg)
│   │   └── dogs/             # 1,000 Dog training images (dog_0.jpg ... dog_999.jpg)
│   └── test/
│       ├── cats/             # 500 Cat test images (cat_0.jpg ... cat_499.jpg)
│       └── dogs/             # 500 Dog test images (dog_0.jpg ... dog_499.jpg)
│
├── models/
│   ├── svm_cat_dog_classifier.pkl  # Serialized best-performing RBF SVM model
│   └── scaler.pkl                  # Serialized StandardScaler config
│
├── outputs/
│   ├── confusion_matrix_heatmap.png # Heatmap plot with counts and percentages
│   ├── actual_vs_predicted_grid.png # 4x4 sample prediction grid (Green=Correct, Red=Incorrect)
│   ├── sample_correct_predictions.png # Sample grid of true positives/negatives
│   └── sample_incorrect_predictions.png # Sample grid of false positives/negatives
│
├── notebooks/
│   └── svm_cat_dog_classifier.ipynb # Main Jupyter Notebook showing pipeline
│
├── setup_data.py             # Automater to download, clean, and organize dataset
├── evaluate_model.py         # Script to calculate classification metrics on test set
├── generate_plots.py         # Script to generate heatmap, grid, and samples
├── load_model_demo.py        # Demo script showing model load/save workflow
├── predict.py                # Command-line interactive prediction tool
├── requirements.txt          # Python packages list
└── README.md                 # Project documentation (this file)
```

---

## 📊 Dataset Information

This project utilizes **Microsoft's Cats & Dogs Dataset** (from the Kaggle competition).
* **Download Automation**: The `setup_data.py` script automatically downloads the raw dataset (~786 MB), filters out corrupted headers and empty image files, and creates a balanced partitioning:
  * **Training Set**: 2,000 images (1,000 cats, 1,000 dogs).
  * **Test Set**: 1,000 images (500 cats, 500 dogs).
* All images are named cleanly (`cat_x.jpg` or `dog_x.jpg`) and sorted into category-specific folders.

---

## 🛠️ Technologies Used

* **Core Language**: Python (3.8+)
* **Machine Learning**: `scikit-learn` (SVC, GridSearchCV, metrics)
* **Computer Vision & Preprocessing**: `opencv-python` (Grayscale, resize, HOG)
* **Serialization**: `joblib` (saving/loading binary models)
* **Data Processing**: `numpy`
* **Plotting & Visualization**: `matplotlib`
* **UI & Console Utilities**: `tqdm`, `requests`

---

## 🔬 Pipeline & Methodology

### 1. Image Preprocessing & Feature Extraction
Unlike raw pixel vectors which are highly sensitive to noise, translations, and lighting, shape and edge-based features are extracted:
* **Grayscale Conversion**: Reduces color dimensions, focusing on structural boundaries.
* **Resizing**: Standardizes image grids to $64\times64$ pixels.
* **HOG Descriptor Configuration**:
  * Window size: $64\times64$
  * Block size: $16\times16$
  * Block stride: $8\times8$
  * Cell size: $8\times8$
  * Orientation bins: $9$
* **Resulting Vector**: Extends $1,764$ descriptive features representing structural histograms.
* **Standardization**: Features are normalized using `StandardScaler` to zero mean and unit variance.

### 2. SVM Model & Training Process
The classifier is trained using scikit-learn's Support Vector Classifier (SVC):
* **Hyperparameter Tuning**: Cross-validation (`GridSearchCV`, 3-fold) optimizes parameters over a grid search:
  * Regularization $C \in [0.1, 1, 10]$
  * Kernel functions: `linear`, `rbf` (Radial Basis Function)
  * Gamma setting: `scale`
* **Optimal Model**: RBF kernel with regularization $C=10$.
* **Model Serialization**: The optimized model is saved using `joblib` to `models/svm_cat_dog_classifier.pkl`.

---

## 📈 Evaluation Metrics & Results

The classifier was evaluated on the unseen test set ($1,000$ images). The metrics show robust, balanced classification:

### Classification Performance Table

| Metric | Score (Macro Avg) | Score (Dog / Positive) | Description |
| :--- | :---: | :---: | :--- |
| **Accuracy** | **76.50%** | **76.50%** | Overall correct predictions |
| **Precision** | **76.50%** | **76.77%** | Out of predicted positives, how many were correct |
| **Recall** | **76.50%** | **76.00%** | Out of actual positives, how many were identified |
| **F1 Score** | **76.50%** | **76.38%** | Harmonic mean of Precision and Recall |

### Detailed Classification Report

```text
              precision    recall  f1-score   support

         Cat       0.76      0.77      0.77       500
         Dog       0.77      0.76      0.76       500

    accuracy                           0.77      1000
   macro avg       0.77      0.77      0.76      1000
weighted avg       0.77      0.77      0.76      1000
```

### Confusion Matrix Results
* **True Cats**: 385 (True Negatives)
* **False Dogs**: 115 (False Positives)
* **False Cats**: 120 (False Negatives)
* **True Dogs**: 380 (True Positives)

All plots (Heatmap, Actual vs Predicted Grid, Correct/Incorrect predictions) can be found in the `outputs/` folder.

---

## 🚀 Installation & Usage

### 1. Clone the Directory & Install Dependencies
Navigate to the directory and run:
```bash
pip install -r requirements.txt
```

### 2. Run Data Setup
Download, extract, validate, and structure the dataset:
```bash
python setup_data.py
```

### 3. Evaluate the Model
Compute metrics (Accuracy, Precision, Recall, F1) on the test set:
```bash
python evaluate_model.py
```

### 4. Generate Visualization Plots
Generate the confusion matrix heatmap, 16-sample predictions grid, and samples of correct/incorrect classifications:
```bash
python generate_plots.py
```

### 5. Interactive Custom Image Prediction
Test predictions on your own custom images using the interactive CLI tool:
```bash
python predict.py
```
*(Enter the path to your image when prompted; type `q` to quit).*

### 6. Verify Model Loading (API Demo)
To verify the serialization load process via scikit-learn/joblib API:
```bash
python load_model_demo.py
```

---

## 💡 Future Improvements

1. **Feature Merging**: Combine HOG shape features with color histogram descriptors (e.g. RGB/HSV histograms) to account for color patterns.
2. **Dimension Reduction**: Apply PCA (Principal Component Analysis) to compress HOG feature vectors and accelerate prediction inference.
3. **Deep Learning Comparison**: Develop a baseline Convolutional Neural Network (CNN) to compare computational efficiency and performance boundaries against classical classifiers.

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
