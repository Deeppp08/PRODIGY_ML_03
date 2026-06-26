import os
import sys
import joblib
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

def extract_hog_features(img_path):
    """
    Reads an image, converts it to grayscale, resizes it to 64x64,
    and computes the HOG feature vector.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    # Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize to 64x64
    resized = cv2.resize(gray, (64, 64))
    
    # Configure OpenCV HOG descriptor
    win_size = (64, 64)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    
    # Compute HOG feature vector
    features = hog.compute(resized)
    
    if features is None:
        return None
        
    return features.flatten()

def load_test_dataset(base_dir):
    """
    Walks through the subfolders of the base directory (cats/ and dogs/),
    extracts HOG features, and accumulates datasets.
    """
    features_list = []
    labels_list = []
    
    categories = ['cats', 'dogs']
    for category_idx, category in enumerate(categories):
        category_dir = os.path.join(base_dir, category)
        if not os.path.exists(category_dir):
            print(f"Warning: Folder {category_dir} not found. Skipping.")
            continue
            
        files = [f for f in os.listdir(category_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Processing {len(files)} files in category: {category}...")
        
        for file_name in tqdm(files, desc=f"Loading {category}"):
            img_path = os.path.join(category_dir, file_name)
            features = extract_hog_features(img_path)
            if features is not None:
                features_list.append(features)
                labels_list.append(category_idx)
                
    return np.array(features_list), np.array(labels_list)

def main():
    print("=" * 60)
    print("          SVM MODEL EVALUATION PIPELINE          ")
    print("=" * 60)
    
    model_path = "models/svm_cat_dog_classifier.pkl"
    scaler_path = "models/scaler.pkl"
    test_dir = "data/test"
    output_dir = "outputs"
    
    # Verify files
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'")
        sys.exit(1)
    if not os.path.exists(scaler_path):
        print(f"Error: Scaler file not found at '{scaler_path}'")
        sys.exit(1)
    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found at '{test_dir}'")
        sys.exit(1)
        
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Dataset
    print("\n[1/4] Loading and preprocessing test dataset...")
    X_test, y_test = load_test_dataset(test_dir)
    print(f"Successfully loaded {len(X_test)} samples.")
    
    # 2. Load Model and Scaler
    print("\n[2/4] Loading trained model and scaler...")
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        sys.exit(1)
        
    # 3. Feature Scaling & Prediction
    print("\n[3/4] Running feature scaling and model inference...")
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    # 4. Metric Computation
    print("\n[4/4] Evaluating metrics...")
    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average='macro')
    rec_macro = recall_score(y_test, y_pred, average='macro')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    prec_binary = precision_score(y_test, y_pred)
    rec_binary = recall_score(y_test, y_pred)
    f1_binary = f1_score(y_test, y_pred)
    
    # Classification Report
    class_report = classification_report(y_test, y_pred, target_names=['Cat', 'Dog'])
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Print Results Professionally to Terminal
    print("\n" + "=" * 60)
    print("                  EVALUATION METRICS                     ")
    print("=" * 60)
    print(f"  Accuracy  : {acc * 100:.2f}%")
    print(f"  Precision : {prec_macro * 100:.2f}% (Macro) | {prec_binary * 100:.2f}% (Binary/Dog)")
    print(f"  Recall    : {rec_macro * 100:.2f}% (Macro) | {rec_binary * 100:.2f}% (Binary/Dog)")
    print(f"  F1 Score  : {f1_macro * 100:.2f}% (Macro) | {f1_binary * 100:.2f}% (Binary/Dog)")
    print("-" * 60)
    print("Classification Report:")
    print(class_report)
    print("=" * 60)
    print("Confusion Matrix:")
    print(f"            Predicted Cat    Predicted Dog")
    print(f"Actual Cat      {cm[0,0]:<13}    {cm[0,1]:<13}")
    print(f"Actual Dog      {cm[1,0]:<13}    {cm[1,1]:<13}")
    print("=" * 60)
    
    # Generate and Save Professional Confusion Matrix Plot
    print("\nGenerating confusion matrix plot...")
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Customize display style
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Cat', 'Dog'])
    
    # Plot using a beautiful custom colormap
    disp.plot(cmap='Blues', values_format='d', ax=ax, colorbar=False)
    
    # Style details
    plt.title('SVM Classifier - Confusion Matrix (Test Set)', fontsize=14, fontweight='bold', pad=20, color='#1e293b')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold', labelpad=10, color='#334155')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold', labelpad=10, color='#334155')
    
    # Remove grid lines and adjust margins
    ax.grid(False)
    plt.tick_params(colors='#475569', labelsize=11)
    
    # Add text box with metrics inside plot area
    metric_text = (
        f"Accuracy: {acc * 100:.1f}%\n"
        f"Precision (Macro): {prec_macro * 100:.1f}%\n"
        f"Recall (Macro): {rec_macro * 100:.1f}%\n"
        f"F1-Score (Macro): {f1_macro * 100:.1f}%"
    )
    props = dict(boxstyle='round,pad=0.5', facecolor='#f8fafc', edgecolor='#cbd5e1', alpha=0.9)
    ax.text(1.55, 0.5, metric_text, transform=ax.transData, fontsize=10,
            verticalalignment='center', bbox=props, color='#1e293b')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "evaluation_confusion_matrix.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion Matrix saved to: {plot_path}")
    print("Evaluation completed successfully.")
    
if __name__ == "__main__":
    main()
