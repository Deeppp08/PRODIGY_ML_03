import os
import sys
import random
import joblib
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

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

def load_test_dataset_with_paths(base_dir):
    """
    Loads test images, extracts HOG features, and tracks image paths.
    """
    features_list = []
    labels_list = []
    paths_list = []
    
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
                paths_list.append(img_path)
                
    return np.array(features_list), np.array(labels_list), paths_list

def plot_confusion_matrix_heatmap(cm, output_path):
    """
    Plots a highly professional and clean Heatmap of the confusion matrix.
    """
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    
    # Custom color mapping using a clean blue palette
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Labels
    classes = ['Cat', 'Dog']
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=11, fontweight='bold', color='#1e293b')
    ax.set_yticklabels(classes, fontsize=11, fontweight='bold', color='#1e293b')
    
    # Rotate tick labels and set alignment
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    
    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            percentage = (count / cm.sum()) * 100
            label_text = f"{count}\n({percentage:.1f}%)"
            
            # Use white text for dark cells, black/navy text for light cells
            color = "white" if cm[i, j] > thresh else "#0f172a"
            ax.text(j, i, label_text,
                    ha="center", va="center",
                    color=color, fontsize=12, fontweight='bold')
    
    # Design borders, labels and titles
    ax.spines[:].set_color('#cbd5e1')
    ax.spines[:].set_linewidth(1.5)
    
    plt.title('SVM Classifier - Confusion Matrix Heatmap', fontsize=13, fontweight='bold', pad=20, color='#0f172a')
    ax.set_ylabel('True Label', fontsize=11, fontweight='bold', labelpad=12, color='#334155')
    ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold', labelpad=12, color='#334155')
    ax.grid(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_image_grid(image_paths, true_labels, pred_labels, title, output_path, cols=4, rows=4):
    """
    Plots a grid of images with Actual vs Predicted labels.
    Correct predictions are labeled in Green, incorrect in Red.
    """
    n_images = cols * rows
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.2))
    axes = axes.flatten()
    
    classes = ['Cat', 'Dog']
    
    for i in range(n_images):
        if i >= len(image_paths):
            axes[i].axis('off')
            continue
            
        img_path = image_paths[i]
        true_lbl = classes[true_labels[i]]
        pred_lbl = classes[pred_labels[i]]
        
        # Load image and convert to RGB
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Display image
        axes[i].imshow(img_rgb)
        axes[i].axis('off')
        
        # Determine title color: green if correct, red if incorrect
        is_correct = true_labels[i] == pred_labels[i]
        color = '#15803d' if is_correct else '#b91c1c'  # Dark Green / Dark Red
        
        status = "Correct" if is_correct else "Incorrect"
        title_text = f"Act: {true_lbl} | Pred: {pred_lbl}\n({status})"
        axes[i].set_title(title_text, fontsize=10, fontweight='bold', color=color, pad=8)
        
        # Add a border around axes
        for spine in axes[i].spines.values():
            spine.set_visible(True)
            spine.set_color('#e2e8f0')
            spine.set_linewidth(1.5)
            
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98, color='#0f172a')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def main():
    print("=" * 60)
    print("          PLOT GENERATION PIPELINE          ")
    print("=" * 60)
    
    model_path = "models/svm_cat_dog_classifier.pkl"
    scaler_path = "models/scaler.pkl"
    test_dir = "data/test"
    output_dir = "outputs"
    
    # Verify files
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("Error: Model or Scaler not found in the 'models' directory.")
        sys.exit(1)
        
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load test dataset
    print("\nLoading test dataset...")
    X_test, y_test, paths_test = load_test_dataset_with_paths(test_dir)
    print(f"Successfully loaded {len(X_test)} test samples.")
    
    # 2. Load model and scaler
    print("\nLoading models...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # 3. Predict
    print("\nRunning inference...")
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    # 4. Generate Confusion Matrix Heatmap
    print("\nGenerating Plot 1: Confusion Matrix Heatmap...")
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix_heatmap(cm, os.path.join(output_dir, "confusion_matrix_heatmap.png"))
    
    # Calculate correct and incorrect indices
    correct_indices = np.where(y_test == y_pred)[0]
    incorrect_indices = np.where(y_test != y_pred)[0]
    
    # 5. Generate Plot 2: Actual vs Predicted Image Grid (Random 16 samples)
    print("\nGenerating Plot 2: Actual vs Predicted Grid...")
    random.seed(42)  # Set seed for reproducible grid selection
    grid_indices = random.sample(range(len(y_test)), 16)
    
    grid_paths = [paths_test[idx] for idx in grid_indices]
    grid_true = [y_test[idx] for idx in grid_indices]
    grid_pred = [y_pred[idx] for idx in grid_indices]
    
    plot_image_grid(
        grid_paths, grid_true, grid_pred,
        "Actual vs Predicted Predictions Grid (Test Set)",
        os.path.join(output_dir, "actual_vs_predicted_grid.png"),
        cols=4, rows=4
    )
    
    # 6. Generate Plot 3: Sample Correct Predictions (8 samples: 4 cats, 4 dogs)
    print("\nGenerating Plot 3: Sample Correct Predictions...")
    correct_cat_indices = np.where((y_test == 0) & (y_pred == 0))[0]
    correct_dog_indices = np.where((y_test == 1) & (y_pred == 1))[0]
    
    # Select 4 random cats and 4 random dogs
    sel_correct_cats = random.sample(list(correct_cat_indices), min(4, len(correct_cat_indices)))
    sel_correct_dogs = random.sample(list(correct_dog_indices), min(4, len(correct_dog_indices)))
    correct_sel_indices = sel_correct_cats + sel_correct_dogs
    
    correct_paths = [paths_test[idx] for idx in correct_sel_indices]
    correct_true = [y_test[idx] for idx in correct_sel_indices]
    correct_pred = [y_pred[idx] for idx in correct_sel_indices]
    
    plot_image_grid(
        correct_paths, correct_true, correct_pred,
        "Sample Correct Predictions (True Positives & True Negatives)",
        os.path.join(output_dir, "sample_correct_predictions.png"),
        cols=4, rows=2
    )
    
    # 7. Generate Plot 4: Sample Incorrect Predictions (8 samples: 4 cat-as-dog, 4 dog-as-cat)
    print("\nGenerating Plot 4: Sample Incorrect Predictions...")
    incorrect_cat_indices = np.where((y_test == 0) & (y_pred == 1))[0]  # Cat predicted as Dog
    incorrect_dog_indices = np.where((y_test == 1) & (y_pred == 0))[0]  # Dog predicted as Cat
    
    sel_incorrect_cats = random.sample(list(incorrect_cat_indices), min(4, len(incorrect_cat_indices)))
    sel_incorrect_dogs = random.sample(list(incorrect_dog_indices), min(4, len(incorrect_dog_indices)))
    incorrect_sel_indices = sel_incorrect_cats + sel_incorrect_dogs
    
    incorrect_paths = [paths_test[idx] for idx in incorrect_sel_indices]
    incorrect_true = [y_test[idx] for idx in incorrect_sel_indices]
    incorrect_pred = [y_pred[idx] for idx in incorrect_sel_indices]
    
    plot_image_grid(
        incorrect_paths, incorrect_true, incorrect_pred,
        "Sample Incorrect Predictions (False Positives & False Negatives)",
        os.path.join(output_dir, "sample_incorrect_predictions.png"),
        cols=4, rows=2
    )
    
    print("\n" + "=" * 60)
    print("All plots generated and saved to the 'outputs' directory.")
    print("=" * 60)

if __name__ == "__main__":
    main()
