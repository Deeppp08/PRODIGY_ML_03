import os
import sys
import joblib
import cv2
import numpy as np

def extract_hog_features(img):
    """
    Computes the HOG feature vector for a 64x64 grayscale image.
    """
    win_size = (64, 64)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    features = hog.compute(img)
    
    if features is None:
        raise ValueError("HOG feature extraction failed.")
        
    return features.flatten()

def main():
    print("=" * 60)
    print("           SVM CAT VS DOG IMAGE PREDICTOR           ")
    print("=" * 60)
    
    model_path = "models/svm_cat_dog_classifier.pkl"
    scaler_path = "models/scaler.pkl"
    
    # 1. Load the saved SVM model & Scaler
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("Error: Trained model and/or scaler files not found in the 'models' directory.")
        print("Please train the model or ensure models exist before running predictions.")
        sys.exit(1)
        
    print("Loading model and scaler...")
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading model files: {e}")
        sys.exit(1)
        
    print("-" * 60)
    
    # 2. Ask user for an image path
    while True:
        img_path = input("Enter the path to an image file (or 'q' to quit): ").strip()
        if img_path.lower() == 'q':
            print("Exiting predictor. Goodbye!")
            break
            
        if not img_path:
            continue
            
        if not os.path.exists(img_path):
            print(f"Error: File does not exist at '{img_path}'. Please try again.\n")
            continue
            
        # 3. Read image
        img = cv2.imread(img_path)
        if img is None:
            print("Error: Could not read image. It may be corrupt or an unsupported format.\n")
            continue
            
        print("\nProcessing image...")
        try:
            # 4. Convert to Grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 5. Resize to 64x64
            resized = cv2.resize(gray, (64, 64))
            
            # 6. Extract HOG features (shape representation)
            features = extract_hog_features(resized)
            
            # 7. Normalize features using StandardScaler
            features_scaled = scaler.transform(features.reshape(1, -1))
            
            # 8. Predict Class
            prediction = model.predict(features_scaled)[0]
            class_label = "Dog" if prediction == 1 else "Cat"
            
            # 9. Get confidence score
            confidence_str = ""
            if hasattr(model, "predict_proba"):
                try:
                    probs = model.predict_proba(features_scaled)[0]
                    confidence = probs[prediction] * 100
                    confidence_str = f"Confidence: {confidence:.2f}%"
                except Exception:
                    pass
            
            # Fallback to decision score if probabilities are not computed
            if not confidence_str and hasattr(model, "decision_function"):
                decision_score = model.decision_function(features_scaled)[0]
                confidence_str = f"Decision Score: {decision_score:.4f} (Positive = Dog, Negative = Cat)"
            
            # Display prediction results professionally
            print("\n" + "*" * 50)
            print(f"  PREDICTED CLASS : {class_label.upper()}")
            if confidence_str:
                print(f"  {confidence_str}")
            print("*" * 50 + "\n")
            
        except Exception as e:
            print(f"Error processing image or predicting: {e}\n")

if __name__ == "__main__":
    main()
