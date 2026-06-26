import os
import joblib

def save_model(model, filepath):
    """
    Save the trained model to a file using joblib.
    """
    # Ensure directory exists
    dir_name = os.path.dirname(filepath)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    
    # Save the model
    joblib.dump(model, filepath)
    print(f"Model saved successfully to: {filepath}")

def load_model(filepath):
    """
    Load the saved model from a file using joblib.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No model file found at: {filepath}")
    
    # Load the model
    model = joblib.load(filepath)
    print(f"Model loaded successfully from: {filepath}")
    return model

if __name__ == "__main__":
    # Example usage for loading the pre-trained SVM model:
    model_path = "models/svm_cat_dog_classifier.pkl"
    scaler_path = "models/scaler.pkl"
    
    print("--- Loading Saved Models ---")
    try:
        # Load the SVM model
        loaded_svm = load_model(model_path)
        print(f"Loaded Model Type: {type(loaded_svm)}")
        print(f"Model Parameters: {loaded_svm.get_params()}")
        
        # Load the fitted StandardScaler
        loaded_scaler = load_model(scaler_path)
        print(f"Loaded Scaler Type: {type(loaded_scaler)}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
