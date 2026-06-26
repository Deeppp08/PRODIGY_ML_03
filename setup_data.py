import os
import zipfile
import requests
import cv2
from tqdm import tqdm

URL = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
ZIP_PATH = "kagglecatsanddogs_5340.zip"
TARGET_DIR = "data"

def download_file(url, dest):
    print(f"Downloading dataset from {url}...")
    print("This file is ~786 MB and will take some time depending on your internet connection.")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024 # 1MB blocks
    
    with open(dest, 'wb') as file, tqdm(
        desc="Downloading ZIP",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)

def is_image_valid(file_path):
    """Check if the image can be loaded successfully by OpenCV."""
    try:
        img = cv2.imread(file_path)
        if img is None:
            return False
        # Double check dimensions
        if img.shape[0] < 10 or img.shape[1] < 10:
            return False
        return True
    except Exception:
        return False

def main():
    # 1. Download ZIP if it doesn't exist
    if not os.path.exists(ZIP_PATH):
        try:
            download_file(URL, ZIP_PATH)
        except Exception as e:
            print(f"\nError downloading dataset: {e}")
            return
    else:
        print(f"Zip file {ZIP_PATH} already exists. Skipping download.")
        
    # 2. Setup target directories
    dest_train_cats = os.path.join(TARGET_DIR, "train", "cats")
    dest_train_dogs = os.path.join(TARGET_DIR, "train", "dogs")
    dest_test_cats = os.path.join(TARGET_DIR, "test", "cats")
    dest_test_dogs = os.path.join(TARGET_DIR, "test", "dogs")
    
    for d in [dest_train_cats, dest_train_dogs, dest_test_cats, dest_test_dogs]:
        os.makedirs(d, exist_ok=True)
        
    # 3. Extract and organize selectively
    print("Selectively extracting and validating images from ZIP...")
    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as z:
            # We want:
            # 1000 train cats, 500 test cats
            # 1000 train dogs, 500 test dogs
            cats_saved = 0
            dogs_saved = 0
            
            # Temporary folder to extract individual images for validation
            temp_extract_dir = "temp_extract"
            os.makedirs(temp_extract_dir, exist_ok=True)
            
            # Filter all files in zip that are jpg
            members = [m for m in z.infolist() if m.filename.endswith('.jpg') and not m.filename.startswith('__MACOSX')]
            
            pbar = tqdm(total=3000, desc="Extracting & Validating")
            
            for member in members:
                if cats_saved >= 1500 and dogs_saved >= 1500:
                    break
                    
                filename = member.filename
                # Structure: PetImages/Cat/0.jpg or PetImages/Dog/0.jpg
                is_cat = "Cat/" in filename
                is_dog = "Dog/" in filename
                
                if (is_cat and cats_saved >= 1500) or (is_dog and dogs_saved >= 1500):
                    continue
                    
                # Extract file to temp folder
                # We rename it to make it clean
                base_name = os.path.basename(filename)
                temp_file_path = os.path.join(temp_extract_dir, base_name)
                
                # Extract content
                with z.open(member) as source, open(temp_file_path, "wb") as target:
                    shutil.copyfileobj(source, target)
                    
                # Validate image
                if is_image_valid(temp_file_path):
                    if is_cat:
                        if cats_saved < 1000:
                            dest_path = os.path.join(dest_train_cats, f"cat_{cats_saved}.jpg")
                        else:
                            dest_path = os.path.join(dest_test_cats, f"cat_{cats_saved - 1000}.jpg")
                        shutil.move(temp_file_path, dest_path)
                        cats_saved += 1
                        pbar.update(1)
                    elif is_dog:
                        if dogs_saved < 1000:
                            dest_path = os.path.join(dest_train_dogs, f"dog_{dogs_saved}.jpg")
                        else:
                            dest_path = os.path.join(dest_test_dogs, f"dog_{dogs_saved - 1000}.jpg")
                        shutil.move(temp_file_path, dest_path)
                        dogs_saved += 1
                        pbar.update(1)
                else:
                    # Delete invalid file
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
            
            pbar.close()
            
            # Clean up temp extract directory
            if os.path.exists(temp_extract_dir):
                shutil.rmtree(temp_extract_dir)
                
            print(f"Extracted: {cats_saved} cats (1000 train, {cats_saved-1000} test)")
            print(f"Extracted: {dogs_saved} dogs (1000 train, {dogs_saved-1000} test)")
            
    except Exception as e:
        print(f"Error during selective extraction: {e}")
        return
        
    # 4. Clean up ZIP file
    print("Cleaning up ZIP file...")
    if os.path.exists(ZIP_PATH):
        try:
            os.remove(ZIP_PATH)
            print("ZIP file deleted successfully.")
        except Exception as e:
            print(f"Could not delete ZIP file: {e}")
            
    print("Dataset setup completed successfully!")

if __name__ == "__main__":
    import shutil
    main()
