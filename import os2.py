import os
import shutil
from sklearn.model_selection import train_test_split

# Define the path to the main folder
main_dir = r'C:/Users/matia/OneDrive/IFTS/MODELIZADO DE SISTEMAS DE IA/Modelizado/image_set'

# Define paths for train, validation, and test folders
train_dir = os.path.join(main_dir, 'train')
val_dir = os.path.join(main_dir, 'val')
test_dir = os.path.join(main_dir, 'test')

# Create the new directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Get the list of all subfolders (categories) in the main directory
subfolders = [folder for folder in os.listdir(main_dir) 
              if os.path.isdir(os.path.join(main_dir, folder)) 
              and folder not in ['train', 'val', 'test']]

# Loop through each subfolder (category) and split its images
for subfolder in subfolders:
    subfolder_path = os.path.join(main_dir, subfolder)
    
    # Get all image filenames in the current subfolder
    images = os.listdir(subfolder_path)
    
    # Split images into train, validation, and test sets
    train_images, temp_images = train_test_split(images, test_size=0.3, random_state=42)
    val_images, test_images = train_test_split(temp_images, test_size=0.5, random_state=42)
    
    # Create subfolders for the current category in train, val, and test directories
    os.makedirs(os.path.join(train_dir, subfolder), exist_ok=True)
    os.makedirs(os.path.join(val_dir, subfolder), exist_ok=True)
    os.makedirs(os.path.join(test_dir, subfolder), exist_ok=True)
    
    # Move images to the corresponding directories
    for image in train_images:
        shutil.move(os.path.join(subfolder_path, image), os.path.join(train_dir, subfolder, image))
    for image in val_images:
        shutil.move(os.path.join(subfolder_path, image), os.path.join(val_dir, subfolder, image))
    for image in test_images:
        shutil.move(os.path.join(subfolder_path, image), os.path.join(test_dir, subfolder, image))
    
    # Remove the now-empty subfolder
    if not os.listdir(subfolder_path):  # Check if the folder is empty
        os.rmdir(subfolder_path)
