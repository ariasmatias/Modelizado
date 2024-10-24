from PIL import Image, UnidentifiedImageError
import os

# Define the root folder containing the original images
root_folder = 'C:/Users/diego/Desktop/TSCDIA/Modelizado de Sistemas de IA/TP1/image_set'
# Define the new folder where resized images will be saved
new_root_folder = 'C:/Users/diego/Desktop/TSCDIA/Modelizado de Sistemas de IA/TP1/Modelizado/image_set'
new_size = (256, 256)  # Desired size (width, height)

# Traverse all subdirectories and files
for dirpath, dirnames, filenames in os.walk(root_folder):
    # Determine the relative path from the root folder
    relative_path = os.path.relpath(dirpath, root_folder)
    new_dirpath = os.path.join(new_root_folder, relative_path)  # New path for resized images

    # Create the new directory if it doesn't exist
    os.makedirs(new_dirpath, exist_ok=True)

    count = 1  # Start numbering from 1 for each subfolder

    for filename in filenames:
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):  # Supported extensions
            img_path = os.path.join(dirpath, filename)

            # Create a new filename based on subfolder names
            new_filename = f"{os.path.basename(os.path.dirname(dirpath))}_{os.path.basename(dirpath)}_{count}.jpg"
            new_img_path = os.path.join(new_dirpath, new_filename)

            # Check if the resized image already exists
            if os.path.exists(new_img_path):
                print(f"Image already processed: {new_img_path}. Skipping.")
                count += 1  # Increment the counter for the next file
                continue

            try:
                # Open the image
                with Image.open(img_path) as img:
                    # If the file is a GIF, convert it to JPEG
                    if filename.endswith('.gif'):
                        img = img.convert('RGB')  # Convert to RGB, flattening transparency
                    else:
                        img = img.convert('RGB')  # Ensure consistency for other formats

                    # Resize the image
                    img_resized = img.resize(new_size)

                    # Save the resized image
                    img_resized.save(new_img_path)
                    print(f"Resized and saved: {new_img_path}")
                    count += 1  # Increment the counter for each image

            except UnidentifiedImageError:
                print(f"Error opening image: {img_path}. Skipping this file.")
            except Exception as e:
                print(f"An error occurred with file {img_path}: {e}")

print("All images resized, renamed, and saved in place.")
