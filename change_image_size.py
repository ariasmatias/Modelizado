from PIL import Image
import os

# Define the folder containing images and the output folder
input_folder = 'C:/Users/diego/Desktop/TSCDIA/Modelizado de Sistemas de IA/vehicle_data/van'
output_folder = 'C:/Users/diego/Desktop/TSCDIA/Modelizado de Sistemas de IA/image_set/van'
new_size = (256, 256)  # Desired size (width, height)



# Loop over all files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):  # Add more extensions if needed
        img_path = os.path.join(input_folder, filename)
        with Image.open(img_path) as img:
            # Resize the image
            img_resized = img.resize(new_size)
            # Handle different modes
            if img_resized.mode == 'P':
                # Convert to RGBA to handle transparency, then to RGB if saving as JPEG
                img_resized = img_resized.convert('RGBA')
            if img_resized.mode == 'RGBA':
                # Convert to RGB if saving as JPEG
                img_resized = img_resized.convert('RGB')
            output_path = os.path.join(output_folder, filename)
            img_resized.save(output_path)

print(f"Images resized and saved to {output_folder}")
