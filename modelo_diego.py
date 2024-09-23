import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Update the paths to point to your image directories with forward slashes
base_dir = r'C:/Users/matia/OneDrive/IFTS/MODELIZADO DE SISTEMAS DE IA/Modelizado/image_set'  # Base directory
train_dir = base_dir + r'/train'
val_dir = base_dir + r'/val'
test_dir = base_dir + r'/test'

# Image data generators with data augmentation for the training set
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2,
                                   height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                   horizontal_flip=True, fill_mode='nearest')

# Validation and test data generators without augmentation
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load the training data with target size set to 256x256 (since your images are already that size)
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(256, 256), 
                                                    batch_size=32, class_mode='categorical')

# Load the validation data with target size 256x256
val_generator = val_datagen.flow_from_directory(val_dir, target_size=(256, 256), 
                                                batch_size=32, class_mode='categorical')

# Load the Xception model as a base (pretrained on ImageNet)
from tensorflow.keras.applications import Xception
from tensorflow.keras import layers, models

base_model = Xception(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
base_model.trainable = False  # Freeze the pretrained model layers

# Build the full model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()

# Train the model
history = model.fit(train_generator, epochs=10, validation_data=val_generator)

# Load and evaluate on test data with target size 256x256
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(256, 256), 
                                                  batch_size=32, class_mode='categorical')

test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc}')

# Predict on a single image
from tensorflow.keras.preprocessing import image
import numpy as np

img_path = r'C:/Users/matia/OneDrive/IFTS/MODELIZADO DE SISTEMAS DE IA/images.jpeg'  # Update this path
img = image.load_img(img_path, target_size=(256, 256))  # Adjust to 256x256
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])
print(f'Predicted class: {predicted_class}')
