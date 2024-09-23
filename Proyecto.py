{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
    "\n",
    "train_dir = 'C:/Users/matia/OneDrive/IFTS/MODELIZADO DE SISTEMAS DE IA/Modelizado/image_set'\n",
    "val_dir = 'C:/Users/matia/OneDrive/IFTS/MODELIZADO DE SISTEMAS DE IA/Modelizado/image_set'\n",
    "\n",
    "# Generadores de imágenes con aumento de datos\n",
    "train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2,\n",
    "                                   height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,\n",
    "                                   horizontal_flip=True, fill_mode='nearest')\n",
    "val_datagen = ImageDataGenerator(rescale=1./255)\n",
    "\n",
    "train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), \n",
    "                                                    batch_size=32, class_mode='categorical')\n",
    "val_generator = val_datagen.flow_from_directory(val_dir, target_size=(224, 224), \n",
    "                                                batch_size=32, class_mode='categorical')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from tensorflow.keras.applications import Xception\n",
    "from tensorflow.keras import layers, models\n",
    "\n",
    "base_model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))\n",
    "base_model.trainable = False  # Congelamos las capas del modelo preentrenado\n",
    "\n",
    "model = models.Sequential([\n",
    "    base_model,\n",
    "    layers.GlobalAveragePooling2D(),\n",
    "    layers.Dense(1024, activation='relu'),\n",
    "    layers.Dropout(0.5),\n",
    "    layers.Dense(train_generator.num_classes, activation='softmax')\n",
    "])\n",
    "\n",
    "model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])\n",
    "\n",
    "model.summary()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from tensorflow.keras.preprocessing import image\n",
    "import numpy as np\n",
    "\n",
    "img_path = 'C:/Users/matia/OneDrive/IFTS/MODELIZADO DE SISTEMAS DE IA/images.jpeg'\n",
    "img = image.load_img(img_path, target_size=(224, 224))\n",
    "img_array = image.img_to_array(img) / 255.0\n",
    "img_array = np.expand_dims(img_array, axis=0)\n",
    "\n",
    "predictions = model.predict(img_array)\n",
    "predicted_class = np.argmax(predictions[0])\n",
    "print(f'Predicted class: {predicted_class}')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "label_list = train_generator.class_indices\n",
    "# Invertimos el diccionario para que podamos buscar los nombres de las clases por índice\n",
    "label_list = {v: k for k, v in label_list.items()}\n",
    "predicted_label = label_list[predicted_class]\n",
    "print(f'This vehicle is predicted to be a: {predicted_label}')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "base_model.trainable = True  # Descongelar todas las capas\n",
    "\n",
    "# O descongelar parcialmente:\n",
    "for layer in base_model.layers[:100]:\n",
    "    layer.trainable = False\n",
    "for layer in base_model.layers[100:]:\n",
    "    layer.trainable = True\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),\n",
    "              loss='categorical_crossentropy', metrics=['accuracy'])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_datagen = ImageDataGenerator(\n",
    "    rescale=1./255,\n",
    "    rotation_range=40,\n",
    "    width_shift_range=0.2,\n",
    "    height_shift_range=0.2,\n",
    "    shear_range=0.2,\n",
    "    zoom_range=0.2,\n",
    "    horizontal_flip=True,\n",
    "    fill_mode='nearest'\n",
    ")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = models.Sequential([\n",
    "    base_model,\n",
    "    layers.GlobalAveragePooling2D(),\n",
    "    layers.Dense(1024, activation='relu'),\n",
    "    layers.Dropout(0.5),\n",
    "    layers.Dense(512, activation='relu'),  # Añadir una capa densa adicional\n",
    "    layers.Dropout(0.5),\n",
    "    layers.Dense(train_generator.num_classes, activation='softmax')\n",
    "])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),  # Tasa más baja\n",
    "              loss='categorical_crossentropy', metrics=['accuracy'])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "class_weights = {0: 1.0, 1: 2.0, ...}  # Ajustar el peso de cada clase\n",
    "model.fit(train_generator, class_weight=class_weights, epochs=50, validation_data=val_generator)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "history = model.fit(train_generator, epochs=50, validation_data=val_generator,\n",
    "                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "test_dir = 'C:/Users/matia/OneDrive/IFTS/MODELIZADO DE SISTEMAS DE IA/Modelizado/image_set'\n",
    "test_datagen = ImageDataGenerator(rescale=1./255)\n",
    "test_generator = test_datagen.flow_from_directory(test_dir, target_size=(224, 224), \n",
    "                                                  batch_size=32, class_mode='categorical')\n",
    "\n",
    "test_loss, test_acc = model.evaluate(test_generator)\n",
    "print(f'Test accuracy: {test_acc}')\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
