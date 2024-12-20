import os
import flet as ft
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import tensorflow as tf

# Suprimir mensajes informativos de TensorFlow
tf.get_logger().setLevel('ERROR')

# Cargar el modelo
model = load_model(r'C:/Users/diego/Desktop/TSCDIA/Modelizado de Sistemas de IA/TP1/Modelizado/Modelos/vehiculos_16.keras')

# Diccionario de etiquetas
label_list = {
    0: 'avión', 1: 'ambulancia', 2: 'bicicleta', 3: 'bote', 4: 'autobús', 
    5: 'coche', 6: 'camión de bomberos', 7: 'helicóptero', 8: 'hovercraft', 9: 'moto de agua', 
    10: 'kayak', 11: 'motocicleta', 12: 'rickshaw', 13: 'scooter', 14: 'segway', 
    15: 'barco', 16: 'patineta', 17: 'tractor', 18: 'tren', 19: 'camión', 20: 'uniciclo', 21: 'furgoneta'
}

def predict_image(file_path):
    # Preprocesar la imagen
    img = image.load_img(file_path, target_size=(256, 256))  # Ajustar tamaño para el modelo
    img_array = image.img_to_array(img) / 255.0  # Normalizar
    img_array = np.expand_dims(img_array, axis=0)  # Expandir dimensiones

    # Hacer la predicción
    predictions = model.predict(img_array)
    
    # Obtener los índices y valores de las tres predicciones más altas
    top_indices = np.argsort(predictions[0])[::-1][:3]
    top_probabilities = predictions[0][top_indices] * 100  # Convertir a porcentaje
    top_labels = [label_list[i] for i in top_indices]
    
    return top_labels, top_probabilities

def main(page):
    page.title = "Clasificador de Imágenes"
    page.bgcolor = ft.colors.GREY_900  # Fondo oscuro

    # Crear y centrar el texto
    page.add(
        ft.Column([ft.Text("Carga una imagen para predecir su clase", color=ft.colors.WHITE, size=44)],
                  alignment=ft.MainAxisAlignment.CENTER)
    )

    # Crear un selector de archivos
    file_picker = ft.FilePicker()
    file_picker.on_result = lambda e: on_file_selected(e, page)
    
    # Crear un botón para activar el selector de archivos
    page.add(
        ft.Column([ft.ElevatedButton("Seleccionar una imagen", on_click=lambda e: file_picker.pick_files(allow_multiple=False), bgcolor=ft.colors.BLUE_400)],
                  alignment=ft.MainAxisAlignment.CENTER)
    )
    
    page.add(file_picker)

    # Crear un lugar para mostrar la imagen y la predicción
    image_container = ft.Column()
    page.add(image_container)

    def on_file_selected(e, page):
        if e.files:
            file_path = e.files[0].path

            # Limpiar y actualizar el contenedor de la imagen
            image_container.controls.clear()

            # Hacer la predicción
            top_labels, top_probabilities = predict_image(file_path)
            
            # Formatear la predicción y la confianza en una sola línea
            prediction_text = ft.Text(
                f"Predicción: {top_labels[0]} ({top_probabilities[0]:.2f}%), "
                f"{top_labels[1]} ({top_probabilities[1]:.2f}%), "
                f"{top_labels[2]} ({top_probabilities[2]:.2f}%)", 
                color=ft.colors.WHITE, 
                size=20
            )
            
            # Añadir el texto de predicción **antes** de la imagen
            image_container.controls.append(prediction_text)

            # Mostrar la imagen
            new_image = ft.Image(src=file_path, fit=ft.ImageFit.CONTAIN)
            image_container.controls.append(new_image)

            # Actualizar la página para reflejar los cambios
            page.update()

# Ejecutar la aplicación
ft.app(target=main)
