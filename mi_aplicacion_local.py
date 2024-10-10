import os
import flet as ft
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import tensorflow as tf

# Suprimir los mensajes informativos de TensorFlow
tf.get_logger().setLevel('ERROR')

# Desactivar optimizaciones oneDNN (si es necesario)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Cargar el modelo
model = load_model(r'C:/Users/matia/OneDrive/IFTS/MODELIZADO DE SISTEMAS DE IA/Modelizado/Modelo/vehiculos2.keras')

# Diccionario de etiquetas
label_list = {
    0: 'airplane', 1: 'ambulance', 2: 'bicycle', 3: 'boat', 4: 'bus', 
    5: 'car', 6: 'fire_truck', 7: 'helicopter', 8: 'hovercraft', 9: 'jet_ski', 
    10: 'kayak', 11: 'motorcycle', 12: 'rickshaw', 13: 'scooter', 14: 'segway', 
    15: 'skateboard', 16: 'tractor', 17: 'truck', 18: 'unicycle', 19: 'van'
}

def predict_image(file_path):
    # Preprocesar la imagen
    img = image.load_img(file_path, target_size=(256, 256))  # Ajustar tamaño
    img_array = image.img_to_array(img) / 255.0  # Normalizar
    img_array = np.expand_dims(img_array, axis=0)  # Ajustar la dimensión

    # Realizar la predicción
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    
    # Obtener la etiqueta de la clase
    predicted_label = label_list.get(predicted_class, "Unknown Class")
    
    return predicted_label

def main(page):
    page.title = "Clasificador de Imágenes"
    page.bgcolor = ft.colors.GREY_900  # Fondo oscuro

    # Crear y centrar el texto
    page.add(
        ft.Column([
            ft.Text("Carga una imagen para predecir su clase", color=ft.colors.WHITE, size=44)
        ], alignment=ft.MainAxisAlignment.CENTER)
    )

    # Crear un contenedor de carga de archivos
    file_picker = ft.FilePicker()
    file_picker.on_result = lambda e: on_file_selected(e, page)  # Usar una función de resultado
    
    # Crear un contenedor con el botón centrado
    page.add(
        ft.Column([
            ft.ElevatedButton("Selecciona una imagen", on_click=lambda e: file_picker.pick_files(allow_multiple=False), bgcolor=ft.colors.BLUE_400)
        ], alignment=ft.MainAxisAlignment.CENTER)
    )
    
    page.add(file_picker)

    # Crear un lugar para mostrar la imagen y la predicción
    image_container = ft.Column()
    page.add(image_container)

    # Crear un cuadro de texto para mostrar la predicción
    prediction_text = ft.Text("", color=ft.colors.WHITE, size=20)
    page.add(prediction_text)

    def on_file_selected(e, page):
        if e.files:
            file_path = e.files[0].path  # Ruta del archivo seleccionado
            # Mostrar la imagen cargada
            image_container.controls.clear()  # Limpiar cualquier imagen anterior
            image_container.controls.append(ft.Image(src=file_path, fit=ft.ImageFit.CONTAIN))  # Mostrar la nueva imagen
            page.update()

            # Hacer la predicción
            predicted_label = predict_image(file_path)
            prediction_text.value = f"Predicción: {predicted_label}"
            page.update()

# Ejecutar la aplicación
ft.app(target=main)
