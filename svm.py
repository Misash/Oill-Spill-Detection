import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Paso 1: Cargar los datos
data_folder = "./Dataset"
image_folder = os.path.join(data_folder, "train_images_256")
mask_folder = os.path.join(data_folder, "train_masks_256")

# Obtener lista de nombres de archivo para las imágenes
image_files = os.listdir(image_folder)


# Paso 2: Preparar los datos
# Redimensionar todas las imágenes a un tamaño específico
image_size = (256, 256)

# Crear listas para almacenar las imágenes y sus máscaras
images = []
masks = []


# filename = "041871.000163.tif"
# image_path = os.path.join(image_folder, filename)
# mask_path = os.path.join(mask_folder, filename)
# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
# Redimensionar las imágenes y máscaras al tamaño especificado
# image = cv2.resize(image, image_size)
# mask = cv2.resize(mask, image_size)
# Normalizar las imágenes y máscaras para que los valores estén entre 0 y 1
# image = image / 255.0
# mask = mask / 255.0
# Cambiar los píxeles en la máscara mayores que 0 a 255
# mask[mask > 0] = 255
# print("mask")
# for row in mask:
#     for pixel in row:
#         print(pixel, end="\t")
#     print()
# window_name = 'image'
# cv2.imshow(window_name, mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# Cargar las imágenes y sus máscaras
for filename in image_files:
    image_path = os.path.join(image_folder, filename)
    mask_path = os.path.join(mask_folder, filename)

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Redimensionar las imágenes y máscaras al tamaño especificado
    image = cv2.resize(image, image_size)
    mask = cv2.resize(mask, image_size)

    # Normalizar las imágenes y máscaras para que los valores estén entre 0 y 1
    image = image / 255.0
    mask = mask / 255.0

    # Cambiar los píxeles en la máscara mayores que 0 a 255
    mask[mask > 0] = 255

    images.append(image.flatten())  # Aplanar la imagen y agregar al conjunto de imágenes
    masks.append(mask.flatten())  # Aplanar la máscara y agregar al conjunto de máscaras

# Convertir las listas en arrays numpy
X = np.array(images)
y = np.array(masks)


# Paso 3: Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Paso 4: Entrenar el SVM
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Paso 5: Evaluar el SVM
# y_pred = svm.predict(X_test)

# Calcular la precisión
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Precisión del SVM: {accuracy}")

# Obtener el informe de clasificación
# report = classification_report(y_test, y_pred)
# print("Informe de clasificación:")
# print(report)

# Paso 6: Predicciones sobre nuevas imágenes (opcional)
# Puedes utilizar el modelo SVM entrenado para hacer predicciones en nuevas imágenes segmentadas.

# Por ejemplo, para predecir en una nueva imagen llamada "nueva_imagen.tif":
# nueva_imagen = cv2.imread("ruta_a_nueva_imagen.tif", cv2.IMREAD_GRAYSCALE)
# nueva_imagen = cv2.resize(nueva_imagen, image_size)
# nueva_imagen = nueva_imagen / 255.0  # Normalizar la imagen
# nueva_imagen_flattened = nueva_imagen.flatten()
# prediccion = svm.predict([nueva_imagen_flattened])
# mascar_petróleo = prediccion.reshape(image_size)
