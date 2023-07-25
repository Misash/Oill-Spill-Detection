import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, classification_report

# # Establecer el dispositivo GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Cambiar "0" por el número de tu GPU
#
# # Restringir el crecimiento de la memoria de la GPU para evitar que ocupe toda la memoria
# physical_devices = tf.config.list_physical_devices("GPU")
# if len(physical_devices) > 0:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Paso 1: Cargar los datos
data_folder = "./Dataset"
image_folder = os.path.join(data_folder, "train_images_256")
mask_folder = os.path.join(data_folder, "train_masks_256")

# Obtener lista de nombres de archivo para las imágenes
image_files = os.listdir(image_folder)

# image_files = os.listdir(image_folder)
# half_length = len(image_files) // 2
# image_files = image_files[:half_length]


# Paso 2: Preparar los datos
# Redimensionar todas las imágenes a un tamaño específico
image_size = (256, 256)

# Crear listas para almacenar las imágenes y sus máscaras
images = []
masks = []

# cargar imagenes y mascaras
for filename in image_files:
    image_path = os.path.join(image_folder, filename)
    mask_path = os.path.join(mask_folder, filename)

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # image = cv2.resize(image, image_size)
    # mask = cv2.resize(mask, image_size)

    # Normalizar las imágenes y máscaras para que los valores estén entre 0 y 1
    image = image / 255.0
    mask = mask / 255.0

    # Cambiar los píxeles en la máscara mayores que 0 a 255
    mask[mask > 0] = 255

    images.append(image)
    masks.append(mask)



# Convertir las listas de imágenes y máscaras a arreglos numpy
images = np.array(images)
masks = np.array(masks)

# Agregar una dimensión adicional para el canal (ya que las imágenes están en escala de grises)
images = np.expand_dims(images, axis=-1)
masks = np.expand_dims(masks, axis=-1)





# Paso 3: Implementar el modelo U-Net
def unet(input_size=(256, 256, 1)):
    inputs = Input(input_size)

    # Contracción: encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)

    # Expansión: decoder
    up6 = UpSampling2D(size=(2, 2))(conv5)
    up6 = Conv2D(512, 2, activation='relu', padding='same')(up6)
    merge6 = concatenate([conv4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = Conv2D(256, 2, activation='relu', padding='same')(up7)
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = UpSampling2D(size=(2, 2))(conv7)
    up8 = Conv2D(128, 2, activation='relu', padding='same')(up8)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = UpSampling2D(size=(2, 2))(conv8)
    up9 = Conv2D(64, 2, activation='relu', padding='same')(up9)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    return model


# Paso 4: Compilar y entrenar el modelo
model = unet(input_size=image_size + (1,))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(images, masks, batch_size=8, epochs=1, validation_split=0.2)

# Paso 5: Usar el modelo para predecir
# Aquí puedes cargar una nueva imagen en escala de grises, redimensionarla y normalizarla de la misma manera que en el paso 2.
# Luego, utiliza el modelo entrenado para hacer una predicción en la imagen y obtener la máscara resultante.

# ejemplo de predicción
test_image = cv2.imread("./Dataset/train_images_256/041871.000163.tif", cv2.IMREAD_GRAYSCALE)
test_image = cv2.resize(test_image, image_size)
test_image = test_image.astype('float32') / 255.0
test_image = np.expand_dims(test_image, axis=0)
test_image = np.expand_dims(test_image, axis=-1)

predicted_mask = model.predict(test_image)

# Mostrar la imagen testeada
plt.imshow(test_image[0, :, :, 0], cmap='gray')
plt.title('Imagen testeada')
plt.axis('off')
plt.show()


# Mostrar la máscara predicha
predicted_mask = predicted_mask.squeeze()
plt.imshow(predicted_mask, cmap='gray')
plt.title('Máscara predicha')
plt.axis('off')
plt.show()


##################################### SVM ##################################

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
