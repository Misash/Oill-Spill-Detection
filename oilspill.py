from osgeo import gdal
import cv2
import numpy as np
import sys
import os
import tempfile
import tensorflow as tf
import argparse
import progressbar


PIXEL_SIZE = 256

# management errors 
def gdal_error_handler(err_level, err_no, err_msg):

    # incorrect format
    if("not recognized as a supported file format." in err_msg):
        print("[x] El formato del archivo seleccionado es incorrecto.")

    # file dont found
    if("No such file or directory" in err_msg):
        print("[x] El archivo seleccionado no existe.")
        
gdal.PushErrorHandler(gdal_error_handler)




def gray_scale(r, g, b):

    # luminus method
    return (0.299*r + 0.587*g + 0.114*b)

def save_tiff(gray_image, path, mod):

    driver = gdal.GetDriverByName('GTiff')
    rows, cols = gray_image.shape
    ind = path.rfind(".")
    fname = path[:ind] + mod + path[ind:]
    
    output_ds = driver.Create(fname, cols, rows, 1, gdal.GDT_Byte)
    output_ds.GetRasterBand(1).WriteArray(gray_image)
    output_ds = None


def add_white_pixels(image, n_rows, n_cols):

    height, width = image.shape

    new_image = np.full((height + n_rows, width + n_cols), 255, dtype=image.dtype)

    new_image[:height, :width] = image

    return new_image


def process_CNN(image, model_path):

    # loading model
    model = tf.keras.models.load_model(model_path)

    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)

    image = model.predict(image)
    return image.squeeze()


def init_image(input_path, output_path, model_path):

    # open real image
    ds = gdal.Open(input_path)

    if ds is None:
        sys.exit(1)  # exit to the program    

    print("[+] Imagen abierta.")

    # obtaining rgb bands 
    if ds.RasterCount < 3: 
        print("[!] El archivo no contine las 3 bandas minimas, contiene:", ds.RasterCount,".")
        sys.exit(1)

    # Obtaining directions
    red = ds.GetRasterBand(1)
    green = ds.GetRasterBand(2)
    blue = ds.GetRasterBand(3)

    # pixel count by length and width
    rows, cols = red.XSize, red.YSize
    int_r, int_c = rows // PIXEL_SIZE, cols // PIXEL_SIZE
    
    # count to rest_pixel is grant to PIXEL_SIZE
    rest_r, rest_c = rows % PIXEL_SIZE, cols % PIXEL_SIZE

    # create temporal folder
    path_temp_folder = tempfile.mkdtemp()
    basename = os.path.basename(input_path)    

    # partition images
    ind = 1 
    
    add_pixel = False
    redimention_list = [] 

    if rest_r or rest_c:
        add_pixel = True

    #barra de carga
    total_iterations  = int_r * int_c

    print("[+] Procesando imagen\n")
    print("[+] Particionando y detectando")
    bar = progressbar.ProgressBar(max_value=total_iterations, widgets=[
    ' [', progressbar.Percentage(), '] ',
    progressbar.Bar(), ' (', progressbar.SimpleProgress(), ') ',
    ])


    for r in range(int_r):
        for c in range(int_c):
            
            r_part = red.ReadAsArray(r*PIXEL_SIZE, c*PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE)
            g_part = green.ReadAsArray(r*PIXEL_SIZE, c*PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE)
            b_part = blue.ReadAsArray(r*PIXEL_SIZE, c*PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE)

            # to gray scale
            gray = gray_scale(r_part, g_part, b_part)

            # add white pixels if is necesary
            if(add_pixel):
                if (r + 1 == int_r or c + 1 == int_c):
                    r_p_gray, c_p_gray = gray.shape
                    r_p_gray = r_p_gray % PIXEL_SIZE
                    c_p_gray = c_p_gray % PIXEL_SIZE
                    compl_r = 256 - r_p_gray if r_p_gray else 0
                    compl_c = 256 - c_p_gray if c_p_gray else 0
                    gray = add_white_pixels(gray, compl_r, compl_c)
                    redimention_list.append(ind)

            # process with CNN
            gray = process_CNN(gray, model_path)

            gray = (gray * 512).astype(np.uint16)

            # save  
            save_tiff(gray, path_temp_folder + "/" + basename, "_part"+str(ind))

            #increment bar 
            bar.update(ind)
            #increment count
            ind +=1

    print("\n")
    # join images

    driver = gdal.GetDriverByName('GTiff')

    mask = driver.Create(output_path, rows, cols, 1, gdal.GDT_Byte)
    
    ind = 1

    i_point = basename.rfind(".")
    
    if(i_point == -1):
        i_point = len(i_point)

    name = basename[:i_point]
    ext = basename[i_point:]

    

    print("[+] recreadno mascara con detecciones")
    bar = progressbar.ProgressBar(max_value=total_iterations, widgets=[
    ' [', progressbar.Percentage(), '] ',
    progressbar.Bar(), ' (', progressbar.SimpleProgress(), ') ',
    ])

    for r in range(int_r):
        for c in range(int_c):

            # open parts
            ds = gdal.Open(path_temp_folder+"/"+name+"_part"+str(ind)+ext)
            part = ds.GetRasterBand(1).ReadAsArray()

            # redimension if is necesary
            if(ind in redimention_list):
                if(r == int_r - 1):
                    part = part[:rows,:]
                if(c == int_c - 1):
                    part = part[:,:cols]

            # mask images
            mask.GetRasterBand(1).WriteArray(part, r*PIXEL_SIZE, c*PIXEL_SIZE)
 
            #increment count
            bar.update(ind)
            ind +=1
            
    print("\n")

    # delete all files in temporal folder
    for file in os.listdir(path_temp_folder):
        ruta_archivo_temporal = os.path.join(path_temp_folder, file)
        os.remove(ruta_archivo_temporal)

    print("[+] Archivo generado.")

def main():

    parser = argparse.ArgumentParser(description='\n\nReconocimiento de hidrocarburos en el mar y costas\n Ejemplo app -I imagen.tif -O mascara.tif -M modelo.h5')

    # Definir un argumento opcional con valor predeterminado
    parser.add_argument("-I", "--input_image", dest = "input_path", help="Enter a input image in GeoTiff format '.tif' o '.tiff'")
    parser.add_argument("-O", "--output_mask", dest = "output_path", help="Enter a output mask in GeoTiff format '.tif' o '.tiff'")
    parser.add_argument("-M", "--model", dest = "model_path", help="Enter a model path of CNN in Unet architecture '.h5'")

    args = parser.parse_args()

    i_path = args.input_path
    o_path = args.output_path
    model_path = args.model_path

    print(i_path, o_path, model_path)

    if(i_path and o_path and model_path):
        init_image(i_path, o_path, model_path)
    else:
        print("[!] Necesitas ingresar los parametros correctos")
        print("[?] Para ver su uso utiliza el parametro '-h' o '--help'")


if __name__ == "__main__":
    main()