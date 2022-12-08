#librerias
import cv2
import numpy as mp
import os

#importamos los datos del repositorio
direccion = 'C:/Users/david/Desktop/nuevo/datos'
lista = os.listdir(direccion)

etiquetas = []
rostro = []
contador = 0

for nameDir in lista:
    #leer las fotos y rostros
    nombre = direccion + '/' + nameDir

    for fileName in os.listdir(nombre):
        etiquetas.append(contador)
        rostro.append(cv2.imread(nombre + '/'+fileName,0))

    contador = contador + 1

# modelo
reconocimiento= cv2.face.LBPHFaceRecognizer_create()

# entrenamiento del modelo
reconocimiento.train(rostro, mp.array(etiquetas))

# guardamos el modelo
reconocimiento.write("modelo.xml")