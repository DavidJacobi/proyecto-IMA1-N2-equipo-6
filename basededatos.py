# librerias
import os
import cv2
import mediapipe as mp

#carpetas repositorios de datos
nombre = '1736243 no tiene cubrebocas'
direccion = 'C:/Users/david/Desktop/nuevo/datos'
carpeta = direccion + '/' + nombre

if not os.path.exists(carpeta):
    print("repositorio creado")
    os.makedirs(carpeta)

#contador
contador = 0
#declaracion del detector
detector = mp.solutions.face_detection
dibujo = mp.solutions.drawing_utils

#video captura
captura = cv2.VideoCapture(0)

#parametros de deteccion
with detector.FaceDetection(min_detection_confidence=0.75) as rostros:

    while True:
        #video captura
        ret, frame = captura.read()

        #frame espejo
        frame = cv2.flip(frame, 1)

        #color
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #resultado deteccion de rostro
        resultado = rostros.process(rgb)

        #filtro
        if resultado.detections is not None:
            for rostro in resultado.detections:
                #dibujo.draw_detection(frame,rostro)

                #ancho y alto de la ventana
                al, an, dat = frame.shape

                #extraer xinicial y yinicial
                xi = rostro.location_data.relative_bounding_box.xmin
                yi = rostro.location_data.relative_bounding_box.ymin

                #extraer el ancho y el alto
                ancho = rostro.location_data.relative_bounding_box.width
                alto = rostro.location_data.relative_bounding_box.height

                #conversion a pixel
                xi = int(xi * an)
                yi = int(yi * al)
                ancho = int(ancho * an)
                alto = int(alto * al)

                #xfinal yfinal
                xf = xi + ancho
                yf = yi + alto

                #extraccion de pixeles
                cara = frame[yi:yf, xi:xf]

                cara = cv2.resize(cara, (150,200), interpolation=cv2.INTER_CUBIC)

                #almacenar imagenes
                cv2.imwrite(carpeta + "/rostro_{}.jpg".format(contador), cara)
                contador = contador + 1




        #fotogramas
        cv2.imshow("reconocimiento facial y de tapabocas", frame)
        #leyendo tecla
        t = cv2.waitKey(1)
        if t == 27 or contador == 500:
            break

captura.release()
cv2.destroyAllWindows()